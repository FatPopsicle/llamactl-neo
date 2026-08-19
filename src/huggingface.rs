


use anyhow::{Context, Result, anyhow, bail};
use regex::Regex;
use reqwest::{
    Url,
    blocking::{Client, Response},
    header::{AUTHORIZATION, CONTENT_RANGE, RANGE, USER_AGENT},
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const ENDPOINT: &str = "https://huggingface.co";
const REVISION: &str = "main";
const MULTIPART_THRESHOLD: u64 = 256 * 1024 * 1024;
const CONNECTIONS_PER_FILE: usize = 8;
const RETRIES: usize = 4;
const COPY_BUFFER_SIZE: usize = 1024 * 1024;

static SHARD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(.*)-(\d{5})-of-(\d{5})\.gguf$").unwrap());
static QUANT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[2-8]_(?:[01]|K(?:_(?:XXL|XL|L|M|S))?)|F(?:16|32)|BF16)",
    )
    .unwrap()
});
static HTML_COMMENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<!--.*?-->").unwrap());
static HTML_UNSAFE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?is)<(?:script|style)(?:\s[^>]*)?>.*?</(?:script|style)\s*>").unwrap()
});
static HTML_IMAGE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<(?:img|source)(?:\s[^>]*)?/?>").unwrap());
static MARKDOWN_IMAGE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!\[[^\]]*\]\([^)]*\)").unwrap());
static HTML_HEADING_OPEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<h[1-6](?:\s[^>]*)?>").unwrap());
static HTML_HEADING_CLOSE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)</h[1-6]\s*>").unwrap());
static HTML_BLOCK_OPEN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)<(?:p|div|section|article|header|footer|table|tr|ul|ol|details|pre)(?:\s[^>]*)?>",
    )
    .unwrap()
});
static HTML_BLOCK_CLOSE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)</(?:p|div|section|article|header|footer|table|tr|ul|ol|details|summary|pre)\s*>",
    )
    .unwrap()
});
static HTML_BREAK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<br(?:\s[^>]*)?/?>").unwrap());
static HTML_LIST_ITEM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<li(?:\s[^>]*)?>").unwrap());
static HTML_CELL_CLOSE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)</(?:td|th)\s*>").unwrap());
static HTML_SUMMARY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<summary(?:\s[^>]*)?>").unwrap());
static HTML_STRONG_OPEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<(?:strong|b)(?:\s[^>]*)?>").unwrap());
static HTML_STRONG_CLOSE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)</(?:strong|b)\s*>").unwrap());
static HTML_CODE_OPEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<code(?:\s[^>]*)?>").unwrap());
static HTML_CODE_CLOSE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)</code\s*>").unwrap());
static HTML_TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<[^>]+>").unwrap());

#[derive(Debug, Clone)]
pub struct TemplateHit {
    pub id: String,
    pub template: String,
    pub downloads: u64,
    pub likes: u64,
}

#[derive(Debug, Clone)]
pub struct Repository {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    pub license: String,
    pub updated: String,
}

#[derive(Debug, Clone)]
pub struct ModelDetails {
    pub id: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub license: String,
    pub updated: String,
    pub task: String,
    pub library: String,
    pub base_model: String,
    pub languages: Vec<String>,
    pub tags: Vec<String>,
    pub url: String,
    pub readme: String,
}

#[derive(Debug, Clone)]
pub struct RemoteFile {
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Artifact {
    pub label: String,
    pub description: String,
    pub quality: u8,
    pub recommended: bool,
    pub files: Vec<RemoteFile>,
    pub size: u64,
    pub shard_count: usize,
    pub has_mmproj: bool,
    pub complete: bool,
}

#[derive(Debug)]
pub enum DownloadEvent {
    FileStarted {
        path: String,
        total: u64,
    },
    FileProgress {
        path: String,
        downloaded: u64,
        total: u64,
    },
    Verifying {
        path: String,
    },
    Retrying {
        path: String,
        attempt: usize,
        message: String,
    },
    FileDone {
        path: String,
        skipped: bool,
    },
    Finished(std::result::Result<DownloadSummary, String>),
}

#[derive(Debug)]
pub struct DownloadSummary {
    pub downloaded: usize,
    pub skipped: usize,
    pub destination: PathBuf,
}

pub struct DownloadHandle {
    pub events: Receiver<DownloadEvent>,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Deserialize)]
struct ApiRepository {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    gated: Value,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, rename = "lastModified")]
    last_modified: String,
    #[serde(default)]
    author: String,
    #[serde(default)]
    pipeline_tag: String,
    #[serde(default)]
    library_name: String,
    #[serde(default, rename = "cardData")]
    card_data: Value,
}

#[derive(Deserialize)]
struct TreeNode {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    sha256: String,
    lfs: Option<LfsInfo>,
}

#[derive(Deserialize)]
struct LfsInfo {
    #[serde(default)]
    oid: String,
    #[serde(default)]
    sha256: String,
    #[serde(default)]
    size: u64,
}

pub fn search(query: &str) -> Result<Vec<Repository>> {
    search_repos(query, Some("gguf"))
}


pub fn search_templates(query: &str) -> Result<Vec<TemplateHit>> {
    let repositories = search_repos(query, None)?;
    let mut hits = Vec::new();
    for repository in repositories {
        match fetch_chat_template(&repository.id) {
            Ok(Some(template)) => hits.push(TemplateHit {
                id: repository.id,
                template,
                downloads: repository.downloads,
                likes: repository.likes,
            }),
            Ok(None) => {}
            Err(_) => {}
        }
    }
    Ok(hits)
}

fn search_repos(query: &str, filter: Option<&str>) -> Result<Vec<Repository>> {
    let mut pairs = vec![
        ("search", query.trim()),
        ("sort", "likes"),
        ("direction", "-1"),
        ("limit", "25"),
        ("full", "true"),
    ];
    if let Some(filter) = filter {
        pairs.push(("filter", filter));
    }
    let response = api_client()?
        .get(format!("{ENDPOINT}/api/models"))
        .query(&pairs)
        .send()
        .context("search Hugging Face")?;
    let response = checked_response(response, "search Hugging Face")?;
    let repositories: Vec<ApiRepository> = response.json().context("decode Hugging Face search")?;
    Ok(repositories
        .into_iter()
        .filter(|repo| !repo.private && is_not_gated(&repo.gated) && valid_repo_id(&repo.id))
        .map(|repo| Repository {
            id: repo.id,
            downloads: repo.downloads,
            likes: repo.likes,
            license: repo
                .tags
                .iter()
                .find_map(|tag| tag.strip_prefix("license:"))
                .unwrap_or("unknown")
                .to_owned(),
            updated: repo
                .last_modified
                .split('T')
                .next()
                .unwrap_or("unknown")
                .to_owned(),
        })
        .collect())
}

pub fn fetch_chat_template(repo: &str) -> Result<Option<String>> {
    validate_repo_id(repo)?;


    let response = api_client()?
        .get(api_url(repo, &[])?)
        .send()
        .with_context(|| format!("read model metadata for {repo}"))?;
    let response = checked_response(response, &format!("read model metadata for {repo}"))?;
    let metadata: Value = response
        .json()
        .with_context(|| format!("decode model metadata for {repo}"))?;
    if let Some(config) = metadata.get("config") {
        for key in ["chat_template_jinja", "chat_template"] {
            if let Some(template) = config.get(key).and_then(Value::as_str)
                && !template.trim().is_empty() {
                    return Ok(Some(template.to_owned()));
                }
        }
    }


    fetch_tokenizer_chat_template(repo)
}

fn fetch_tokenizer_chat_template(repo: &str) -> Result<Option<String>> {
    let (owner, name) = split_repo(repo)?;
    let mut url = Url::parse(ENDPOINT)?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("invalid Hugging Face endpoint"))?;
        segments.extend([owner, name, "raw", REVISION, "tokenizer_config.json"]);
    }
    let response = api_client()?
        .get(url)
        .send()
        .with_context(|| format!("read tokenizer config for {repo}"))?;
    let response = checked_response(response, &format!("read tokenizer config for {repo}"))?;
    let config: Value = response
        .json()
        .with_context(|| format!("decode tokenizer config for {repo}"))?;
    Ok(config
        .get("chat_template")
        .and_then(Value::as_str)
        .filter(|template| !template.trim().is_empty())
        .map(str::to_owned))
}

pub fn details(repo: &str) -> Result<ModelDetails> {
    validate_repo_id(repo)?;
    let response = api_client()?
        .get(api_url(repo, &[])?)
        .send()
        .with_context(|| format!("read model details for {repo}"))?;
    let response = checked_response(response, &format!("read model details for {repo}"))?;
    let metadata: ApiRepository = response
        .json()
        .with_context(|| format!("decode model details for {repo}"))?;

    let license = card_string(&metadata.card_data, "license")
        .or_else(|| {
            metadata
                .tags
                .iter()
                .find_map(|tag| tag.strip_prefix("license:"))
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "unknown".into());
    let base_model = card_strings(&metadata.card_data, "base_model").join(", ");
    let languages = card_strings(&metadata.card_data, "language");
    let tags = metadata
        .tags
        .iter()
        .filter(|tag| {
            !tag.starts_with("license:")
                && !tag.starts_with("base_model:")
                && !tag.starts_with("region:")
                && *tag != "gguf"
        })
        .take(16)
        .cloned()
        .collect::<Vec<_>>();
    let readme = fetch_readme(repo)
        .unwrap_or_else(|error| format!("Model card README unavailable: {error:#}"));
    Ok(ModelDetails {
        id: metadata.id,
        author: if metadata.author.is_empty() {
            repo.split('/').next().unwrap_or("unknown").into()
        } else {
            metadata.author
        },
        downloads: metadata.downloads,
        likes: metadata.likes,
        license,
        updated: metadata
            .last_modified
            .split('T')
            .next()
            .unwrap_or("unknown")
            .into(),
        task: if metadata.pipeline_tag.is_empty() {
            card_string(&metadata.card_data, "pipeline_tag").unwrap_or_else(|| "unknown".into())
        } else {
            metadata.pipeline_tag
        },
        library: if metadata.library_name.is_empty() {
            "unknown".into()
        } else {
            metadata.library_name
        },
        base_model: if base_model.is_empty() {
            "unknown".into()
        } else {
            base_model
        },
        languages,
        tags,
        url: format!("{ENDPOINT}/{repo}"),
        readme,
    })
}

pub fn artifacts(repo: &str) -> Result<Vec<Artifact>> {
    validate_repo_id(repo)?;
    let nodes = repository_tree(repo)?;
    let mut models = Vec::new();
    let mut projectors = Vec::new();
    for node in nodes {
        if node.kind != "file" && node.kind != "blob" {
            continue;
        }
        ensure_safe_relative_path(&node.path)?;
        if !node.path.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }
        let size = node
            .lfs
            .as_ref()
            .map(|lfs| lfs.size)
            .filter(|size| *size > 0)
            .unwrap_or(node.size);
        let sha256 = if !node.sha256.is_empty() {
            Some(node.sha256)
        } else {
            node.lfs.as_ref().and_then(|lfs| {
                if !lfs.sha256.is_empty() {
                    Some(lfs.sha256.clone())
                } else if !lfs.oid.is_empty() {
                    Some(lfs.oid.clone())
                } else {
                    None
                }
            })
        };
        let file = RemoteFile {
            path: node.path,
            size,
            sha256,
        };
        if is_mmproj(&file.path) {
            projectors.push(file);
        } else {
            models.push(file);
        }
    }

    let projector = preferred_mmproj(&projectors).cloned();
    let mut groups = BTreeMap::<String, Vec<RemoteFile>>::new();
    for file in models {
        groups
            .entry(shard_group_key(&file.path))
            .or_default()
            .push(file);
    }

    let mut output = groups
        .into_values()
        .map(|mut files| {
            files.sort_by(|left, right| left.path.cmp(&right.path));
            let expected_shards = files
                .first()
                .and_then(|file| shard_parts(&file.path))
                .map(|(_, total)| total)
                .unwrap_or(1);
            let complete = files.len() == expected_shards;
            let quant = files
                .first()
                .and_then(|file| quantization(&file.path))
                .unwrap_or_else(|| "UNKNOWN".into());
            let mut all_files = files;
            let has_mmproj = projector.is_some();
            if let Some(projector) = &projector {
                all_files.push(projector.clone());
            }
            let size = all_files.iter().map(|file| file.size).sum();
            Artifact {
                label: quant.clone(),
                description: quant_description(&quant).into(),
                quality: quant_quality(&quant),
                recommended: quant == "Q4_K_M",
                files: all_files,
                size,
                shard_count: expected_shards,
                has_mmproj,
                complete,
            }
        })
        .collect::<Vec<_>>();
    output.sort_by(|left, right| {
        right
            .recommended
            .cmp(&left.recommended)
            .then(right.quality.cmp(&left.quality))
            .then(left.size.cmp(&right.size))
            .then(left.label.cmp(&right.label))
    });
    Ok(output)
}

pub fn spawn_download(
    repo: String,
    files: Vec<RemoteFile>,
    destination: PathBuf,
) -> DownloadHandle {
    let (tx, events) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancel.clone();
    thread::spawn(move || {
        let result = run_download(&repo, &files, &destination, &worker_cancel, &tx)
            .map_err(|error| format!("{error:#}"));
        let _ = tx.send(DownloadEvent::Finished(result));
    });
    DownloadHandle { events, cancel }
}

fn run_download(
    repo: &str,
    files: &[RemoteFile],
    destination: &Path,
    cancel: &AtomicBool,
    events: &Sender<DownloadEvent>,
) -> Result<DownloadSummary> {
    validate_repo_id(repo)?;
    fs::create_dir_all(destination).with_context(|| format!("create {}", destination.display()))?;
    let client = download_client()?;
    let token = std::env::var("HF_TOKEN")
        .ok()
        .filter(|token| !token.is_empty());
    let mut downloaded = 0;
    let mut skipped = 0;
    for remote in files {
        check_cancel(cancel)?;
        ensure_safe_relative_path(&remote.path)?;
        let target = destination.join(Path::new(&remote.path));
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        match download_file(
            &client,
            token.as_deref(),
            repo,
            remote,
            &target,
            cancel,
            events,
        )? {
            true => skipped += 1,
            false => downloaded += 1,
        }
    }
    Ok(DownloadSummary {
        downloaded,
        skipped,
        destination: destination.to_owned(),
    })
}

fn download_file(
    client: &Client,
    token: Option<&str>,
    repo: &str,
    remote: &RemoteFile,
    target: &Path,
    cancel: &AtomicBool,
    events: &Sender<DownloadEvent>,
) -> Result<bool> {
    let _ = events.send(DownloadEvent::FileStarted {
        path: remote.path.clone(),
        total: remote.size,
    });

    if target.is_file() && file_matches(target, remote, cancel, events)? {
        let _ = events.send(DownloadEvent::FileProgress {
            path: remote.path.clone(),
            downloaded: remote.size,
            total: remote.size,
        });
        let _ = events.send(DownloadEvent::FileDone {
            path: remote.path.clone(),
            skipped: true,
        });
        return Ok(true);
    }

    let partial = suffixed_path(target, ".part");
    if partial.is_file()
        && fs::metadata(&partial)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
            == remote.size
    {
        let _ = events.send(DownloadEvent::FileProgress {
            path: remote.path.clone(),
            downloaded: remote.size,
            total: remote.size,
        });
        match verify_file(&partial, remote, cancel, events) {
            Ok(()) => {
                fs::rename(&partial, target)
                    .with_context(|| format!("finalize {}", target.display()))?;
                let _ = events.send(DownloadEvent::FileDone {
                    path: remote.path.clone(),
                    skipped: false,
                });
                return Ok(false);
            }
            Err(error) if cancel.load(Ordering::Relaxed) => return Err(error),
            Err(_) => {


                fs::remove_file(&partial)
                    .with_context(|| format!("remove corrupt {}", partial.display()))?;
                for index in 0..CONNECTIONS_PER_FILE {
                    let _ = fs::remove_file(suffixed_path(target, &format!(".part-{index:02}")));
                }
            }
        }
    }

    let url = resolve_url(repo, &remote.path)?;
    let chunk_files = if remote.size >= MULTIPART_THRESHOLD && remote.sha256.is_some() {
        download_multipart(client, token, &url, remote, target, cancel, events)?
    } else {
        download_single(client, token, &url, remote, &partial, cancel, events)?;
        Vec::new()
    };

    if fs::metadata(&partial)
        .with_context(|| format!("inspect {}", partial.display()))?
        .len()
        != remote.size
    {
        bail!("downloaded size does not match for {}", remote.path)
    }
    if let Err(error) = verify_file(&partial, remote, cancel, events) {
        let _ = fs::remove_file(&partial);
        for chunk in &chunk_files {
            let _ = fs::remove_file(chunk);
        }
        return Err(error);
    }
    fs::rename(&partial, target).with_context(|| format!("finalize {}", target.display()))?;
    for chunk in chunk_files {
        let _ = fs::remove_file(chunk);
    }
    let _ = events.send(DownloadEvent::FileDone {
        path: remote.path.clone(),
        skipped: false,
    });
    Ok(false)
}

fn download_single(
    client: &Client,
    token: Option<&str>,
    url: &Url,
    remote: &RemoteFile,
    partial: &Path,
    cancel: &AtomicBool,
    events: &Sender<DownloadEvent>,
) -> Result<()> {
    if fs::metadata(partial)
        .map(|metadata| metadata.len())
        .unwrap_or(0)
        > remote.size
    {
        File::create(partial).with_context(|| format!("truncate {}", partial.display()))?;
    }
    let resumed = fs::metadata(partial)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let _ = events.send(DownloadEvent::FileProgress {
        path: remote.path.clone(),
        downloaded: resumed,
        total: remote.size,
    });
    let mut last_error = None;
    for attempt in 0..=RETRIES {
        check_cancel(cancel)?;
        let position = fs::metadata(partial)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        if position == remote.size {
            return Ok(());
        }
        let result = (|| -> Result<()> {
            let mut request = request(client, token, url.clone());
            if position > 0 {
                request = request.header(RANGE, format!("bytes={position}-{}", remote.size - 1));
            }
            let mut response = request.send().with_context(|| format!("GET {url}"))?;
            let status = response.status();
            let restart = position > 0 && status.as_u16() == 200;
            if !status.is_success() {
                bail!("GET {url}: HTTP {status}")
            }
            if position > 0 && !restart && status.as_u16() != 206 {
                bail!("GET {url}: expected HTTP 206, got {status}")
            }
            let mut file = if restart {
                File::create(partial)?
            } else {
                OpenOptions::new().create(true).append(true).open(partial)?
            };
            let initial = if restart { 0 } else { position };
            stream_response(
                &mut response,
                &mut file,
                initial,
                remote,
                cancel,
                events,
                None,
            )?;
            file.flush()?;
            let actual = fs::metadata(partial)?.len();
            if actual != remote.size {
                bail!("connection ended at {actual} of {} bytes", remote.size)
            }
            Ok(())
        })();
        match result {
            Ok(()) => return Ok(()),
            Err(error) if cancel.load(Ordering::Relaxed) => return Err(error),
            Err(error) => {
                last_error = Some(error);
                if attempt < RETRIES {
                    let _ = events.send(DownloadEvent::Retrying {
                        path: remote.path.clone(),
                        attempt: attempt + 1,
                        message: last_error.as_ref().unwrap().to_string(),
                    });
                    cancellable_sleep(cancel, retry_delay(attempt))?;
                }
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("download failed")))
}

fn download_multipart(
    client: &Client,
    token: Option<&str>,
    url: &Url,
    remote: &RemoteFile,
    target: &Path,
    cancel: &AtomicBool,
    events: &Sender<DownloadEvent>,
) -> Result<Vec<PathBuf>> {
    let connections = CONNECTIONS_PER_FILE.min(remote.size.max(1) as usize);
    let chunk_size = remote.size.div_ceil(connections as u64);
    let chunks = (0..connections)
        .filter_map(|index| {
            let start = index as u64 * chunk_size;
            (start < remote.size).then(|| {
                let end = (start + chunk_size - 1).min(remote.size - 1);
                (
                    index,
                    start,
                    end,
                    suffixed_path(target, &format!(".part-{index:02}")),
                )
            })
        })
        .collect::<Vec<_>>();

    let initial = chunks
        .iter()
        .map(|(_, start, end, path)| {
            let expected = end - start + 1;
            let length = fs::metadata(path)
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            if length > expected {
                let _ = File::create(path);
                0
            } else {
                length
            }
        })
        .sum::<u64>();
    let progress = Arc::new(AtomicU64::new(initial));
    let failed = Arc::new(AtomicBool::new(false));
    let _ = events.send(DownloadEvent::FileProgress {
        path: remote.path.clone(),
        downloaded: initial,
        total: remote.size,
    });

    let results = thread::scope(|scope| {
        let mut handles = Vec::new();
        for (_, start, end, path) in &chunks {
            let client = client.clone();
            let url = url.clone();
            let remote = remote.clone();
            let path = path.clone();
            let progress = progress.clone();
            let failed = failed.clone();
            let events = events.clone();
            handles.push(scope.spawn(move || {
                let result = download_range_part(
                    &client, token, &url, &remote, &path, *start, *end, cancel, &failed, &progress,
                    &events,
                );
                if result.is_err() {
                    failed.store(true, Ordering::Relaxed);
                }
                result
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow!("download worker panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    });
    results?;
    check_cancel(cancel)?;

    let partial = suffixed_path(target, ".part");
    let mut output =
        File::create(&partial).with_context(|| format!("assemble {}", partial.display()))?;
    let mut buffer = vec![0u8; COPY_BUFFER_SIZE];
    for (_, _, _, chunk) in &chunks {
        let mut input = File::open(chunk).with_context(|| format!("open {}", chunk.display()))?;
        loop {
            check_cancel(cancel)?;
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            output.write_all(&buffer[..read])?;
        }
    }
    output.flush()?;
    Ok(chunks.into_iter().map(|(_, _, _, path)| path).collect())
}

#[allow(clippy::too_many_arguments)]
fn download_range_part(
    client: &Client,
    token: Option<&str>,
    url: &Url,
    remote: &RemoteFile,
    part: &Path,
    start: u64,
    end: u64,
    cancel: &AtomicBool,
    failed: &AtomicBool,
    progress: &AtomicU64,
    events: &Sender<DownloadEvent>,
) -> Result<()> {
    let expected = end - start + 1;
    let mut last_error = None;
    for attempt in 0..=RETRIES {
        check_stopped(cancel, failed)?;
        let position = fs::metadata(part)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        if position == expected {
            return Ok(());
        }
        let result = (|| -> Result<()> {
            let mut response = request(client, token, url.clone())
                .header(RANGE, format!("bytes={}-{}", start + position, end))
                .send()
                .with_context(|| format!("GET range {}-{}", start + position, end))?;
            if response.status().as_u16() != 206 {
                bail!("range request returned HTTP {}", response.status())
            }
            if let Some(content_range) = response.headers().get(CONTENT_RANGE)
                && !content_range
                    .to_str()
                    .unwrap_or_default()
                    .starts_with(&format!("bytes {}-", start + position))
            {
                bail!("server returned an unexpected Content-Range")
            }
            let mut output = OpenOptions::new().create(true).append(true).open(part)?;
            stream_response(
                &mut response,
                &mut output,
                position,
                remote,
                cancel,
                events,
                Some((failed, progress)),
            )?;
            output.flush()?;
            let actual = fs::metadata(part)?.len();
            if actual != expected {
                bail!("range ended at {actual} of {expected} bytes")
            }
            Ok(())
        })();
        match result {
            Ok(()) => return Ok(()),
            Err(error) if cancel.load(Ordering::Relaxed) || failed.load(Ordering::Relaxed) => {
                return Err(error);
            }
            Err(error) => {
                last_error = Some(error);
                if attempt < RETRIES {
                    let _ = events.send(DownloadEvent::Retrying {
                        path: remote.path.clone(),
                        attempt: attempt + 1,
                        message: last_error.as_ref().unwrap().to_string(),
                    });
                    cancellable_sleep_with_failure(cancel, failed, retry_delay(attempt))?;
                }
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("range download failed")))
}

#[allow(clippy::too_many_arguments)]
fn stream_response(
    response: &mut Response,
    output: &mut File,
    initial: u64,
    remote: &RemoteFile,
    cancel: &AtomicBool,
    events: &Sender<DownloadEvent>,
    multipart: Option<(&AtomicBool, &AtomicU64)>,
) -> Result<()> {
    let mut buffer = vec![0u8; COPY_BUFFER_SIZE];
    let mut downloaded = initial;
    loop {
        if let Some((failed, _)) = multipart {
            check_stopped(cancel, failed)?;
        } else {
            check_cancel(cancel)?;
        }
        let read = response.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        let current = if let Some((_, total)) = multipart {
            total.fetch_add(read as u64, Ordering::Relaxed) + read as u64
        } else {
            downloaded += read as u64;
            downloaded
        };
        let _ = events.send(DownloadEvent::FileProgress {
            path: remote.path.clone(),
            downloaded: current.min(remote.size),
            total: remote.size,
        });
    }
    Ok(())
}

fn file_matches(
    path: &Path,
    remote: &RemoteFile,
    cancel: &AtomicBool,
    events: &Sender<DownloadEvent>,
) -> Result<bool> {
    if fs::metadata(path)?.len() != remote.size {
        return Ok(false);
    }
    if remote.sha256.is_none() {
        return Ok(true);
    }
    verify_file(path, remote, cancel, events)
        .map(|()| true)
        .or(Ok(false))
}

fn verify_file(
    path: &Path,
    remote: &RemoteFile,
    cancel: &AtomicBool,
    events: &Sender<DownloadEvent>,
) -> Result<()> {
    if fs::metadata(path)?.len() != remote.size {
        bail!("size mismatch for {}", remote.path)
    }
    let Some(expected) = remote.sha256.as_deref() else {
        return Ok(());
    };
    let _ = events.send(DownloadEvent::Verifying {
        path: remote.path.clone(),
    });
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; COPY_BUFFER_SIZE];
    loop {
        check_cancel(cancel)?;
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected) {
        bail!(
            "SHA-256 mismatch for {}: expected {expected}, got {actual}",
            remote.path
        )
    }
    Ok(())
}

fn fetch_readme(repo: &str) -> Result<String> {
    let (owner, name) = split_repo(repo)?;
    let mut url = Url::parse(ENDPOINT)?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("invalid Hugging Face endpoint"))?;
        segments.extend([owner, name, "raw", REVISION, "README.md"]);
    }
    let response = api_client()?
        .get(url)
        .send()
        .with_context(|| format!("read model card for {repo}"))?;
    let mut response = checked_response(response, &format!("read model card for {repo}"))?;
    let mut bytes = Vec::new();
    response
        .by_ref()
        .take(256 * 1024)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read model card body for {repo}"))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    Ok(sanitize_model_card(strip_frontmatter(&text)))
}

fn sanitize_model_card(markdown: &str) -> String {
    let mut text = HTML_COMMENT_RE.replace_all(markdown, "").into_owned();
    text = HTML_UNSAFE_RE.replace_all(&text, "").into_owned();
    text = HTML_IMAGE_RE.replace_all(&text, "").into_owned();
    text = MARKDOWN_IMAGE_RE.replace_all(&text, "").into_owned();
    text = HTML_HEADING_OPEN_RE.replace_all(&text, "\n# ").into_owned();
    text = HTML_HEADING_CLOSE_RE.replace_all(&text, "\n").into_owned();
    text = HTML_SUMMARY_RE.replace_all(&text, "\n## ").into_owned();
    text = HTML_BREAK_RE.replace_all(&text, "\n").into_owned();
    text = HTML_LIST_ITEM_RE.replace_all(&text, "\n- ").into_owned();
    text = HTML_CELL_CLOSE_RE.replace_all(&text, " | ").into_owned();
    text = HTML_BLOCK_OPEN_RE.replace_all(&text, "\n").into_owned();
    text = HTML_BLOCK_CLOSE_RE.replace_all(&text, "\n").into_owned();
    text = HTML_STRONG_OPEN_RE.replace_all(&text, "**").into_owned();
    text = HTML_STRONG_CLOSE_RE.replace_all(&text, "**").into_owned();
    text = HTML_CODE_OPEN_RE.replace_all(&text, "`").into_owned();
    text = HTML_CODE_CLOSE_RE.replace_all(&text, "`").into_owned();
    text = HTML_TAG_RE.replace_all(&text, "").into_owned();
    text = decode_html_entities(&text);

    let mut clean = Vec::new();
    let mut blank = false;
    for line in text.lines() {
        let line = line.trim_end().trim_matches('|').trim_end();
        if line.trim().is_empty() {
            if !blank && !clean.is_empty() {
                clean.push(String::new());
            }
            blank = true;
        } else {
            clean.push(line.to_owned());
            blank = false;
        }
    }
    while clean.last().is_some_and(String::is_empty) {
        clean.pop();
    }
    clean.join("\n")
}

fn decode_html_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

fn card_string(card: &Value, key: &str) -> Option<String> {
    match card.get(key)? {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn card_strings(card: &Value, key: &str) -> Vec<String> {
    match card.get(key) {
        Some(Value::String(value)) => vec![value.clone()],
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn strip_frontmatter(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("---\n") else {
        return text;
    };
    rest.find("\n---\n")
        .map(|end| &rest[end + 5..])
        .unwrap_or(text)
}

fn repository_tree(repo: &str) -> Result<Vec<TreeNode>> {
    let mut url = api_url(repo, &["tree", REVISION])?;
    url.query_pairs_mut()
        .append_pair("recursive", "true")
        .append_pair("expand", "false")
        .append_pair("limit", "1000");
    let client = api_client()?;
    let mut nodes = Vec::new();
    let mut next = Some(url);
    while let Some(url) = next.take() {
        let response = client
            .get(url.clone())
            .send()
            .with_context(|| format!("list files in {repo}"))?;
        let response = checked_response(response, &format!("list files in {repo}"))?;
        next = response
            .headers()
            .get("link")
            .and_then(|header| header.to_str().ok())
            .and_then(next_link)
            .and_then(|link| Url::parse(&link).ok());
        nodes.extend(
            response
                .json::<Vec<TreeNode>>()
                .with_context(|| format!("decode file list for {repo}"))?,
        );
    }
    Ok(nodes)
}

fn api_client() -> Result<Client> {
    Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(45))
        .user_agent(format!("llamactl/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .context("build Hugging Face API client")
}

fn download_client() -> Result<Client> {
    Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .user_agent(format!("llamactl/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .context("build Hugging Face download client")
}

fn request(client: &Client, token: Option<&str>, url: Url) -> reqwest::blocking::RequestBuilder {
    let mut request = client.get(url).header(
        USER_AGENT,
        format!("llamactl/{}", env!("CARGO_PKG_VERSION")),
    );
    if let Some(token) = token {
        request = request.header(AUTHORIZATION, format!("Bearer {token}"));
    }
    request
}

fn checked_response(response: Response, action: &str) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    match status.as_u16() {
        401 => bail!("{action}: authentication required"),
        403 => bail!("{action}: access denied or repository terms must be accepted"),
        404 => bail!("{action}: repository not found"),
        429 => bail!("{action}: Hugging Face rate limit exceeded; try again later"),
        _ => bail!("{action}: HTTP {status}"),
    }
}

fn api_url(repo: &str, suffix: &[&str]) -> Result<Url> {
    let (owner, name) = split_repo(repo)?;
    let mut url = Url::parse(ENDPOINT)?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("invalid Hugging Face endpoint"))?;
        segments.extend(["api", "models", owner, name]);
        segments.extend(suffix.iter().copied());
    }
    Ok(url)
}

fn resolve_url(repo: &str, relative: &str) -> Result<Url> {
    let (owner, name) = split_repo(repo)?;
    ensure_safe_relative_path(relative)?;
    let mut url = Url::parse(ENDPOINT)?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("invalid Hugging Face endpoint"))?;
        segments.extend([owner, name, "resolve", REVISION]);
        for component in relative.split('/') {
            segments.push(component);
        }
    }
    url.query_pairs_mut().append_pair("download", "true");
    Ok(url)
}

fn split_repo(repo: &str) -> Result<(&str, &str)> {
    let mut parts = repo.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if owner.is_empty() || name.is_empty() || parts.next().is_some() || owner == "." || name == "."
    {
        bail!("invalid Hugging Face repository '{repo}'")
    }
    if owner == ".." || name == ".." || owner.contains('\\') || name.contains('\\') {
        bail!("unsafe Hugging Face repository '{repo}'")
    }
    Ok((owner, name))
}

fn valid_repo_id(repo: &str) -> bool {
    split_repo(repo).is_ok()
}

fn validate_repo_id(repo: &str) -> Result<()> {
    split_repo(repo).map(|_| ())
}

fn ensure_safe_relative_path(relative: &str) -> Result<()> {
    if relative.is_empty() || relative.contains('\\') || relative.starts_with('/') {
        bail!("refusing unsafe repository path {relative:?}")
    }
    if Path::new(relative)
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("refusing unsafe repository path {relative:?}")
    }
    Ok(())
}

fn is_not_gated(gated: &Value) -> bool {
    matches!(gated, Value::Bool(false))
}

fn next_link(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let (url, relation) = part.trim().split_once(';')?;
        relation.contains("rel=\"next\"").then(|| {
            url.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_owned()
        })
    })
}

fn is_mmproj(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    name.starts_with("mmproj") || name.contains("-mmproj")
}

fn preferred_mmproj(files: &[RemoteFile]) -> Option<&RemoteFile> {
    files.iter().min_by_key(|file| {
        let name = file.path.to_ascii_lowercase();
        if name.contains("f16") && !name.contains("bf16") {
            0
        } else if name.contains("bf16") {
            1
        } else if name.contains("f32") {
            2
        } else {
            3
        }
    })
}

fn shard_group_key(path: &str) -> String {
    let (directory, filename) = path.rsplit_once('/').unwrap_or(("", path));
    let filename = SHARD_RE
        .captures(filename)
        .map(|captures| format!("{}.gguf", &captures[1]))
        .unwrap_or_else(|| filename.to_owned());
    if directory.is_empty() {
        filename
    } else {
        format!("{directory}/{filename}")
    }
}

fn shard_parts(path: &str) -> Option<(usize, usize)> {
    let filename = path.rsplit('/').next().unwrap_or(path);
    let captures = SHARD_RE.captures(filename)?;
    Some((captures[2].parse().ok()?, captures[3].parse().ok()?))
}

fn quantization(path: &str) -> Option<String> {
    QUANT_RE
        .captures(path.rsplit('/').next().unwrap_or(path))
        .map(|captures| captures[1].to_ascii_uppercase())
}

fn quant_quality(quant: &str) -> u8 {
    match quant {
        "IQ1_S" | "IQ1_M" | "Q2_K" | "Q2_K_S" | "Q2_K_L" | "IQ2_S" | "IQ2_M" | "IQ2_XS"
        | "IQ2_XXS" => 1,
        "Q2_K_XL" | "Q3_K_S" | "Q3_K_M" | "Q3_K_L" | "IQ3_S" | "IQ3_XS" | "IQ3_XXS" | "IQ3_M" => 2,
        "Q3_K_XL" | "Q4_0" | "Q4_1" | "Q4_K_S" | "IQ4_NL" | "IQ4_XS" => 3,
        "Q4_K_M" | "Q4_K_XL" | "Q5_0" | "Q5_1" | "Q5_K_S" => 4,
        "Q5_K_M" | "Q5_K_XL" | "Q6_K" | "Q6_K_XL" | "Q8_0" | "Q8_K_XL" | "F16" | "F32" | "BF16" => {
            5
        }
        _ => 3,
    }
}

fn quant_description(quant: &str) -> &'static str {
    match quant {
        "Q2_K" | "Q2_K_S" | "Q2_K_L" => "smallest - significant quality loss",
        "Q3_K_S" | "Q3_K_M" | "Q3_K_L" => "small - noticeable quality loss",
        "Q4_K_M" | "Q4_K_XL" => "recommended balance",
        "Q4_0" | "Q4_1" | "Q4_K_S" | "IQ4_NL" | "IQ4_XS" => "good balance",
        "Q5_K_M" | "Q5_K_XL" | "Q5_K_S" => "excellent quality",
        "Q6_K" | "Q6_K_XL" => "near-lossless",
        "Q8_0" | "Q8_K_XL" => "minimal loss",
        "F16" | "BF16" | "F32" => "full precision",
        _ if quant.starts_with("IQ1") || quant.starts_with("IQ2") => "very compact",
        _ if quant.starts_with("IQ3") => "compact",
        _ => "GGUF model",
    }
}

fn suffixed_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value: OsString = path.as_os_str().to_owned();
    value.push(suffix);
    PathBuf::from(value)
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!("download cancelled")
    }
    Ok(())
}

fn check_stopped(cancel: &AtomicBool, failed: &AtomicBool) -> Result<()> {
    check_cancel(cancel)?;
    if failed.load(Ordering::Relaxed) {
        bail!("another download range failed")
    }
    Ok(())
}

fn retry_delay(attempt: usize) -> Duration {
    let base = 400.0 * 1.6f64.powi(attempt as i32);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    Duration::from_millis((base as u64).min(10_000) + nanos % 121)
}

fn cancellable_sleep(cancel: &AtomicBool, duration: Duration) -> Result<()> {
    let sentinel = AtomicBool::new(false);
    cancellable_sleep_with_failure(cancel, &sentinel, duration)
}

fn cancellable_sleep_with_failure(
    cancel: &AtomicBool,
    failed: &AtomicBool,
    duration: Duration,
) -> Result<()> {
    let mut remaining = duration;
    while remaining > Duration::ZERO {
        check_stopped(cancel, failed)?;
        let step = remaining.min(Duration::from_millis(50));
        thread::sleep(step);
        remaining = remaining.saturating_sub(step);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn remote(path: &str, size: u64) -> RemoteFile {
        RemoteFile {
            path: path.into(),
            size,
            sha256: Some("hash".into()),
        }
    }

    #[test]
    fn rejects_paths_that_escape_the_model_directory() {
        for path in [
            "",
            "../model.gguf",
            "/tmp/model.gguf",
            "a/../../model.gguf",
            "a\\model.gguf",
        ] {
            assert!(
                ensure_safe_relative_path(path).is_err(),
                "accepted {path:?}"
            );
        }
        assert!(ensure_safe_relative_path("quants/model-Q4_K_M.gguf").is_ok());
    }

    #[test]
    fn groups_split_gguf_shards() {
        let first = "BF16/model-BF16-00001-of-00002.gguf";
        let second = "BF16/model-BF16-00002-of-00002.gguf";
        assert_eq!(shard_group_key(first), "BF16/model-BF16.gguf");
        assert_eq!(shard_group_key(first), shard_group_key(second));
        assert_eq!(shard_parts(second), Some((2, 2)));
    }

    #[test]
    fn selects_f16_projector_before_bf16_and_f32() {
        let files = vec![
            remote("mmproj-F32.gguf", 3),
            remote("mmproj-BF16.gguf", 2),
            remote("mmproj-F16.gguf", 1),
        ];
        assert_eq!(preferred_mmproj(&files).unwrap().path, "mmproj-F16.gguf");
    }

    #[test]
    fn parses_dynamic_quantization_without_shorter_false_match() {
        assert_eq!(
            quantization("model-UD-Q4_K_XL.gguf").as_deref(),
            Some("Q4_K_XL")
        );
        assert_eq!(quantization("model-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_quality("Q4_K_M"), 4);
    }

    #[test]
    fn strips_model_card_frontmatter() {
        assert_eq!(
            strip_frontmatter("---\nlicense: apache-2.0\n---\n# Model\nDetails"),
            "# Model\nDetails"
        );
        assert_eq!(strip_frontmatter("# Model"), "# Model");
    }

    #[test]
    fn sanitizes_html_model_cards_for_terminal_rendering() {
        let card = r#"
<div align="center"><img src="logo.png"></div>
<h1>Model &amp; Details</h1>
<p>Use <strong>carefully</strong>.<br>See <a href="https://example.com">docs</a>.</p>
<script>window.alert('no')</script>
<ul><li>First</li><li>Second</li></ul>
"#;
        let clean = sanitize_model_card(card);
        assert!(!clean.contains('<'));
        assert!(!clean.contains("logo.png"));
        assert!(!clean.contains("window.alert"));
        assert!(clean.contains("# Model & Details"));
        assert!(clean.contains("Use **carefully**."));
        assert!(clean.contains("See docs."));
        assert!(clean.contains("- First"));
    }

    #[test]
    fn reads_scalar_and_list_card_metadata() {
        let card = serde_json::json!({
            "license": "apache-2.0",
            "base_model": ["owner/base", "owner/merge"],
        });
        assert_eq!(card_string(&card, "license").as_deref(), Some("apache-2.0"));
        assert_eq!(
            card_strings(&card, "base_model"),
            vec!["owner/base", "owner/merge"]
        );
    }

    #[test]
    fn builds_percent_encoded_resolve_url() {
        let url = resolve_url("owner/repo", "folder/model name-Q4_K_M.gguf").unwrap();
        assert_eq!(
            url.as_str(),
            "https://huggingface.co/owner/repo/resolve/main/folder/model%20name-Q4_K_M.gguf?download=true"
        );
    }

    #[test]
    fn single_transfer_resumes_an_existing_partial() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            let request = String::from_utf8_lossy(&request);
            assert!(request.contains("range: bytes=4-9") || request.contains("Range: bytes=4-9"));
            stream
                .write_all(
                    b"HTTP/1.1 206 Partial Content\r\nContent-Length: 6\r\nContent-Range: bytes 4-9/10\r\nConnection: close\r\n\r\nefghij",
                )
                .unwrap();
        });

        let directory = tempfile::tempdir().unwrap();
        let partial = directory.path().join("model.gguf.part");
        fs::write(&partial, b"abcd").unwrap();
        let remote = RemoteFile {
            path: "model.gguf".into(),
            size: 10,
            sha256: None,
        };
        let (events, _rx) = mpsc::channel();
        download_single(
            &download_client().unwrap(),
            None,
            &Url::parse(&format!("http://{address}/model.gguf")).unwrap(),
            &remote,
            &partial,
            &AtomicBool::new(false),
            &events,
        )
        .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(partial).unwrap(), b"abcdefghij");
    }
}
