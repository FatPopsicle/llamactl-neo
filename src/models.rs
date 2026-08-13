use crate::config::{Config, Paths};
use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    hash::{Hash, Hasher},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex, Once},
    time::UNIX_EPOCH,
};
use walkdir::WalkDir;

pub const MIN_MODEL_BYTES: u64 = 50_000_000;

static SHARD_SUFFIX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-\d{5}-of-\d{5}$").unwrap());
static ID_SANITIZE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^A-Za-z0-9._-]").unwrap());
static SHARD_PART_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-(\d{5})-of-(\d{5})\.gguf$").unwrap());
static SHARD_LEAD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-(\d{5})-of-\d{5}\.gguf$").unwrap());
static LAYER_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^blk\.(\d+)\.").unwrap());

#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    pub relative: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub vision: bool,
    pub kind: ModelKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Main,
    Draft,
}

pub fn canonical_stem(path: &Path) -> String {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    SHARD_SUFFIX_RE.replace(&stem, "").to_string()
}
pub fn sanitize_id(s: &str) -> String {
    ID_SANITIZE_RE.replace_all(s, "-").to_lowercase()
}
fn parts(path: &Path) -> Vec<PathBuf> {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let Some(cap) = SHARD_PART_RE.captures(&name) else {
        return vec![path.to_owned()];
    };
    let prefix = &name[..cap.get(0).unwrap().start()];
    let total = cap[2].parse::<usize>().unwrap_or(1);
    (1..=total)
        .map(|i| path.with_file_name(format!("{prefix}-{i:05}-of-{total:05}.gguf")))
        .filter(|p| p.is_file())
        .collect()
}
pub fn model_bytes(path: &Path) -> u64 {
    parts(path)
        .iter()
        .filter_map(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .sum()
}

pub fn models_fingerprint(cfg: &Config) -> u64 {
    // Order-independent aggregation avoids allocating and sorting one formatted
    // string per model on every UI refresh.
    let mut fingerprint = 0u64;
    let mut count = 0u64;
    for root in &cfg.models_dirs {
        if !root.is_dir() {
            continue;
        }
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if !entry.file_type().is_file()
                || !path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
                || path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| starts_with_ignore_ascii_case(name, "mmproj"))
            {
                continue;
            }
            if let Ok(metadata) = entry.metadata() {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                path.hash(&mut hasher);
                metadata.len().hash(&mut hasher);
                metadata.modified().ok().hash(&mut hasher);
                fingerprint ^= hasher.finish().rotate_left((count % 64) as u32);
                count += 1;
            }
        }
    }
    fingerprint ^ count.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

pub fn scan(cfg: &Config) -> Vec<Model> {
    let mut result = BTreeMap::<String, Model>::new();
    for root in &cfg.models_dirs {
        if !root.is_dir() {
            continue;
        }
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if !entry.file_type().is_file()
                || path
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| !x.eq_ignore_ascii_case("gguf"))
                    .unwrap_or(true)
            {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if starts_with_ignore_ascii_case(&name, "mmproj") {
                continue;
            }
            if SHARD_LEAD_RE
                .captures(&name)
                .is_some_and(|c| &c[1] != "00001")
            {
                continue;
            }
            let bytes = model_bytes(path);
            if bytes < MIN_MODEL_BYTES {
                continue;
            }
            let relative = path
                .strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned();
            let base = sanitize_id(&canonical_stem(path));
            let mut id = base.clone();
            if result.contains_key(&id) {
                let publisher = relative.split('/').next().unwrap_or("");
                id = format!("{}--{base}", sanitize_id(publisher));
            }
            let vision = path
                .parent()
                .and_then(|p| fs::read_dir(p).ok())
                .is_some_and(|it| {
                    it.filter_map(Result::ok).any(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .to_lowercase()
                            .starts_with("mmproj")
                            && e.path().extension().is_some_and(|x| x == "gguf")
                    })
                });
            let kind = if is_draft(path) {
                ModelKind::Draft
            } else {
                ModelKind::Main
            };
            result.insert(
                id.clone(),
                Model {
                    id,
                    relative,
                    path: path.to_owned(),
                    bytes,
                    vision,
                    kind,
                },
            );
        }
    }
    let mut models = result.into_values().collect::<Vec<_>>();
    models.sort_by(|left, right| {
        (left.kind == ModelKind::Draft)
            .cmp(&(right.kind == ModelKind::Draft))
            .then_with(|| left.id.cmp(&right.id))
    });
    models
}

pub fn resolve(
    cfg: &Config,
    query: Option<&str>,
) -> Result<(Vec<String>, Option<PathBuf>, String)> {
    let query = query
        .filter(|s| !s.is_empty())
        .unwrap_or(&cfg.default_model);
    if query.is_empty() {
        bail!("no model given and no default_model in config")
    }
    let path = PathBuf::from(query);
    if path.is_file() {
        let id = sanitize_id(&canonical_stem(&path));
        return Ok((model_args(&path), Some(path), id));
    }
    let models = scan(cfg);
    if let Some(m) = models.iter().find(|m| {
        query == m.id || query == m.relative || m.path.file_name().is_some_and(|n| n == query)
    }) {
        return Ok((model_args(&m.path), Some(m.path.clone()), m.id.clone()));
    }
    let hits = models
        .iter()
        .filter(|m| m.relative.to_lowercase().contains(&query.to_lowercase()))
        .collect::<Vec<_>>();
    if hits.len() == 1 {
        let m = hits[0];
        return Ok((model_args(&m.path), Some(m.path.clone()), m.id.clone()));
    }
    if hits.len() > 1 {
        bail!(
            "'{query}' is ambiguous, matches:\n{}",
            hits.iter()
                .map(|m| format!("  {}", m.relative))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    if query.contains('/') {
        return Ok((vec!["-hf".into(), query.into()], None, sanitize_id(query)));
    }
    bail!("no model matching '{query}' — see 'llamactl models'")
}

fn starts_with_ignore_ascii_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|start| start.eq_ignore_ascii_case(prefix))
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GgufMetadata {
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub attention_heads: Option<u64>,
    pub kv_heads: Vec<u64>,
    pub key_length: Vec<u64>,
    pub value_length: Vec<u64>,
    pub sliding_window: Option<u64>,
    pub sliding_pattern: Vec<u64>,
    pub pooling_type: Option<u64>,
    pub nextn_layers: Option<u64>,
    pub mtp_markers: Vec<String>,
    pub tokenizer: Option<TokenizerMetadata>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenizerMetadata {
    pub token_count: u64,
    pub prefix_hashes: BTreeMap<u64, String>,
    pub bos_token_id: Option<u64>,
    pub eos_token_id: Option<u64>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FileStamp {
    len: u64,
    modified: Option<u64>,
}

#[derive(Clone, Serialize, Deserialize)]
struct CachedMetadata {
    stamp: FileStamp,
    basic: GgufMetadata,
    full: Option<GgufMetadata>,
}

static METADATA_CACHE: LazyLock<Mutex<HashMap<PathBuf, CachedMetadata>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static METADATA_CACHE_INIT: Once = Once::new();
static METADATA_CACHE_PATH: LazyLock<Mutex<Option<PathBuf>>> = LazyLock::new(|| Mutex::new(None));
type TensorInfo = (String, Option<usize>, u64);
type TensorCache = HashMap<PathBuf, (Vec<FileStamp>, Vec<TensorInfo>)>;
static TENSOR_CACHE: LazyLock<Mutex<TensorCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn file_stamp(path: &Path) -> Option<FileStamp> {
    let metadata = fs::metadata(path).ok()?;
    Some(FileStamp {
        len: metadata.len(),
        modified: metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos().min(u64::MAX as u128) as u64),
    })
}

pub fn init_metadata_cache(paths: &Paths) {
    if let Ok(mut path) = METADATA_CACHE_PATH.lock() {
        *path = Some(paths.metadata_cache.clone());
    }
    METADATA_CACHE_INIT.call_once(|| {
        let Ok(text) = fs::read_to_string(&paths.metadata_cache) else {
            return;
        };
        let Ok(cache) = serde_json::from_str::<HashMap<PathBuf, CachedMetadata>>(&text) else {
            return;
        };
        if let Ok(mut memory) = METADATA_CACHE.lock() {
            *memory = cache;
        }
    });
}

fn persist_metadata_cache() {
    let path = METADATA_CACHE_PATH
        .lock()
        .ok()
        .and_then(|path| path.clone());
    let Some(path) = path else { return };
    let Some(parent) = path.parent() else { return };
    let Ok(cache) = METADATA_CACHE.lock() else {
        return;
    };
    if fs::create_dir_all(parent).is_err() {
        return;
    }
    let Ok(mut temporary) = tempfile::NamedTempFile::new_in(parent) else {
        return;
    };
    if serde_json::to_writer(&mut temporary, &*cache).is_ok() {
        let _ = temporary.persist(path);
    }
}

fn cached_gguf_metadata(path: &Path, markers: bool) -> Option<GgufMetadata> {
    let stamp = file_stamp(path)?;
    if let Ok(cache) = METADATA_CACHE.lock()
        && let Some(cached) = cache.get(path)
        && cached.stamp == stamp
    {
        if !markers {
            return Some(cached.basic.clone());
        }
        if let Some(full) = &cached.full {
            return Some(full.clone());
        }
    }

    let before = stamp.clone();
    let parsed = read_gguf_metadata(path, markers)?;
    if file_stamp(path).as_ref() != Some(&before) {
        return None;
    }
    let mut changed = false;
    if let Ok(mut cache) = METADATA_CACHE.lock() {
        let entry = cache
            .entry(path.to_owned())
            .or_insert_with(|| CachedMetadata {
                stamp: stamp.clone(),
                basic: parsed.clone(),
                full: None,
            });
        if entry.stamp != stamp {
            *entry = CachedMetadata {
                stamp,
                basic: parsed.clone(),
                full: None,
            };
        }
        if markers {
            entry.basic = parsed.clone();
            entry.basic.mtp_markers.clear();
            entry.full = Some(parsed.clone());
        } else {
            entry.basic = parsed.clone();
        }
        changed = true;
    }
    if changed {
        persist_metadata_cache();
    }
    Some(parsed)
}

pub fn gguf_metadata(path: &Path) -> Option<GgufMetadata> {
    cached_gguf_metadata(path, true)
}

fn basic_gguf_metadata(path: &Path) -> Option<GgufMetadata> {
    cached_gguf_metadata(path, false)
}

fn read_gguf_metadata(path: &Path, scan_markers: bool) -> Option<GgufMetadata> {
    let mut file = BufReader::with_capacity(64 * 1024, fs::File::open(path).ok()?);
    let mut magic = [0; 4];
    file.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let _version = read_u32(&mut file)?;
    let tensors = read_u64(&mut file)?;
    let kv_count = read_u64(&mut file)?;
    let mut values = BTreeMap::<String, MetaValue>::new();
    let mut tokenizer = None;
    for _ in 0..kv_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        if key == "tokenizer.ggml.tokens" && value_type == 9 {
            tokenizer = Some(read_tokenizer_metadata(&mut file)?);
        } else {
            values.insert(key, read_meta_value(&mut file, value_type)?);
        }
    }
    let mut markers = vec![];
    if scan_markers {
        for _ in 0..tensors {
            let name = read_string(&mut file)?;
            let dimensions = read_u32(&mut file)?;
            skip_bytes(&mut file, u64::from(dimensions) * 8 + 12).ok()?;
            for (suffix, marker) in [
                (".nextn.eh_proj.weight", "eh_proj"),
                (".nextn.enorm.weight", "enorm"),
                (".nextn.hnorm.weight", "hnorm"),
            ] {
                if name.ends_with(suffix) && !markers.iter().any(|item| item == marker) {
                    markers.push(marker.to_owned());
                }
            }
        }
    }
    let architecture = values.get("general.architecture")?.string()?.to_owned();
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    Some(GgufMetadata {
        context_length: values
            .get(&key("context_length"))
            .and_then(MetaValue::number),
        block_count: values.get(&key("block_count")).and_then(MetaValue::number),
        embedding_length: values
            .get(&key("embedding_length"))
            .and_then(MetaValue::number),
        attention_heads: values
            .get(&key("attention.head_count"))
            .and_then(MetaValue::number),
        kv_heads: values
            .get(&key("attention.head_count_kv"))
            .map(MetaValue::numbers)
            .unwrap_or_default(),
        key_length: values
            .get(&key("attention.key_length"))
            .map(MetaValue::numbers)
            .unwrap_or_default(),
        value_length: values
            .get(&key("attention.value_length"))
            .map(MetaValue::numbers)
            .unwrap_or_default(),
        sliding_window: values
            .get(&key("attention.sliding_window"))
            .and_then(MetaValue::number),
        sliding_pattern: values
            .get(&key("attention.sliding_window_pattern"))
            .map(MetaValue::numbers)
            .unwrap_or_default(),
        pooling_type: values.get(&key("pooling_type")).and_then(MetaValue::number),
        nextn_layers: values
            .get(&key("nextn_predict_layers"))
            .and_then(MetaValue::number),
        mtp_markers: markers,
        tokenizer: tokenizer.map(|mut tokenizer| {
            tokenizer.bos_token_id = values
                .get("tokenizer.ggml.bos_token_id")
                .and_then(MetaValue::number);
            tokenizer.eos_token_id = values
                .get("tokenizer.ggml.eos_token_id")
                .and_then(MetaValue::number);
            tokenizer.add_bos_token = values
                .get("tokenizer.ggml.add_bos_token")
                .and_then(MetaValue::boolean);
            tokenizer.add_eos_token = values
                .get("tokenizer.ggml.add_eos_token")
                .and_then(MetaValue::boolean);
            tokenizer
        }),
    })
}

#[derive(Debug, Clone)]
enum MetaValue {
    Number(u64),
    Numbers(Vec<u64>),
    String(String),
}
impl MetaValue {
    fn number(&self) -> Option<u64> {
        match self {
            Self::Number(value) => Some(*value),
            Self::Numbers(_) | Self::String(_) => None,
        }
    }
    fn string(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            Self::Number(_) | Self::Numbers(_) => None,
        }
    }
    fn boolean(&self) -> Option<bool> {
        self.number().map(|value| value != 0)
    }
    fn numbers(&self) -> Vec<u64> {
        match self {
            Self::Number(value) => vec![*value],
            Self::Numbers(values) => values.clone(),
            Self::String(_) => vec![],
        }
    }
}
fn read_u32(reader: &mut impl Read) -> Option<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes).ok()?;
    Some(u32::from_le_bytes(bytes))
}
fn read_u64(reader: &mut impl Read) -> Option<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes).ok()?;
    Some(u64::from_le_bytes(bytes))
}
fn read_string(reader: &mut impl Read) -> Option<String> {
    let length = read_u64(reader)? as usize;
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes).ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}
fn skip_bytes(reader: &mut impl Read, mut amount: u64) -> std::io::Result<()> {
    let mut scratch = [0u8; 4096];
    while amount > 0 {
        let chunk = amount.min(scratch.len() as u64) as usize;
        reader.read_exact(&mut scratch[..chunk])?;
        amount -= chunk as u64;
    }
    Ok(())
}
fn scalar_size(value_type: u32) -> Option<u64> {
    Some(match value_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}
fn read_tokenizer_metadata(file: &mut impl Read) -> Option<TokenizerMetadata> {
    if read_u32(file)? != 8 {
        return None;
    }
    let token_count = read_u64(file)?;
    let first = 5u64;
    let hash_from = first.max(token_count.saturating_sub(128));
    let mut hasher = Sha256::new();
    let mut prefix_hashes = BTreeMap::new();
    if first >= hash_from {
        prefix_hashes.insert(first, format!("{:x}", hasher.clone().finalize()));
    }
    for token_id in 0..token_count {
        let token = read_string(file)?;
        if token_id < first {
            continue;
        }
        let token = if token.is_empty() {
            format!("[EMPTY_{token_id}]")
        } else {
            token
        };
        let bytes = token.as_bytes();
        hasher.update((token_id as u32).to_le_bytes());
        hasher.update((bytes.len() as u32).to_le_bytes());
        hasher.update(bytes);
        let end = token_id + 1;
        if end >= hash_from {
            prefix_hashes.insert(end, format!("{:x}", hasher.clone().finalize()));
        }
    }
    Some(TokenizerMetadata {
        token_count,
        prefix_hashes,
        ..TokenizerMetadata::default()
    })
}

fn read_meta_value(file: &mut impl Read, value_type: u32) -> Option<MetaValue> {
    if value_type == 8 {
        return Some(MetaValue::String(read_string(file)?));
    }
    if value_type == 9 {
        let element = read_u32(file)?;
        let count = read_u64(file)?;
        if element == 8 {
            for _ in 0..count {
                let length = read_u64(file)?;
                skip_bytes(file, length).ok()?;
            }
            return Some(MetaValue::Number(count));
        }
        let size = scalar_size(element)?;
        if count <= 4096 {
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let mut bytes = [0u8; 8];
                file.read_exact(&mut bytes[..size as usize]).ok()?;
                values.push(match size {
                    1 => bytes[0] as u64,
                    2 => u16::from_le_bytes(bytes[..2].try_into().ok()?) as u64,
                    4 => u32::from_le_bytes(bytes[..4].try_into().ok()?) as u64,
                    8 => u64::from_le_bytes(bytes),
                    _ => return None,
                });
            }
            return Some(MetaValue::Numbers(values));
        }
        skip_bytes(file, size * count).ok()?;
        return Some(MetaValue::Number(count));
    }
    let size = scalar_size(value_type)?;
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes[..size as usize]).ok()?;
    let value = match size {
        1 => bytes[0] as u64,
        2 => u16::from_le_bytes(bytes[..2].try_into().ok()?) as u64,
        4 => u32::from_le_bytes(bytes[..4].try_into().ok()?) as u64,
        8 => u64::from_le_bytes(bytes),
        _ => return None,
    };
    Some(MetaValue::Number(value))
}

pub struct Estimate {
    pub vram: u64,
    pub ram: u64,
}

fn ggml_type_size(ttype: u32) -> Option<(u64, u64)> {
    Some(match ttype {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 40),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        29 => (256, 56),
        30 => (1, 2),
        31..=33 => (32, 18),
        34 => (256, 54),
        35 => (256, 66),
        39 => (32, 17),
        40 => (16, 9),
        _ => return None,
    })
}

fn tensor_sizes(path: &Path) -> Option<Vec<TensorInfo>> {
    let parts = parts(path);
    let stamps = parts
        .iter()
        .map(|part| file_stamp(part))
        .collect::<Option<Vec<_>>>()?;
    if let Ok(cache) = TENSOR_CACHE.lock()
        && let Some((cached_stamps, tensors)) = cache.get(path)
        && cached_stamps == &stamps
    {
        return Some(tensors.clone());
    }

    let mut out = Vec::new();
    for part in &parts {
        let mut file = BufReader::with_capacity(1 << 20, fs::File::open(part).ok()?);
        let mut magic = [0; 4];
        file.read_exact(&mut magic).ok()?;
        if &magic != b"GGUF" {
            return None;
        }
        let _version = read_u32(&mut file)?;
        let tensors = read_u64(&mut file)?;
        let kv_count = read_u64(&mut file)?;
        for _ in 0..kv_count {
            let _key = read_string(&mut file)?;
            let value_type = read_u32(&mut file)?;
            let _ = read_meta_value(&mut file, value_type)?;
        }
        for _ in 0..tensors {
            let name = read_string(&mut file)?;
            let dims = read_u32(&mut file)?;
            let mut elements = 1u64;
            for _ in 0..dims {
                elements = elements.saturating_mul(read_u64(&mut file)?);
            }
            let ttype = read_u32(&mut file)?;
            let _offset = read_u64(&mut file)?;
            let size = ggml_type_size(ttype)
                .map(|(block, bytes)| {
                    elements
                        .checked_mul(bytes)
                        .and_then(|product| product.checked_div(block))
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            let layer = LAYER_RE
                .captures(&name)
                .and_then(|c| c[1].parse::<usize>().ok());
            out.push((name, layer, size));
        }
    }
    if let Ok(mut cache) = TENSOR_CACHE.lock() {
        cache.insert(path.to_owned(), (stamps, out.clone()));
    }
    Some(out)
}

struct Placement {
    gpu: u64,
    total: u64,
    layer_gpu: Vec<bool>,
}

fn tensor_placement(
    tensors: &[TensorInfo],
    layers: u64,
    ngl: u64,
    ncmoe: u64,
    overrides: &[(Regex, bool)],
) -> Option<Placement> {
    let total: u64 = tensors.iter().map(|(_, _, size)| size).sum();
    if total == 0 || layers == 0 {
        return None;
    }
    let boundary = ngl.min(layers + 1);
    let moe_layers = ncmoe.min(layers);
    let mut gpu = 0u64;
    let mut layer_votes = vec![(0u64, 0u64); layers as usize];
    let mut attn_votes = vec![(0u64, 0u64); layers as usize];
    for (name, layer, size) in tensors {
        let mut on_gpu = match layer {
            Some(l) => (*l as u64) < boundary,
            None => boundary > layers,
        };
        let is_expert = name.contains("_exps");
        if moe_layers > 0 && is_expert && layer.is_some_and(|l| (l as u64) < moe_layers) {
            on_gpu = false;
        }
        for (rx, to_cpu) in overrides {
            if rx.is_match(name) {
                on_gpu = !to_cpu;
            }
        }
        if on_gpu {
            gpu += size;
        }
        let Some(l) = layer else { continue };
        let li = *l;
        if li >= layers as usize || is_expert {
            continue;
        }
        let (g, t) = &mut layer_votes[li];
        if on_gpu {
            *g += size;
        }
        *t += size;
        if name.contains("attn_") || name.contains(".attention") || name.contains("_attention") {
            let (g, t) = &mut attn_votes[li];
            if on_gpu {
                *g += size;
            }
            *t += size;
        }
    }
    let layer_gpu = (0..layers as usize)
        .map(|i| {
            let (g, t) = attn_votes[i];
            if t > 0 {
                g * 2 >= t
            } else {
                let (g, t) = layer_votes[i];
                if t > 0 {
                    g * 2 >= t
                } else {
                    (i as u64) < boundary.min(layers)
                }
            }
        })
        .collect();
    Some(Placement {
        gpu,
        total,
        layer_gpu,
    })
}

pub fn estimate(path: &Path, args: &[String]) -> Estimate {
    let weights = model_bytes(path);
    let metadata = basic_gguf_metadata(path);
    let value = |flags: &[&str]| {
        args.windows(2)
            .rev()
            .find(|pair| flags.iter().any(|flag| pair[0] == *flag))
            .map(|pair| pair[1].as_str())
    };
    let parse = |flags: &[&str], default: u64| {
        value(flags)
            .and_then(|item| item.parse().ok())
            .unwrap_or(default)
    };
    let ctx = parse(
        &["-c", "--ctx-size"],
        metadata
            .as_ref()
            .and_then(|item| item.context_length)
            .unwrap_or(4096),
    );
    let parallel = parse(&["-np", "--parallel"], 1);
    let layers = metadata
        .as_ref()
        .and_then(|item| item.block_count)
        .unwrap_or(1);
    let ngl = value(&["-ngl", "--gpu-layers", "--n-gpu-layers"])
        .map(|item| {
            if matches!(item, "all" | "auto") {
                layers + 1
            } else {
                item.parse().unwrap_or(layers + 1)
            }
        })
        .unwrap_or(layers + 1);
    let fraction = ngl.min(layers + 1) as f64 / (layers + 1) as f64;

    let ncmoe = value(&["-ncmoe", "--n-cpu-moe"])
        .map(|v| {
            if v.eq_ignore_ascii_case("all") || v.eq_ignore_ascii_case("max") {
                layers
            } else {
                v.parse::<u64>().unwrap_or(0).min(layers)
            }
        })
        .unwrap_or(0);
    let overrides = value(&["-ot", "--override-tensor"])
        .into_iter()
        .flat_map(|spec| spec.split(','))
        .filter_map(|item| {
            let mut split = item.splitn(2, '=');
            let pattern = split.next()?.trim();
            let buffer = split.next()?.trim();
            Some((
                Regex::new(pattern).ok()?,
                buffer.to_ascii_uppercase().starts_with("CPU"),
            ))
        })
        .collect::<Vec<_>>();
    let tensors = (!overrides.is_empty() || ncmoe > 0)
        .then(|| tensor_sizes(path))
        .flatten();
    let placement = tensors
        .as_deref()
        .and_then(|tensors| tensor_placement(tensors, layers, ngl, ncmoe, &overrides));
    let mut gpu_weights = weights as f64 * fraction;
    if let Some(final_placement) = &placement
        && let Some(base) = tensors
            .as_deref()
            .and_then(|tensors| tensor_placement(tensors, layers, ngl, 0, &[]))
    {
        let shifted = gpu_weights
            + weights as f64 * (final_placement.gpu as f64 - base.gpu as f64)
                / final_placement.total as f64;
        gpu_weights = shifted.clamp(0.0, weights as f64);
    }
    let embedding = metadata
        .as_ref()
        .and_then(|item| item.embedding_length)
        .unwrap_or(4096);
    let heads = metadata
        .as_ref()
        .and_then(|item| item.attention_heads)
        .unwrap_or(32)
        .max(1);
    let per_layer = |values: &[u64], index: usize, default: u64| {
        values
            .get(index)
            .copied()
            .or_else(|| values.first().copied())
            .unwrap_or(default)
    };
    let bpw = |flags: &[&str]| match value(flags).unwrap_or("f16") {
        "f32" => 4.0,
        "q8_0" => 1.0625,
        "q5_1" => 0.75,
        "q5_0" => 0.6875,
        "q4_1" => 0.625,
        "q4_0" | "iq4_nl" => 0.5625,
        _ => 2.0,
    };
    let unified = args.iter().any(|arg| arg == "--kv-unified")
        || !args.iter().any(|arg| arg == "--no-kv-unified");
    let multiplier = if unified { 1 } else { parallel };
    let kv_layers = (0..layers as usize)
        .map(|index| {
            let item = metadata.as_ref();
            let kv_heads = item
                .map(|meta| per_layer(&meta.kv_heads, index, heads))
                .unwrap_or(heads);
            let key = item
                .map(|meta| per_layer(&meta.key_length, index, embedding / heads))
                .unwrap_or(embedding / heads);
            let val = item
                .map(|meta| per_layer(&meta.value_length, index, embedding / heads))
                .unwrap_or(embedding / heads);
            let layer_ctx = item
                .and_then(|meta| meta.sliding_window.map(|window| (meta, window)))
                .map(|(meta, window)| {
                    let is_sliding = meta
                        .sliding_pattern
                        .get(index % meta.sliding_pattern.len().max(1))
                        .copied()
                        .unwrap_or(if (index + 1) % 6 == 0 { 0 } else { 1 })
                        != 0;
                    if is_sliding { ctx.min(window) } else { ctx }
                })
                .unwrap_or(ctx);
            kv_heads as f64
                * (key as f64 * bpw(&["-ctk", "--cache-type-k"])
                    + val as f64 * bpw(&["-ctv", "--cache-type-v"]))
                * layer_ctx as f64
                * multiplier as f64
        })
        .collect::<Vec<_>>();
    let kv_full: f64 = kv_layers.iter().sum();
    let kv = if let Some(placement) = &placement {
        kv_layers
            .iter()
            .zip(placement.layer_gpu.iter())
            .filter(|(_, gpu)| **gpu)
            .map(|(bytes, _)| bytes)
            .sum::<f64>()
    } else {
        kv_full * ngl.min(layers) as f64 / layers as f64
    };

    let eval_batch = parse(&["-b", "--batch-size"], 512);
    let phys_batch = parse(&["-ub", "--ubatch-size"], 512);
    let batch = ctx.min(eval_batch).min(phys_batch).max(1);
    let batch_factor = batch as f64 / 512.0;
    let flash = value(&["--flash-attn"]).is_some_and(|v| v != "off")
        || !args.iter().any(|a| a == "--no-flash-attn");
    let known_arch = heads > 0;
    let input_buf = if known_arch {
        batch as f64
            + embedding as f64 * batch as f64
            + batch as f64
            + ctx as f64 * batch as f64
            + ctx as f64
            + batch as f64
    } else {
        0.0
    };
    let compute_buf = if known_arch {
        if flash {
            (gpu_weights + kv) * 0.05 * batch_factor
        } else {
            (ctx as f64 / 1024.0 * 2.0 + 0.75) * heads as f64 * 1024.0 * 1024.0 * batch_factor
        }
    } else {
        0.0
    };
    let buffers = input_buf + compute_buf;

    let context_cal = if known_arch { 1.0 / 2.2 } else { 1.0 };
    let projector = model_args(path)
        .windows(2)
        .find(|items| items[0] == "--mmproj")
        .and_then(|items| fs::metadata(&items[1]).ok())
        .map(|metadata| metadata.len() as f64 + 300e6)
        .unwrap_or(0.0);
    let draft = value(&["-md", "--model-draft", "--spec-draft-model"])
        .map(PathBuf::from)
        .filter(|draft| draft.is_file())
        .map(|draft| model_bytes(&draft) as f64 + 300e6)
        .unwrap_or(0.0);
    let vram =
        ((gpu_weights + projector + draft) * 1.03 + (kv + buffers) * 1.03 * context_cal) as u64;

    let ram_weights = (weights as f64 - gpu_weights).max(0.0) + draft * 0.01;
    let ram_kv = (kv_full - kv).max(0.0);
    let cpu_layer_frac = if layers > 0 {
        (layers - ngl.min(layers)) as f64 / layers as f64
    } else {
        0.0
    };
    let ram_overhead = buffers * cpu_layer_frac;
    let ram = ((ram_weights + ram_kv + ram_overhead + 1e9) * 1.05) as u64;
    Estimate { vram, ram }
}

pub fn estimate_vram(path: &Path, args: &[String]) -> u64 {
    estimate(path, args).vram
}

const DRAFT_ARCHITECTURES: [&str; 3] = ["dflash", "dspark", "eagle3"];

fn architecture_and_block_count(path: &Path) -> Option<(String, Option<u64>)> {
    // These keys are near the front of normal GGUF files. Stop as soon as both
    // are known so library discovery does not parse enormous tokenizer arrays.
    let mut file = BufReader::with_capacity(64 * 1024, fs::File::open(path).ok()?);
    let mut magic = [0; 4];
    file.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let _version = read_u32(&mut file)?;
    let _tensors = read_u64(&mut file)?;
    let kv_count = read_u64(&mut file)?;
    let mut architecture = None::<String>;
    let mut block_count = None;
    for _ in 0..kv_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        let value = read_meta_value(&mut file, value_type)?;
        if key == "general.architecture" {
            architecture = value.string().map(str::to_owned);
        } else if architecture
            .as_ref()
            .is_some_and(|architecture| key == format!("{architecture}.block_count"))
        {
            block_count = value.number();
        }
        if architecture.is_some() && block_count.is_some() {
            break;
        }
    }
    Some((architecture?, block_count))
}

pub fn is_draft(path: &Path) -> bool {
    let Some((architecture, block_count)) = architecture_and_block_count(path) else {
        return false;
    };
    if architecture.ends_with("-assistant") || DRAFT_ARCHITECTURES.contains(&architecture.as_str())
    {
        return true;
    }
    block_count == Some(1) && gguf_metadata(path).is_some_and(|full| !full.mtp_markers.is_empty())
}
pub fn context_limit(path: &Path) -> Option<u64> {
    basic_gguf_metadata(path)?.context_length
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DraftCompatibility {
    Compatible,
    Incompatible(String),
    Unknown(String),
}

fn draft_tokenizer_compatibility(
    main_vocab: &TokenizerMetadata,
    draft_vocab: &TokenizerMetadata,
) -> DraftCompatibility {
    // llama.cpp only compares a special token ID when that token is configured
    // to be added. GGUFs can legitimately use different EOS IDs while both
    // have add_eos_token=false.
    for (name, main_add, draft_add, main_id, draft_id) in [
        (
            "BOS",
            main_vocab.add_bos_token,
            draft_vocab.add_bos_token,
            main_vocab.bos_token_id,
            draft_vocab.bos_token_id,
        ),
        (
            "EOS",
            main_vocab.add_eos_token,
            draft_vocab.add_eos_token,
            main_vocab.eos_token_id,
            draft_vocab.eos_token_id,
        ),
    ] {
        if matches!((main_add, draft_add), (Some(main), Some(draft)) if main != draft) {
            return DraftCompatibility::Incompatible(format!(
                "{name} tokenizer add settings differ"
            ));
        }
        if main_add == Some(true)
            && draft_add == Some(true)
            && matches!((main_id, draft_id), (Some(main), Some(draft)) if main != draft)
        {
            return DraftCompatibility::Incompatible(format!("{name} tokenizer token IDs differ"));
        }
    }
    if main_vocab.token_count.abs_diff(draft_vocab.token_count) > 128 {
        return DraftCompatibility::Incompatible(format!(
            "vocabulary sizes differ by more than 128 ({} vs {})",
            main_vocab.token_count, draft_vocab.token_count
        ));
    }
    let prefix = main_vocab.token_count.min(draft_vocab.token_count);
    match (
        main_vocab.prefix_hashes.get(&prefix),
        draft_vocab.prefix_hashes.get(&prefix),
    ) {
        (Some(main_hash), Some(draft_hash)) if main_hash == draft_hash => {
            DraftCompatibility::Compatible
        }
        (Some(_), Some(_)) => {
            DraftCompatibility::Incompatible("tokenizer vocabulary prefixes differ".into())
        }
        _ => DraftCompatibility::Unknown("tokenizer prefix hash unavailable".into()),
    }
}

pub fn draft_compatibility(main: &Path, draft: &Path) -> DraftCompatibility {
    if main == draft {
        return DraftCompatibility::Incompatible("draft is the main model".into());
    }
    let Some(main_meta) = basic_gguf_metadata(main) else {
        return DraftCompatibility::Unknown("main GGUF metadata unavailable".into());
    };
    let Some(draft_meta) = basic_gguf_metadata(draft) else {
        return DraftCompatibility::Unknown("draft GGUF metadata unavailable".into());
    };
    let (Some(main_vocab), Some(draft_vocab)) = (main_meta.tokenizer, draft_meta.tokenizer) else {
        return DraftCompatibility::Unknown("tokenizer vocabulary metadata unavailable".into());
    };
    draft_tokenizer_compatibility(&main_vocab, &draft_vocab)
}
pub fn has_mtp(path: &Path) -> bool {
    gguf_metadata(path).is_some_and(|metadata| {
        metadata.nextn_layers.unwrap_or(0) > 0
            && ["eh_proj", "enorm", "hnorm"]
                .iter()
                .all(|marker| metadata.mtp_markers.iter().any(|item| item == marker))
    })
}
pub fn serving_mode(path: &Path) -> Option<&'static str> {
    match basic_gguf_metadata(path)?.pooling_type.unwrap_or(0) {
        4 => Some("rerank"),
        0 => None,
        _ => Some("embed"),
    }
}
pub fn model_args(path: &Path) -> Vec<String> {
    let mut args = vec!["-m".into(), path.display().to_string()];
    if let Some(parent) = path.parent()
        && let Ok(entries) = fs::read_dir(parent)
        && let Some(mm) = entries.filter_map(Result::ok).map(|e| e.path()).find(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().to_lowercase().starts_with("mmproj"))
                && p.extension().is_some_and(|x| x == "gguf")
        })
    {
        args.extend(["--mmproj".into(), mm.display().to_string()]);
    }
    args
}

pub fn delete(cfg: &Config, id: &str) -> Result<Vec<PathBuf>> {
    let model = scan(cfg)
        .into_iter()
        .find(|m| m.id == id)
        .with_context(|| format!("unknown model '{id}'"))?;
    if cfg.scheduler_pinned_models.iter().any(|p| p == id) {
        bail!("model '{id}' is pinned; unpin it before deletion")
    }
    let mut files = parts(&model.path);
    if let Some(parent) = model.path.parent() {
        files.extend(
            fs::read_dir(parent)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .is_some_and(|n| n.to_string_lossy().to_lowercase().starts_with("mmproj"))
                }),
        );
    }
    for file in &files {
        fs::remove_file(file).with_context(|| format!("remove {}", file.display()))?;
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_shard_suffix_is_removed_from_id() {
        let path = Path::new("deepseek-v4-IQ2-00001-of-00003.gguf");
        assert_eq!(sanitize_id(&canonical_stem(path)), "deepseek-v4-iq2");
    }

    #[test]
    fn model_id_is_api_safe() {
        assert_eq!(
            sanitize_id("Publisher/Model Name (Q8)"),
            "publisher-model-name--q8-"
        );
    }

    #[test]
    fn fingerprint_changes_when_a_model_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        fs::write(&path, b"GGUF").unwrap();
        let cfg = Config {
            models_dirs: vec![dir.path().to_owned()],
            ..Config::default()
        };
        let before = models_fingerprint(&cfg);
        fs::write(&path, b"GGUF changed").unwrap();
        assert_ne!(before, models_fingerprint(&cfg));
    }

    fn test_vocab(eos: u64, add_eos: bool) -> TokenizerMetadata {
        TokenizerMetadata {
            token_count: 10,
            prefix_hashes: BTreeMap::from([(10, "same-vocabulary".into())]),
            eos_token_id: Some(eos),
            add_eos_token: Some(add_eos),
            ..TokenizerMetadata::default()
        }
    }

    #[test]
    fn draft_allows_unused_eos_ids_to_differ() {
        let main = test_vocab(106, false);
        let draft = test_vocab(1, false);
        assert_eq!(
            draft_tokenizer_compatibility(&main, &draft),
            DraftCompatibility::Compatible
        );
    }

    #[test]
    fn draft_rejects_different_eos_id_when_eos_is_added() {
        let main = test_vocab(106, true);
        let draft = test_vocab(1, true);
        assert!(matches!(
            draft_tokenizer_compatibility(&main, &draft),
            DraftCompatibility::Incompatible(reason) if reason.contains("EOS")
        ));
    }

    #[test]
    fn draft_rejects_different_special_token_add_settings() {
        let main = test_vocab(1, true);
        let draft = test_vocab(1, false);
        assert!(matches!(
            draft_tokenizer_compatibility(&main, &draft),
            DraftCompatibility::Incompatible(reason) if reason.contains("EOS")
        ));
    }
}

#[cfg(test)]
mod draft_tests {
    use super::*;

    #[test]
    fn classifies_real_model_files() {
        let cases = [
            (
                "/home/popstarts/.lmstudio/models/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/mtp-gemma-4-12B-it.gguf",
                true,
                "gemma4-assistant draft",
            ),
            (
                "/home/popstarts/.lmstudio/models/HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP/mtp-gemma-4-31B-it.gguf",
                true,
                "gemma4-assistant draft",
            ),
            (
                "/home/popstarts/.lmstudio/models/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
                false,
                "main model",
            ),
            (
                "/home/popstarts/.lmstudio/models/AIOpsInSpace/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-MTP/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-MTP-Q8_K_P.gguf",
                false,
                "MTP-enabled main (must NOT be draft)",
            ),
            (
                "/home/popstarts/.lmstudio/models/morikomorizz/Qwen3.6-35B-A3B-Uncensored-HauhauCS-MTP/Qwen3.6-35B-A3B-Uncensored-HauhauCS-MTP-Q8_K_P.gguf",
                false,
                "MTP-enabled main (must NOT be draft)",
            ),
            (
                "/home/popstarts/.lmstudio/models/HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
                false,
                "main model",
            ),
            (
                "/home/popstarts/.lmstudio/models/Blackfrost-Research/Muse-Glimmer-30B-Abliterated-GGUF/dflash-Muse-Glimmer-30B-Abliterated-F16.gguf",
                true,
                "dflash draft (dflash arch)",
            ),
            (
                "/home/popstarts/.lmstudio/models/Blackfrost-Research/Muse-Glimmer-30B-Abliterated-GGUF/dflash-Muse-Glimmer-30B-kquant.gguf",
                true,
                "dflash draft (dflash arch)",
            ),
            (
                "/home/popstarts/.lmstudio/models/Myric/Laguna-S-2.1-APEX-GGUF/laguna-s-2.1-DFlash-Q8_0.gguf",
                true,
                "laguna dflash draft (dflash arch)",
            ),
            (
                "/home/popstarts/.lmstudio/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-BF16.gguf",
                true,
                "dspark draft (dflash arch)",
            ),
            (
                "/home/popstarts/.lmstudio/models/Blackfrost-Research/Muse-Glimmer-30B-Abliterated-GGUF/Muse-Glimmer-30B-Abliterated-Q8_0.gguf",
                false,
                "muse main (muse-glimmer arch)",
            ),
            (
                "/home/popstarts/.lmstudio/models/Myric/Laguna-S-2.1-APEX-GGUF/Laguna-S-2.1-APEX-i-quality.gguf",
                false,
                "laguna main (laguna arch)",
            ),
        ];
        for (path, expected, label) in cases {
            let path = Path::new(path);
            if !path.is_file() {
                eprintln!("SKIP (missing): {label} -> {path:?}");
                continue;
            }
            assert_eq!(is_draft(path), expected, "{label}: {path:?} misclassified");
        }
    }
}
