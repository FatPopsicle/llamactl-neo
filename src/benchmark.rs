use crate::{
    config::{Config, Paths, atomic_json},
    models, process,
    profiles::Profiles,
};
use anyhow::{Context, Result, bail};
use reqwest::{
    blocking::{Client, Response},
    header::{AUTHORIZATION, HeaderMap, HeaderValue},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader},
    os::unix::process::CommandExt,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::Sender,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_GENERATION_TOKENS: u64 = 700;
const CODING_TOKENS: u64 = 512;
const PROSE_TOKENS: u64 = 700;


pub const PREFILL_CASES: [(&str, f64); 3] = [
    ("prefill-short", 0.05),
    ("prefill-medium", 0.25),
    ("prefill-long", 0.75),
];


pub const TOTAL_CASES: usize = PREFILL_CASES.len() + 3 * 2;

const CODING_PROMPT: &str = "Write a complete, correct Python module implementing a Red-Black tree with insert, delete, and search. Include left/right rotation helpers, docstrings with type hints, and a self-test under `if __name__ == '__main__':` that inserts, searches, and deletes several values and prints the inorder traversal. The code must run as-is.";
const PROSE_EN_PROMPT: &str = "Write a 500-word short story in English about a lighthouse keeper who finds a message in a bottle that changes everything. Use vivid sensory imagery and give it a clear beginning, middle, and end.";
const PROSE_ZH_PREFIX: &str = "Translate the following English short story into natural, fluent Simplified Chinese. Preserve tone, imagery, and paragraph breaks:\n\n";
const CANCEL_TERM_GRACE: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProfileBenchmarks {
    pub profiles: BTreeMap<String, Vec<BenchmarkRun>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub timestamp_unix: u64,
    pub profile: String,
    pub profile_hash: String,
    pub model: String,
    pub model_path: String,
    pub model_bytes: u64,
    pub runtime: String,
    pub runtime_version: String,
    pub backend: String,
    pub effective_context: u64,
    pub output_tokens: u64,
    pub load_ms: u64,
    pub effective_args: Vec<String>,
    pub cases: Vec<BenchmarkCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCase {
    pub name: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub concurrency: u64,
    pub target_prompt_tokens: u64,
    pub actual_prompt_tokens: u64,
    pub actual_decode_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub prompt_ms: f64,
    pub prompt_tokens_per_second: f64,
    pub decode_ms: f64,
    pub decode_tokens_per_second: f64,
    pub decode_peak_tokens_per_second: f64,
    pub decode_median_tokens_per_second: f64,
    pub time_to_first_response_ms: f64,
    pub peak_vram_bytes: u64,
    pub peak_ram_bytes: u64,
    pub token_arrival_ms: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum BenchmarkProgress {
    Preparing,
    StoppingServer,
    LoadingRuntime,
    Ready {
        runtime: String,
        effective_context: u64,
        load_ms: u64,
    },
    CaseStarted {
        name: String,
        target_prompt_tokens: u64,
        started_at: Instant,
    },
    CaseCompleted(BenchmarkCase),
    RestoringServer,
}

impl ProfileBenchmarks {
    pub fn load(paths: &Paths) -> Result<Self> {
        if !paths.profile_benchmarks.is_file() {
            return Ok(Self::default());
        }
        let mut store: Self = serde_json::from_str(&fs::read_to_string(&paths.profile_benchmarks)?)
            .with_context(|| format!("invalid JSON in {}", paths.profile_benchmarks.display()))?;


        for case in store
            .profiles
            .values_mut()
            .flat_map(|runs| runs.iter_mut())
            .flat_map(|run| run.cases.iter_mut())
        {
            let (peak, median) = decode_distribution(&case.token_arrival_ms);
            case.decode_peak_tokens_per_second = peak;
            case.decode_median_tokens_per_second = median;
        }
        Ok(store)
    }

    pub fn save(&self, paths: &Paths) -> Result<()> {
        atomic_json(&paths.profile_benchmarks, self)
    }

    pub fn add(&mut self, profile: &str, run: BenchmarkRun) {
        let runs = self.profiles.entry(profile.to_owned()).or_default();
        runs.push(run);
        if runs.len() > 2 {
            runs.drain(..runs.len() - 2);
        }
    }

    pub fn rename(&mut self, old: &str, new: &str) {
        if let Some(mut runs) = self.profiles.remove(old) {
            for run in &mut runs {
                run.profile = new.to_owned();
            }
            self.profiles.insert(new.to_owned(), runs);
        }
    }

    pub fn remove(&mut self, profile: &str) {
        self.profiles.remove(profile);
    }
}

pub fn run_cancellable_with_progress(
    cfg: &Config,
    paths: &Paths,
    profiles: &Profiles,
    profile: &str,
    cancelled: Arc<AtomicBool>,
    progress: Sender<BenchmarkProgress>,
) -> Result<BenchmarkRun> {
    run_inner(
        cfg,
        paths,
        profiles,
        profile,
        cancelled,
        Some(progress),
        TOTAL_CASES,
    )
}

pub fn run_partial(
    cfg: &Config,
    paths: &Paths,
    profiles: &Profiles,
    profile: &str,
    cancelled: Arc<AtomicBool>,
    max_cases: usize,
) -> Result<BenchmarkRun> {
    run_inner(cfg, paths, profiles, profile, cancelled, None, max_cases)
}

fn run_inner(
    cfg: &Config,
    paths: &Paths,
    profiles: &Profiles,
    profile: &str,
    cancelled: Arc<AtomicBool>,
    progress: Option<Sender<BenchmarkProgress>>,
    max_cases: usize,
) -> Result<BenchmarkRun> {
    emit(&progress, BenchmarkProgress::Preparing);
    if cancelled.load(Ordering::Relaxed) {
        bail!("benchmark cancelled")
    }
    let profile_value = profiles
        .profiles
        .get(profile)
        .with_context(|| format!("unknown profile '{profile}'"))?;
    let owner = profiles
        .owner(profile)
        .with_context(|| format!("profile '{profile}' has no owner model"))?;
    let (_, model_path, model_id) = models::resolve(cfg, Some(owner))?;
    let model_path = model_path.context("profile owner is not a local model")?;

    let mut effective_args = process::common_args(cfg);
    effective_args.extend(profiles.args(profile)?);
    let configured_context = flag_u64(&effective_args, &["--ctx-size", "-c"])
        .unwrap_or(cfg.ctx_size)
        .max(512);
    let effective_context = models::context_limit(&model_path)
        .map(|limit| configured_context.min(limit))
        .unwrap_or(configured_context);
    if effective_context <= MAX_GENERATION_TOKENS + 32 {
        bail!("effective context {effective_context} is too small to benchmark")
    }

    let mut bench_cfg = cfg.clone();
    bench_cfg.host = "127.0.0.1".into();
    let (binary, args, swap) =
        process::build_command(&bench_cfg, paths, profiles, Some(profile), &[])?;
    if swap {
        bail!("profile benchmark unexpectedly resolved to llama-swap")
    }
    let base = format!("http://127.0.0.1:{}", bench_cfg.port);
    let mut headers = HeaderMap::new();
    if let Some(key) = bench_cfg.keys().first() {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {key}"))?,
        );
    }
    let client = Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(900))
        .build()?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log)?;
    let err = log.try_clone()?;
    let was_running = process::pid(paths).is_some();
    let service_was_active = service_active();
    if was_running {
        emit(&progress, BenchmarkProgress::StoppingServer);
        process::stop(paths)?;
        if cancelled.load(Ordering::Relaxed) {
            restore_server(cfg, paths, profiles, was_running, service_was_active)?;
            bail!("benchmark cancelled")
        }
    }
    emit(&progress, BenchmarkProgress::LoadingRuntime);
    let started = Instant::now();
    let child_result = process::runtime_command(&binary, paths)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(err)
        .process_group(0)
        .spawn()
        .with_context(|| format!("start benchmark runtime {}", binary.display()));
    let mut child = match child_result {
        Ok(child) => child,
        Err(error) => {
            let _ = restore_server(cfg, paths, profiles, was_running, service_was_active);
            return Err(error);
        }
    };

    let child_id = child.id();
    let watcher_done = Arc::new(AtomicBool::new(false));
    let watcher = {
        let cancelled = cancelled.clone();
        let done = watcher_done.clone();
        thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                if cancelled.load(Ordering::Relaxed) {
                    let process_group = format!("-{child_id}");
                    let _ = Command::new("kill")
                        .args(["-TERM", &process_group])
                        .status();
                    let deadline = Instant::now() + CANCEL_TERM_GRACE;
                    while Instant::now() < deadline && !done.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_millis(50));
                    }
                    if !done.load(Ordering::Relaxed) {
                        let _ = Command::new("kill")
                            .args(["-KILL", &process_group])
                            .status();
                    }
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
        })
    };
    let mut partial_run = None;
    let result = (|| {
        wait_ready(&client, &base, &mut child, &cancelled)?;
        let load_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let props = runtime_props(&client, &base);
        let effective_context = props.map(|(ctx, _)| ctx).unwrap_or(effective_context);
        if effective_context <= MAX_GENERATION_TOKENS + 32 {
            bail!("runtime context {effective_context} is too small to benchmark")
        }
        emit(
            &progress,
            BenchmarkProgress::Ready {
                runtime: resolved_runtime(cfg, paths),
                effective_context,
                load_ms,
            },
        );
        let seed_tokens = tokenize(&client, &base)?;
        if seed_tokens.is_empty() {
            bail!("runtime tokenizer returned no tokens")
        }
        let available = effective_context.saturating_sub(32);
        let profile_hash = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(profile_value).unwrap_or_default())
        );
        partial_run = Some(BenchmarkRun {
            timestamp_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            profile: profile.to_owned(),
            profile_hash,
            model: model_id,
            model_path: model_path.display().to_string(),
            model_bytes: models::model_bytes(&model_path),
            runtime: resolved_runtime(cfg, paths),
            runtime_version: runtime_version(paths),
            backend: cfg.backend.clone(),
            effective_context,
            output_tokens: MAX_GENERATION_TOKENS,
            load_ms,
            effective_args,
            cases: Vec::with_capacity(TOTAL_CASES),
        });
        let slots = props.map(|(_, slots)| slots).unwrap_or(1).max(1);
        let mut next = 0usize;


        for (name, fraction) in [PREFILL_CASES[0], PREFILL_CASES[1]] {
            if next >= max_cases {
                break;
            }
            let target =
                ((effective_context as f64 * fraction).round() as u64).clamp(32, available);
            let prompt = prefill_prompt(&seed_tokens, next, target);
            record_case(
                &progress,
                &mut partial_run,
                &cancelled,
                &mut next,
                max_cases,
                name,
                target,
                || run_prefill(&client, &base, name, target, prompt, child_id),
            )?;
        }


        record_case(
            &progress,
            &mut partial_run,
            &cancelled,
            &mut next,
            max_cases,
            "coding-single",
            0,
            || {
                run_generation(
                    &client,
                    &base,
                    "coding-single",
                    "coding",
                    CODING_PROMPT.to_owned(),
                    CODING_TOKENS,
                    child_id,
                )
                .map(|(case, _)| case)
            },
        )?;


        let mut story_single = String::new();
        let mut stories_slots: Vec<String> = Vec::new();

        record_case(
            &progress,
            &mut partial_run,
            &cancelled,
            &mut next,
            max_cases,
            "prose-en-single",
            0,
            || {
                let (case, story) = run_generation(
                    &client,
                    &base,
                    "prose-en-single",
                    "prose",
                    PROSE_EN_PROMPT.to_owned(),
                    PROSE_TOKENS,
                    child_id,
                )?;
                story_single = story;
                Ok(case)
            },
        )?;

        record_case(
            &progress,
            &mut partial_run,
            &cancelled,
            &mut next,
            max_cases,
            "prose-zh-single",
            0,
            || {
                let zh_prompt = format!("{PROSE_ZH_PREFIX}{story_single}");
                run_generation(
                    &client,
                    &base,
                    "prose-zh-single",
                    "prose",
                    zh_prompt,
                    PROSE_TOKENS,
                    child_id,
                )
                .map(|(case, _)| case)
            },
        )?;


        if next < max_cases {
            let (name, fraction) = PREFILL_CASES[2];
            let target =
                ((effective_context as f64 * fraction).round() as u64).clamp(32, available);
            let prompt = prefill_prompt(&seed_tokens, next, target);
            record_case(
                &progress,
                &mut partial_run,
                &cancelled,
                &mut next,
                max_cases,
                name,
                target,
                || run_prefill(&client, &base, name, target, prompt, child_id),
            )?;
        }


        if slots > 1 {
            record_case(
                &progress,
                &mut partial_run,
                &cancelled,
                &mut next,
                max_cases,
                "coding-slots",
                0,
                || {
                    run_generation_multi(
                        &client,
                        &base,
                        "coding-slots",
                        "coding",
                        vec![CODING_PROMPT.to_owned(); slots as usize],
                        CODING_TOKENS,
                        child_id,
                    )
                    .map(|(case, _)| case)
                },
            )?;
        }

        if slots > 1 {
            record_case(
                &progress,
                &mut partial_run,
                &cancelled,
                &mut next,
                max_cases,
                "prose-en-slots",
                0,
                || {
                    let (case, stories) = run_generation_multi(
                        &client,
                        &base,
                        "prose-en-slots",
                        "prose",
                        vec![PROSE_EN_PROMPT.to_owned(); slots as usize],
                        PROSE_TOKENS,
                        child_id,
                    )?;
                    stories_slots = stories;
                    Ok(case)
                },
            )?;
        }

        if slots > 1 {
            record_case(
                &progress,
                &mut partial_run,
                &cancelled,
                &mut next,
                max_cases,
                "prose-zh-slots",
                0,
                || {
                    let zh_prompts = stories_slots
                        .iter()
                        .map(|story| format!("{PROSE_ZH_PREFIX}{story}"))
                        .collect::<Vec<_>>();
                    run_generation_multi(
                        &client,
                        &base,
                        "prose-zh-slots",
                        "prose",
                        zh_prompts,
                        PROSE_TOKENS,
                        child_id,
                    )
                    .map(|(case, _)| case)
                },
            )?;
        }
        Ok(partial_run
            .take()
            .expect("benchmark run metadata initialized"))
    })();

    emit(&progress, BenchmarkProgress::RestoringServer);
    terminate(&mut child);
    watcher_done.store(true, Ordering::Relaxed);
    let _ = watcher.join();
    let restore = restore_server(cfg, paths, profiles, was_running, service_was_active);
    let was_cancelled = cancelled.load(Ordering::Relaxed);
    let run = match result {
        Ok(run) => run,
        Err(error) if was_cancelled => match partial_run.take() {
            Some(run) if !run.cases.is_empty() => run,
            _ => {
                restore?;
                return Err(error);
            }
        },
        Err(error) => {
            restore?;
            return Err(error);
        }
    };
    let mut store = ProfileBenchmarks::load(paths)?;
    store.add(profile, run.clone());
    store.save(paths)?;
    restore?;
    Ok(run)
}

fn emit(progress: &Option<Sender<BenchmarkProgress>>, update: BenchmarkProgress) {
    if let Some(progress) = progress {
        let _ = progress.send(update);
    }
}

#[allow(clippy::too_many_arguments)]
fn record_case(
    progress: &Option<Sender<BenchmarkProgress>>,
    partial_run: &mut Option<BenchmarkRun>,
    cancelled: &AtomicBool,
    next: &mut usize,
    max_cases: usize,
    name: &str,
    target: u64,
    run: impl FnOnce() -> Result<BenchmarkCase>,
) -> Result<()> {
    if *next >= max_cases {
        return Ok(());
    }
    if cancelled.load(Ordering::Relaxed) {
        bail!("benchmark cancelled")
    }
    emit(progress, BenchmarkProgress::CaseStarted {
        name: name.to_owned(),
        target_prompt_tokens: target,
        started_at: Instant::now(),
    });
    let case = run()?;
    emit(progress, BenchmarkProgress::CaseCompleted(case.clone()));
    partial_run
        .as_mut()
        .expect("benchmark run metadata initialized")
        .cases
        .push(case);
    *next += 1;
    Ok(())
}

pub fn summary(run: &BenchmarkRun) -> String {
    let values = run
        .cases
        .iter()
        .map(|case| {
            if case.kind == "prefill" {
                format!(
                    "{} pp {:.1} ({:.0} tok)",
                    case.name, case.prompt_tokens_per_second, case.actual_prompt_tokens as f64
                )
            } else {
                format!(
                    "{} x{} dec {:.1} - median {:.1}",
                    case.name,
                    case.concurrency,
                    case.decode_tokens_per_second,
                    case.decode_median_tokens_per_second
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let status = if run.cases.len() < TOTAL_CASES {
        " - partial"
    } else {
        ""
    };
    format!(
        "profile {}{status} - runtime {} - ctx {} - load {:.2}s\n{}",
        run.profile,
        run.runtime,
        run.effective_context,
        run.load_ms as f64 / 1000.0,
        values
    )
}

fn run_prefill(
    client: &Client,
    base: &str,
    name: &str,
    target: u64,
    prompt: Vec<i64>,
    runtime_pid: u32,
) -> Result<BenchmarkCase> {
    let sampler = MemorySampler::start(runtime_pid);
    let response: Value = client
        .post(format!("{base}/completion"))
        .json(&json!({
            "prompt": prompt,
            "n_predict": 0,
            "temperature": 0.0,
            "seed": 1,
            "cache_prompt": false,
            "stream": false
        }))
        .send()?
        .error_for_status()?
        .json()?;
    let (peak_vram_bytes, peak_ram_bytes) = sampler.stop();

    let timings = response.get("timings").cloned().unwrap_or(Value::Null);
    let actual_prompt = timings
        .get("prompt_n")
        .and_then(Value::as_u64)
        .unwrap_or(target);
    let prompt_ms = timings
        .get("prompt_ms")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let pp = timings
        .get("prompt_per_second")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| rate(actual_prompt, prompt_ms));
    Ok(BenchmarkCase {
        name: name.to_owned(),
        kind: "prefill".to_owned(),
        concurrency: 1,
        target_prompt_tokens: target,
        actual_prompt_tokens: actual_prompt,
        actual_decode_tokens: 0,
        cached_prompt_tokens: 0,
        prompt_ms,
        prompt_tokens_per_second: pp,
        decode_ms: 0.0,
        decode_tokens_per_second: 0.0,
        decode_peak_tokens_per_second: 0.0,
        decode_median_tokens_per_second: 0.0,
        time_to_first_response_ms: prompt_ms,
        peak_vram_bytes,
        peak_ram_bytes,
        token_arrival_ms: Vec::new(),
    })
}

fn run_generation(
    client: &Client,
    base: &str,
    name: &str,
    kind: &str,
    prompt: String,
    n_predict: u64,
    runtime_pid: u32,
) -> Result<(BenchmarkCase, String)> {
    let sampler = MemorySampler::start(runtime_pid);
    let request_started = Instant::now();
    let mut response = client
        .post(format!("{base}/completion"))
        .json(&json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "seed": 1,
            "stream": true,
            "cache_prompt": false,
            "return_tokens": true,
            "ignore_eos": true
        }))
        .send()?
        .error_for_status()?;
    let mut arrivals = Vec::with_capacity(n_predict as usize);
    let mut final_event = Value::Null;
    let mut content = String::new();
    read_completion_stream(
        &mut response,
        request_started,
        &mut arrivals,
        &mut final_event,
        &mut content,
    )?;
    let (peak_vram_bytes, peak_ram_bytes) = sampler.stop();

    let timings = final_event.get("timings").unwrap_or(&Value::Null);
    let actual_prompt = timings
        .get("prompt_n")
        .and_then(Value::as_u64)
        .or_else(|| final_event.get("tokens_evaluated").and_then(Value::as_u64))
        .unwrap_or(0);
    let actual_decode = timings
        .get("predicted_n")
        .and_then(Value::as_u64)
        .unwrap_or(arrivals.len() as u64);
    let cached_prompt_tokens = final_event
        .get("tokens_cached")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let prompt_ms = timings
        .get("prompt_ms")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let decode_ms = timings
        .get("predicted_ms")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let pp = timings
        .get("prompt_per_second")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| rate(actual_prompt, prompt_ms));
    let decode = timings
        .get("predicted_per_second")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| rate(actual_decode, decode_ms));
    let first = arrivals.first().copied().unwrap_or_default();
    let (decode_peak, decode_median) = decode_distribution(&arrivals);
    let case = BenchmarkCase {
        name: name.to_owned(),
        kind: kind.to_owned(),
        concurrency: 1,
        target_prompt_tokens: 0,
        actual_prompt_tokens: actual_prompt,
        actual_decode_tokens: actual_decode,
        cached_prompt_tokens,
        prompt_ms,
        prompt_tokens_per_second: pp,
        decode_ms,
        decode_tokens_per_second: decode,
        decode_peak_tokens_per_second: decode_peak,
        decode_median_tokens_per_second: decode_median,
        time_to_first_response_ms: first,
        peak_vram_bytes,
        peak_ram_bytes,
        token_arrival_ms: arrivals,
    };
    Ok((case, content))
}

fn run_generation_multi(
    client: &Client,
    base: &str,
    name: &str,
    kind: &str,
    prompts: Vec<String>,
    n_predict: u64,
    runtime_pid: u32,
) -> Result<(BenchmarkCase, Vec<String>)> {
    let concurrency = prompts.len() as u64;
    let sampler = MemorySampler::start(runtime_pid);
    let started = Instant::now();
    let handles: Vec<_> = prompts
        .into_iter()
        .enumerate()
        .map(|(seed, prompt)| {
            let client = client.clone();
            let base = base.to_string();
            thread::spawn(move || -> Result<(u64, u64, String)> {
                let response: Value = client
                    .post(format!("{base}/completion"))
                    .json(&json!({
                        "prompt": prompt,
                        "n_predict": n_predict,
                        "temperature": 0.0,
                        "seed": seed as u64,
                        "cache_prompt": false,
                        "stream": false
                    }))
                    .send()?
                    .error_for_status()?
                    .json()?;
                let timings = response.get("timings").cloned().unwrap_or_default();
                let prompt_n = timings.get("prompt_n").and_then(Value::as_u64).unwrap_or(0);
                let decode_n = timings
                    .get("predicted_n")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let content = response
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                Ok((prompt_n, decode_n, content))
            })
        })
        .collect();

    let mut prompt_tokens = 0u64;
    let mut decode_tokens = 0u64;
    let mut contents = Vec::with_capacity(concurrency as usize);
    for handle in handles {
        match handle.join() {
            Ok(Ok((prompt_n, decode_n, content))) => {
                prompt_tokens += prompt_n;
                decode_tokens += decode_n;
                contents.push(content);
            }
            Ok(Err(error)) => return Err(error),
            Err(_) => bail!("benchmark worker thread panicked"),
        }
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let (peak_vram_bytes, peak_ram_bytes) = sampler.stop();
    let case = BenchmarkCase {
        name: name.to_owned(),
        kind: kind.to_owned(),
        concurrency,
        target_prompt_tokens: 0,
        actual_prompt_tokens: prompt_tokens,
        actual_decode_tokens: decode_tokens,
        cached_prompt_tokens: 0,
        prompt_ms: wall_ms,
        prompt_tokens_per_second: rate(prompt_tokens, wall_ms),
        decode_ms: wall_ms,
        decode_tokens_per_second: rate(decode_tokens, wall_ms),
        decode_peak_tokens_per_second: 0.0,
        decode_median_tokens_per_second: 0.0,
        time_to_first_response_ms: 0.0,
        peak_vram_bytes,
        peak_ram_bytes,
        token_arrival_ms: Vec::new(),
    };
    Ok((case, contents))
}

fn read_completion_stream(
    response: &mut Response,
    started: Instant,
    arrivals: &mut Vec<f64>,
    final_event: &mut Value,
    content: &mut String,
) -> Result<()> {
    for line in BufReader::new(response).lines() {
        let line = line?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let event: Value = serde_json::from_str(data)?;
        if let Some(text) = event.get("content").and_then(Value::as_str) {
            content.push_str(text);
        }
        let token_count = event
            .get("tokens")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or_else(|| {
                usize::from(
                    event
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(|content| !content.is_empty()),
                )
            });
        let at = started.elapsed().as_secs_f64() * 1000.0;
        arrivals.extend(std::iter::repeat_n(at, token_count));
        if event.get("stop").and_then(Value::as_bool).unwrap_or(false)
            || event.get("timings").is_some()
        {
            *final_event = event;
        }
    }
    if final_event.is_null() {
        bail!("completion stream ended without final timings")
    }
    Ok(())
}

fn runtime_props(client: &Client, base: &str) -> Option<(u64, u64)> {
    let props = client
        .get(format!("{base}/props"))
        .send()
        .ok()?
        .json::<Value>()
        .ok()?;
    let ctx = props
        .get("default_generation_settings")?
        .get("n_ctx")?
        .as_u64()?;
    let slots = props.get("total_slots")?.as_u64().unwrap_or(1);
    Some((ctx, slots))
}

fn tokenize(client: &Client, base: &str) -> Result<Vec<i64>> {
    let response = client
        .post(format!("{base}/tokenize"))
        .json(&json!({
            "content": "The quick brown fox jumps over the lazy dog. Profile benchmark prompt processing sequence. ",
            "add_special": true
        }))
        .send()?
        .error_for_status()?
        .json::<Value>()?;
    Ok(response
        .get("tokens")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_i64)
        .collect())
}

fn repeated_prompt(seed: &[i64], target: usize) -> Vec<i64> {
    let mut prompt = Vec::with_capacity(target);
    while prompt.len() < target {
        let remaining = target - prompt.len();
        prompt.extend(seed.iter().take(remaining));
    }
    prompt
}

fn prefill_prompt(seed: &[i64], index: usize, target: u64) -> Vec<i64> {
    let mut case_seed = seed.to_vec();
    if !case_seed.is_empty() {
        let length = case_seed.len();
        case_seed.rotate_left((index * 7) % length);
    }
    repeated_prompt(&case_seed, target as usize)
}

fn wait_ready(
    client: &Client,
    base: &str,
    child: &mut std::process::Child,
    cancelled: &AtomicBool,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(900);
    while Instant::now() < deadline {
        if cancelled.load(Ordering::Relaxed) {
            bail!("benchmark cancelled")
        }
        if let Some(status) = child.try_wait()? {
            bail!("benchmark runtime exited during load with {status}")
        }
        if client
            .get(format!("{base}/health"))
            .send()
            .is_ok_and(|response| response.status().is_success())
        {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(250));
    }
    bail!("benchmark runtime did not become ready within 15 minutes")
}

fn service_active() -> bool {
    Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", "llamactl.service"])
        .status()
        .is_ok_and(|status| status.success())
}

fn restore_server(
    cfg: &Config,
    paths: &Paths,
    profiles: &Profiles,
    was_running: bool,
    service_was_active: bool,
) -> Result<()> {
    if !was_running {
        return Ok(());
    }
    if service_was_active {
        let status = Command::new("systemctl")
            .args(["--user", "start", "llamactl.service"])
            .status()?;
        if !status.success() {
            bail!("benchmark completed, but the llamactl service could not be restarted")
        }
    } else {
        process::restart(cfg, paths, profiles)?;
    }
    Ok(())
}

fn terminate(child: &mut std::process::Child) {
    let _ = Command::new("kill")
        .args(["-TERM", &format!("-{}", child.id())])
        .status();
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if child.try_wait().ok().flatten().is_some() {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn flag_u64(args: &[String], names: &[&str]) -> Option<u64> {
    args.iter().enumerate().rev().find_map(|(index, arg)| {
        names
            .contains(&arg.as_str())
            .then(|| args.get(index + 1)?.parse().ok())?
    })
}

fn resolved_runtime(cfg: &Config, paths: &Paths) -> String {
    if cfg.runtime != "managed" {
        return cfg.runtime.clone();
    }
    fs::canonicalize(&paths.current)
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .map(|name| format!("managed:{name}"))
        .unwrap_or_else(|| "managed".into())
}

fn runtime_version(paths: &Paths) -> String {
    let Some(binary) = process::server_binary(paths) else {
        return "unknown".into();
    };
    process::runtime_command(&binary, paths)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or("unknown")
                .trim()
                .to_owned()
        })
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn rate(tokens: u64, milliseconds: f64) -> f64 {
    if milliseconds > 0.0 {
        tokens as f64 * 1000.0 / milliseconds
    } else {
        0.0
    }
}

fn decode_distribution(arrivals: &[f64]) -> (f64, f64) {
    let window_tokens = 16usize.min(arrivals.len().saturating_sub(1));
    if window_tokens == 0 {
        return (0.0, 0.0);
    }
    let mut rates = arrivals
        .windows(window_tokens + 1)
        .filter_map(|window| {
            let elapsed = window[window_tokens] - window[0];
            (elapsed > 0.0).then_some(window_tokens as f64 * 1000.0 / elapsed)
        })
        .collect::<Vec<_>>();
    if rates.is_empty() {
        return (0.0, 0.0);
    }
    let peak = rates.iter().copied().fold(0.0, f64::max);
    rates.sort_by(f64::total_cmp);
    let median = if rates.len() % 2 == 1 {
        rates[rates.len() / 2]
    } else {
        (rates[rates.len() / 2 - 1] + rates[rates.len() / 2]) / 2.0
    };
    (peak, median)
}

struct MemorySampler {
    stop: Arc<AtomicBool>,
    peak_vram: Arc<AtomicU64>,
    peak_ram: Arc<AtomicU64>,
    thread: Option<thread::JoinHandle<()>>,
}

impl MemorySampler {
    fn start(runtime_pid: u32) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak_vram = Arc::new(AtomicU64::new(0));
        let peak_ram = Arc::new(AtomicU64::new(0));
        let thread_stop = stop.clone();
        let thread_vram = peak_vram.clone();
        let thread_ram = peak_ram.clone();
        let handle = thread::spawn(move || {
            let mut sample = 0u64;
            while !thread_stop.load(Ordering::Relaxed) {
                let drm = crate::drm::read().for_pid(runtime_pid as i32);
                let vram = if drm == 0 && sample.is_multiple_of(10) {
                    nvidia_process_vram(runtime_pid)
                } else {
                    drm
                };
                thread_vram.fetch_max(vram, Ordering::Relaxed);
                thread_ram.fetch_max(process_rss_bytes(runtime_pid), Ordering::Relaxed);
                sample += 1;
                thread::sleep(Duration::from_millis(100));
            }
        });
        Self {
            stop,
            peak_vram,
            peak_ram,
            thread: Some(handle),
        }
    }

    fn stop(mut self) -> (u64, u64) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
        (
            self.peak_vram.load(Ordering::Relaxed),
            self.peak_ram.load(Ordering::Relaxed),
        )
    }
}

fn process_rss_bytes(pid: u32) -> u64 {
    let text = fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    text.lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|line| line.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
        * 1024
}

fn nvidia_process_vram(pid: u32) -> u64 {
    Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter_map(|line| line.split_once(','))
                .filter(|(process, _)| process.trim().parse::<u32>().ok() == Some(pid))
                .filter_map(|(_, memory)| memory.trim().parse::<u64>().ok())
                .sum::<u64>()
                * 1024
                * 1024
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_only_two_runs_per_profile() {
        let mut store = ProfileBenchmarks::default();
        for timestamp in 1..=3 {
            store.add(
                "p",
                BenchmarkRun {
                    timestamp_unix: timestamp,
                    profile: "p".into(),
                    profile_hash: String::new(),
                    model: String::new(),
                    model_path: String::new(),
                    model_bytes: 0,
                    runtime: String::new(),
                    runtime_version: String::new(),
                    backend: String::new(),
                    effective_context: 4096,
                    output_tokens: MAX_GENERATION_TOKENS,
                    load_ms: 0,
                    effective_args: vec![],
                    cases: vec![],
                },
            );
        }
        let runs = &store.profiles["p"];
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].timestamp_unix, 2);
        assert_eq!(runs[1].timestamp_unix, 3);
    }

    #[test]
    fn repeated_prompt_has_exact_length() {
        assert_eq!(repeated_prompt(&[1, 2, 3], 8), vec![1, 2, 3, 1, 2, 3, 1, 2]);
    }

    #[test]
    fn calculates_decode_peak_and_median() {
        let arrivals = (0..=32)
            .map(|index| index as f64 * 20.0)
            .collect::<Vec<_>>();
        let (peak, median) = decode_distribution(&arrivals);
        assert!((peak - 50.0).abs() < 0.001);
        assert!((median - 50.0).abs() < 0.001);
    }

    #[test]
    fn rolling_median_ignores_bursty_single_token_intervals() {
        let arrivals = (0..=64)
            .map(|index| (index / 4) as f64 * 40.0 + (index % 4) as f64 * 0.025)
            .collect::<Vec<_>>();
        let (_, median) = decode_distribution(&arrivals);
        assert!(median > 75.0 && median < 125.0, "median was {median}");
    }
}
