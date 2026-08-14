use crate::{
    benchmark,
    config::{Config, Paths},
    models, process, profiles,
    profiles::Profiles,
};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::border,
    text::{Line, Span},
    widgets::{
        Block, BorderType, Borders, Clear, Gauge, List, ListItem, ListState, Padding, Paragraph,
        Row, Table, Wrap,
    },
};
use serde_json::Value;
use std::{
    collections::{BTreeMap, VecDeque},
    fs, io,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};
use throbber_widgets_tui::{Throbber, ThrobberState};
use tui_checkbox::Checkbox;

const PAGES: &[&str] = &[
    "Dashboard",
    "Models",
    "Profiles",
    "Settings",
    "Logs",
    "Maintenance",
];
type SlotSamples = BTreeMap<(String, usize), VecDeque<(Instant, u64, u64)>>;

type UpdateStatus = (String, bool, String, bool);

struct App<'a> {
    cfg: Config,
    paths: &'a Paths,
    profiles: Profiles,
    benchmarks: benchmark::ProfileBenchmarks,
    page: usize,
    selected: usize,
    filter: String,
    rename_input: Option<RenameState>,
    profile_delete_confirm: Option<String>,
    key_help: bool,
    profile_editor: Option<ProfileEditor>,
    benchmark_view: Option<BenchmarkView>,
    benchmark_dialog: Option<BenchmarkDialog>,
    runtime_picker: Option<RuntimePicker>,
    background: Option<BackgroundTask>,
    last_tok: BTreeMap<String, f64>,
    profile_estimates: BTreeMap<String, (f64, f64)>,
    profile_fingerprint: String,
    notice: String,
    models: Vec<models::Model>,
    models_fingerprint: Option<u64>,
    scan_rx: Option<std::sync::mpsc::Receiver<(u64, Option<Vec<models::Model>>)>>,
    estimate_rx: Option<std::sync::mpsc::Receiver<ProfileEstimateUpdate>>,
    estimate_target: Option<String>,
    telemetry_rx: Option<std::sync::mpsc::Receiver<(Telemetry, Option<Telemetry>)>>,
    last_check: Option<UpdateStatus>,
    update_check_rx: Option<std::sync::mpsc::Receiver<Result<UpdateStatus>>>,
    log: String,
    last_refresh: Instant,
    last_telemetry: Instant,
    telemetry: Telemetry,
    token_sample: Option<(u64, Instant)>,
    slot_samples: SlotSamples,
    marquee_started: Instant,
    throbber_state: ThrobberState,
}

#[derive(Clone, Default)]
struct Telemetry {
    vram_used: u64,
    vram_total: u64,
    ram_used: u64,
    ram_total: u64,
    gpu_temps: Vec<f64>,
    prompt_done: u64,
    prompt_total: u64,
    generated: u64,
    tokens_per_second: Option<f64>,
    total_requests: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_cache_tokens: u64,
    last_request: Option<RequestPerformance>,
    historical_tok_s: BTreeMap<String, f64>,
    active_requests: usize,
    model_name: String,
    model_state: ModelState,
    llama_cpp_version: String,
    slot_details: Vec<SlotDetail>,
    slot_count: usize,
}

#[derive(Clone, Default)]
struct RequestPerformance {
    model: String,
    prompt_tokens: u64,
    output_tokens: u64,
    cache_tokens: u64,
    draft_tokens: u64,
    draft_accepted: u64,
    prompt_tok_s: f64,
    generation_tok_s: f64,
    ttft_ms: Option<f64>,
    duration_ms: u64,
}

#[derive(Clone, Default)]
struct SlotDetail {
    model_name: String,
    slot_id: usize,
    prompt_progress: f64,
    decoded: u64,
    prompt_done: u64,
    pp_tok_s: Option<f64>,
    td_tok_s: Option<f64>,
    is_processing: bool,
}

#[derive(Clone)]
struct RenameState {
    original: String,
    text: String,
    cursor: usize,
}

struct RuntimePicker {
    options: Vec<String>,
    selected: usize,
}

struct BenchmarkView {
    runs: Vec<benchmark::BenchmarkRun>,
}

enum BenchmarkDialog {
    Confirm {
        profile: String,
    },
    Running {
        profile: String,
        cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
        result_rx: std::sync::mpsc::Receiver<Result<String>>,
        progress_rx: std::sync::mpsc::Receiver<benchmark::BenchmarkProgress>,
        phase: String,
        runtime: String,
        effective_context: Option<u64>,
        load_ms: Option<u64>,
        completed: Vec<benchmark::BenchmarkCase>,
    },
}

#[derive(Clone)]
struct EditorSettings {
    ctx: u64,
    context_step: u64,
    parallel: u64,
    batch: u64,
    ubatch: u64,
    cache_k: String,
    cache_v: String,
    flash: String,
    gpu_layers: String,
    split: String,
    tensor_split: String,
    threads: u64,
    threads_batch: u64,
    kv_unified: bool,
    spec_type: String,
    spec_draft_model: String,
    spec_draft_nmax: u64,
    spec_draft_ngl: String,
    fit: String,
    fit_target: u64,
    load_mode: String,
    mlock: bool,
    direct_io: bool,
    numa: String,
    n_cpu_moe: u64,
    spec_draft_n_cpu_moe: u64,
    kv_offload: bool,
    override_tensor: String,
    rope_base: String,
    rope_scale: String,
    seed: String,
    reasoning: String,
    jinja: bool,
    chat_template: String,
    chat_template_file: String,
    chat_template_kwargs: String,
    extra: Vec<String>,
    assign: bool,
}
impl Default for EditorSettings {
    fn default() -> Self {
        Self {
            ctx: 4096,
            context_step: 4096,
            parallel: 1,
            batch: 2048,
            ubatch: 512,
            cache_k: "f16".into(),
            cache_v: "f16".into(),
            flash: "on".into(),
            gpu_layers: "all".into(),
            split: "layer".into(),
            tensor_split: "1,1,1,1".into(),
            threads: 32,
            threads_batch: 32,
            kv_unified: true,
            spec_type: "none".into(),
            spec_draft_model: String::new(),
            spec_draft_nmax: 0,
            spec_draft_ngl: "99".into(),
            fit: "on".into(),
            fit_target: 512,
            load_mode: String::new(),
            mlock: false,
            direct_io: false,
            numa: "off".into(),
            n_cpu_moe: 0,
            spec_draft_n_cpu_moe: 0,
            kv_offload: true,
            override_tensor: String::new(),
            rope_base: String::new(),
            rope_scale: String::new(),
            seed: String::new(),
            reasoning: String::new(),
            jinja: true,
            chat_template: String::new(),
            chat_template_file: String::new(),
            chat_template_kwargs: String::new(),
            extra: vec![],
            assign: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EditorField {
    Ctx,
    ContextStep,
    Parallel,
    Batch,
    Ubatch,
    CacheK,
    CacheV,
    Flash,
    GpuLayers,
    Split,
    TensorSplit,
    Threads,
    ThreadsBatch,
    KvUnified,
    SpecType,
    SpecDraftModel,
    SpecDraftNMax,
    SpecDraftNgl,
    Extra,
    Assign,
    Advanced,
    Fit,
    FitTarget,
    LoadMode,
    Mlock,
    DirectIO,
    Numa,
    KvOffload,
    CpuMoe,
    DraftCpuMoe,
    OverrideTensor,
    RopeBase,
    RopeScale,
    Seed,
    Reasoning,
    Jinja,
    ChatTemplate,
    ChatTemplateFile,
    ChatTemplateKwargs,
}

struct EditorPrompt {
    label: String,
    text: String,
    cursor: usize,
    field: EditorField,
}

struct ProfileEditor {
    name: String,
    owner: String,
    path: PathBuf,
    fields: Vec<EditorField>,
    selected: usize,
    settings: EditorSettings,
    prompt: Option<EditorPrompt>,
    estimate: EstimateState,
    estimate_rx: Option<std::sync::mpsc::Receiver<models::Estimate>>,
    advanced: bool,
    notice: String,
}

enum EstimateState {
    Pending,
    Ready(models::Estimate),
}

enum ProfileEstimateUpdate {
    Estimate(String, (f64, f64)),
    Done,
}

struct BackgroundTask {
    label: String,
    rx: std::sync::mpsc::Receiver<Result<String>>,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
enum ModelState {
    #[default]
    None,
    Loading,
    Loaded,
}
impl<'a> App<'a> {
    fn new(cfg: Config, paths: &'a Paths) -> Result<Self> {
        let profiles = Profiles::load(paths)?;
        let benchmarks = benchmark::ProfileBenchmarks::load(paths).unwrap_or_default();
        Ok(Self {
            cfg,
            paths,
            profiles,
            benchmarks,
            page: 0,
            selected: 0,
            filter: String::new(),
            rename_input: None,
            profile_delete_confirm: None,
            key_help: false,
            profile_editor: None,
            benchmark_view: None,
            benchmark_dialog: None,
            runtime_picker: None,
            background: None,
            last_tok: BTreeMap::new(),
            profile_estimates: BTreeMap::new(),
            profile_fingerprint: String::new(),
            notice: "Ready · arrows navigate · Enter acts · ? controls".into(),
            models: Vec::new(),
            models_fingerprint: None,
            scan_rx: None,
            estimate_rx: None,
            estimate_target: None,
            telemetry_rx: None,
            last_check: None,
            update_check_rx: None,
            log: String::new(),
            last_refresh: Instant::now(),
            last_telemetry: Instant::now(),
            telemetry: Telemetry::default(),
            token_sample: None,
            slot_samples: BTreeMap::new(),
            marquee_started: Instant::now(),
            throbber_state: ThrobberState::default(),
        })
    }
    fn refresh(&mut self) {
        self.cfg = Config::load(self.paths).unwrap_or_else(|_| self.cfg.clone());
        self.profiles = Profiles::load(self.paths).unwrap_or_else(|_| self.profiles.clone());
        self.benchmarks = benchmark::ProfileBenchmarks::load(self.paths)
            .unwrap_or_else(|_| self.benchmarks.clone());

        // Start estimates before the recurring model scan. Previously the scan
        // was started first and start_estimates() refused to run while it was
        // active, permanently starving the Profiles table.
        let profile_fingerprint = self.estimates_fingerprint();
        if profile_fingerprint != self.profile_fingerprint {
            self.start_estimates(profile_fingerprint);
        }
        if self.scan_rx.is_none() {
            self.start_scan();
        }
        self.log = process::upstream_log(&self.cfg, 300);
        self.last_refresh = Instant::now();
    }
    fn start_scan(&mut self) {
        if self.scan_rx.is_some() {
            return;
        }
        let cfg = self.cfg.clone();
        let previous = self.models_fingerprint;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let fingerprint = models::models_fingerprint(&cfg);
            let models = (Some(fingerprint) != previous).then(|| models::scan(&cfg));
            let _ = tx.send((fingerprint, models));
        });
        self.scan_rx = Some(rx);
    }
    fn poll_scan(&mut self) {
        let result = self.scan_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some((fingerprint, models)) = result {
            self.scan_rx = None;
            if let Some(models) = models {
                self.models = models;
                self.models_fingerprint = Some(fingerprint);

                self.profile_fingerprint = String::new();
                self.estimate_rx = None;
                self.estimate_target = None;
            }
            let target = self.estimates_fingerprint();
            if target != self.profile_fingerprint {
                self.start_estimates(target);
            }
        }
    }
    fn estimates_fingerprint(&self) -> String {
        serde_json::to_string(&(
            &self.profiles.profiles,
            &self.profiles.expose,
            &self.cfg.gpu_layers,
            self.cfg.ctx_size,
            &self.cfg.extra_args,
            self.models_fingerprint,
        ))
        .unwrap_or_default()
    }
    fn start_estimates(&mut self, fingerprint: String) {
        if self.scan_rx.is_some() {
            return;
        }
        if self.models.is_empty() {
            self.profile_estimates.clear();
            self.profile_fingerprint = fingerprint;
            return;
        }
        if self.estimate_rx.is_some() {
            if self.estimate_target.as_deref() == Some(fingerprint.as_str()) {
                return;
            }

            self.estimate_rx = None;
            self.estimate_target = None;
        }
        let cfg = self.cfg.clone();
        let profiles = self.profiles.clone();
        let models = self.models.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            compute_profile_estimates(&cfg, &profiles, &models, &tx);
        });
        self.estimate_rx = Some(rx);
        self.estimate_target = Some(fingerprint);
    }
    fn poll_estimates(&mut self) {
        loop {
            let result = self.estimate_rx.as_ref().map(|rx| rx.try_recv());
            match result {
                Some(Ok(ProfileEstimateUpdate::Estimate(name, estimate))) => {
                    self.profile_estimates.insert(name, estimate);
                }
                Some(Ok(ProfileEstimateUpdate::Done)) => {
                    self.estimate_rx = None;
                    if let Some(target) = self.estimate_target.take() {
                        self.profile_fingerprint = target;
                    }
                    break;
                }
                Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    self.estimate_rx = None;
                    self.estimate_target = None;
                    break;
                }
                Some(Err(std::sync::mpsc::TryRecvError::Empty)) | None => break,
            }
        }
    }
    fn start_telemetry(&mut self, include_serving: bool) {
        if self.telemetry_rx.is_some() {
            return;
        }
        let cfg = self.cfg.clone();
        let paths = self.paths.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let system = system_telemetry(&cfg, include_serving);
            let serving = include_serving.then(|| serving_telemetry(&cfg, &paths));
            let _ = tx.send((system, serving));
        });
        self.telemetry_rx = Some(rx);
        self.last_telemetry = Instant::now();
    }
    fn poll_telemetry(&mut self) {
        let result = self.telemetry_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some((system, serving)) = result {
            self.telemetry_rx = None;
            if let Some(serving) = serving {
                self.apply_telemetry(system, serving);
            } else {
                self.telemetry.ram_used = system.ram_used;
                self.telemetry.ram_total = system.ram_total;
                self.telemetry.vram_used = system.vram_used;
                self.telemetry.vram_total = system.vram_total;
                self.telemetry.gpu_temps = system.gpu_temps;
            }
        }
    }
    fn start_update_check(&mut self) {
        if self.update_check_rx.is_some() {
            return;
        }
        let paths = self.paths.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(crate::update::check(&paths));
        });
        self.update_check_rx = Some(rx);
    }
    fn poll_update_check(&mut self) {
        let result = self
            .update_check_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());
        if let Some(result) = result {
            self.update_check_rx = None;
            match result {
                Ok((llama, llama_changed, swap, swap_changed)) => {
                    self.last_check =
                        Some((llama.clone(), llama_changed, swap.clone(), swap_changed));
                    self.notice = format!(
                        "llama.cpp {llama}: {} · llama-swap {swap}: {}",
                        if llama_changed {
                            "update available"
                        } else {
                            "current"
                        },
                        if swap_changed {
                            "update available"
                        } else {
                            "current"
                        }
                    );
                }
                Err(error) => self.notice = format!("✗ {error:#}"),
            }
        }
    }
    fn apply_telemetry(&mut self, mut telemetry: Telemetry, mut serving: Telemetry) {
        telemetry.prompt_done = serving.prompt_done;
        telemetry.prompt_total = serving.prompt_total;
        telemetry.generated = serving.generated;
        telemetry.active_requests = serving.active_requests;
        telemetry.total_requests = serving.total_requests;
        telemetry.total_input_tokens = serving.total_input_tokens;
        telemetry.total_output_tokens = serving.total_output_tokens;
        telemetry.total_cache_tokens = serving.total_cache_tokens;
        telemetry.last_request = serving.last_request.clone();
        self.last_tok.extend(serving.historical_tok_s.clone());
        telemetry.historical_tok_s = serving.historical_tok_s.clone();
        telemetry.model_name = serving.model_name.clone();
        telemetry.model_state = serving.model_state;
        telemetry.llama_cpp_version = serving.llama_cpp_version;
        let now = Instant::now();
        for slot in &mut serving.slot_details {
            let samples = self
                .slot_samples
                .entry((slot.model_name.clone(), slot.slot_id))
                .or_default();
            if !slot.is_processing {
                samples.clear();
                slot.pp_tok_s = None;
                slot.td_tok_s = None;
                continue;
            }
            samples.push_back((now, slot.prompt_done, slot.decoded));
            while samples
                .front()
                .is_some_and(|(at, _, _)| now.duration_since(*at) > Duration::from_secs(5))
            {
                samples.pop_front();
            }
            if let (
                Some((first_at, first_prompt, first_decoded)),
                Some((last_at, last_prompt, last_decoded)),
            ) = (samples.front(), samples.back())
            {
                let elapsed = last_at.duration_since(*first_at).as_secs_f64();
                if elapsed > 0.0 {
                    if last_prompt >= first_prompt {
                        slot.pp_tok_s = Some((last_prompt - first_prompt) as f64 / elapsed);
                    }
                    if last_decoded >= first_decoded {
                        slot.td_tok_s = Some((last_decoded - first_decoded) as f64 / elapsed);
                    }
                }
            }
            if slot.is_processing
                && let Some(td) = slot.td_tok_s
            {
                self.last_tok.insert(slot.model_name.clone(), td);
            }
        }
        self.slot_samples.retain(|(model, slot_id), samples| {
            serving
                .slot_details
                .iter()
                .any(|slot| slot.model_name == *model && slot.slot_id == *slot_id)
                || samples
                    .back()
                    .is_some_and(|(at, _, _)| now.duration_since(*at) <= Duration::from_secs(30))
        });
        let models = serving
            .model_name
            .split(", ")
            .filter(|model| !model.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut expected = 0;
        for model in &models {
            let configured = self
                .profiles
                .profiles
                .get(model)
                .and_then(|profile| profile.get("parallel"))
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize;
            expected += configured;
            serving
                .slot_details
                .retain(|slot| slot.model_name != *model || slot.slot_id < configured);
            for slot_id in 0..configured {
                if !serving
                    .slot_details
                    .iter()
                    .any(|slot| slot.model_name == *model && slot.slot_id == slot_id)
                {
                    serving.slot_details.push(SlotDetail {
                        model_name: model.clone(),
                        slot_id,
                        ..SlotDetail::default()
                    });
                }
            }
        }
        serving.slot_count = expected.max(serving.slot_details.len());
        serving.slot_details.sort_by(|left, right| {
            left.model_name
                .cmp(&right.model_name)
                .then(left.slot_id.cmp(&right.slot_id))
        });
        telemetry.slot_count = serving.slot_count;
        telemetry.slot_details = serving.slot_details.clone();

        telemetry.tokens_per_second = self.token_sample.and_then(|(old, at)| {
            let elapsed = now.duration_since(at).as_secs_f64();
            (elapsed > 0.0 && elapsed <= 5.0 && serving.generated >= old)
                .then(|| (serving.generated - old) as f64 / elapsed)
        });

        self.token_sample = match self.token_sample {
            Some((_old, at))
                if serving.active_requests > 0
                    && now.duration_since(at) <= Duration::from_secs(5) =>
            {
                Some((serving.generated, at))
            }
            _ if serving.active_requests > 0 => Some((serving.generated, now)),
            _ => None,
        };
        self.telemetry = telemetry;
    }
    fn count(&self) -> usize {
        match self.page {
            1 => self.visible_models().len(),
            2 => self.profiles.profiles.len(),
            3 => settings(&self.cfg).len(),
            5 => 4,
            _ => 1,
        }
    }
    fn visible_models(&self) -> Vec<&models::Model> {
        self.models
            .iter()
            .filter(|m| {
                self.filter.is_empty()
                    || m.id.to_lowercase().contains(&self.filter.to_lowercase())
                    || m.relative
                        .to_lowercase()
                        .contains(&self.filter.to_lowercase())
            })
            .collect()
    }
    fn selected_profile(&self) -> Option<String> {
        self.profiles.profiles.keys().nth(self.selected).cloned()
    }
    fn spawn_task<F>(&mut self, label: impl Into<String>, work: F)
    where
        F: FnOnce() -> Result<String> + Send + 'static,
    {
        if let Some(task) = &self.background {
            self.notice = format!("Busy: {} is still running", task.label);
            return;
        }
        let label = label.into();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(work());
        });
        self.notice = format!("… {label}");
        self.background = Some(BackgroundTask { label, rx });
    }
    fn poll_background(&mut self) {
        let done = self
            .background
            .as_ref()
            .and_then(|task| task.rx.try_recv().ok());
        if let Some(result) = done {
            let label = self.background.as_ref().map(|task| task.label.clone());
            let succeeded = result.is_ok();
            self.background = None;
            match result {
                Ok(message) => self.notice = message,
                Err(error) => self.notice = format!("✗ {error:#}"),
            }
            if succeeded
                && matches!(
                    label.as_deref(),
                    Some("updating llama.cpp" | "updating llama-swap" | "building llama.cpp")
                )
            {
                self.start_update_check();
            }
            self.refresh();
        }
    }
    fn regenerate_swap_async(&mut self, message: String) {
        let cfg = self.cfg.clone();
        let paths = self.paths.clone();
        let profiles = self.profiles.clone();
        self.spawn_task("regenerating swap config", move || {
            process::write_swap_config(&cfg, &paths, &profiles)?;
            Ok(message)
        });
    }
    fn start_exact_profile(&mut self) {
        let Some(profile) = self.selected_profile() else {
            self.notice = "No profile selected".into();
            return;
        };
        let cfg = self.cfg.clone();
        let paths = self.paths.clone();
        let profiles = self.profiles.clone();
        self.spawn_task("loading profile", move || {
            let swap_running = process::pid(&paths).is_some() && process::swap_mode(&paths);
            if !swap_running {
                if process::pid(&paths).is_some() {
                    process::stop(&paths)?;
                }
                process::start(&cfg, &paths, &profiles, None, &[])?;
                process::wait_swap_ready(&cfg, Duration::from_secs(10))?;
            }
            process::swap_load(&cfg, &profiles, &profile)?;
            Ok(format!("Loaded {profile} through scheduler"))
        });
    }
    fn save_runtime_state(&mut self) {
        if let Err(error) = self.cfg.save(self.paths) {
            self.notice = format!("✗ {error:#}");
            self.refresh();
            return;
        }
        let message = if self.page == 3 && self.selected == 3 {
            format!(
                "Selected runtime {} · restart the server to apply",
                self.cfg.runtime
            )
        } else {
            "Saved and reloaded scheduler configuration".into()
        };
        self.regenerate_swap_async(message);
        self.refresh();
    }
    fn toggle_setting(&mut self, direction: i32) {
        match self.selected {
            0 => {
                self.cfg.host = if self.cfg.host == "127.0.0.1" {
                    "0.0.0.0".into()
                } else {
                    "127.0.0.1".into()
                }
            }
            3 => {
                self.open_runtime_picker();
                return;
            }
            4 => {
                let position = crate::config::BACKENDS
                    .iter()
                    .position(|backend| *backend == self.cfg.backend)
                    .unwrap_or(0) as i32;
                let length = crate::config::BACKENDS.len() as i32;
                self.cfg.backend = crate::config::BACKENDS
                    [(position + direction).rem_euclid(length) as usize]
                    .into();
            }
            5 => {
                let step = context_step(&self.cfg) as i64 * direction.signum() as i64;
                self.cfg.ctx_size = (self.cfg.ctx_size as i64 + step).max(1024) as u64;
            }
            6 => self.cfg.scheduler_enabled = !self.cfg.scheduler_enabled,
            7 => {
                self.cfg.scheduler_vram_fraction =
                    (self.cfg.scheduler_vram_fraction + direction as f64 * 0.05).clamp(0.1, 1.0)
            }
            9 => self.cfg.advertise_base_models = !self.cfg.advertise_base_models,
            10 => self.cfg.advertise_profiles = !self.cfg.advertise_profiles,
            11 => {
                let enable = !crate::service_enabled();
                let paths = self.paths.clone();
                self.spawn_task(
                    if enable {
                        "enabling start on boot"
                    } else {
                        "disabling start on boot"
                    },
                    move || {
                        crate::set_start_on_boot(&paths, enable)?;
                        Ok(if enable {
                            "Start on boot enabled".into()
                        } else {
                            "Start on boot disabled".into()
                        })
                    },
                );
                return;
            }
            _ => {
                self.notice = "This setting requires a typed value; use `llamactl config`".into();
                return;
            }
        }
        self.save_runtime_state();
    }
    fn open_runtime_picker(&mut self) {
        let options = available_runtimes(self.paths, &self.cfg.runtime);
        let selected = options
            .iter()
            .position(|runtime| runtime == &self.cfg.runtime)
            .unwrap_or(0);
        self.runtime_picker = Some(RuntimePicker { options, selected });
    }
    fn handle_runtime_picker(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.runtime_picker = None,
            KeyCode::Up | KeyCode::Char('k') => {
                if let Some(picker) = &mut self.runtime_picker {
                    picker.selected = picker.selected.saturating_sub(1);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if let Some(picker) = &mut self.runtime_picker {
                    picker.selected =
                        (picker.selected + 1).min(picker.options.len().saturating_sub(1));
                }
            }
            KeyCode::Home => {
                if let Some(picker) = &mut self.runtime_picker {
                    picker.selected = 0;
                }
            }
            KeyCode::End => {
                if let Some(picker) = &mut self.runtime_picker {
                    picker.selected = picker.options.len().saturating_sub(1);
                }
            }
            KeyCode::Enter => {
                let runtime = self
                    .runtime_picker
                    .as_ref()
                    .and_then(|picker| picker.options.get(picker.selected))
                    .cloned();
                self.runtime_picker = None;
                if let Some(runtime) = runtime {
                    self.cfg.runtime = runtime;
                    self.save_runtime_state();
                }
            }
            _ => {}
        }
    }
    fn create_profile_for_selected_model(&mut self) {
        let Some((model_id, model_path)) = self
            .visible_models()
            .get(self.selected)
            .map(|model| (model.id.clone(), model.path.clone()))
        else {
            return;
        };
        let base = format!("{model_id}-custom");
        let mut name = base.clone();
        let mut suffix = 2;
        while self.profiles.profiles.contains_key(&name) {
            name = format!("{base}-{suffix}");
            suffix += 1;
        }
        let mut profile = serde_json::Map::new();
        profile.insert("_model".into(), Value::String(model_id.clone()));
        profile.insert(
            "ctx-size".into(),
            Value::from(
                models::context_limit(&model_path)
                    .unwrap_or(self.cfg.ctx_size)
                    .min(32768),
            ),
        );
        profile.insert("parallel".into(), Value::from(1));
        profile.insert("cache-type-k".into(), Value::String("q8_0".into()));
        profile.insert("cache-type-v".into(), Value::String("q8_0".into()));
        profile.insert("flash-attn".into(), Value::String("on".into()));
        profile.insert("n-gpu-layers".into(), Value::String("all".into()));
        profile.insert(
            "_extra_args".into(),
            Value::Array(vec![Value::String("--kv-unified".into())]),
        );
        self.profiles.profiles.insert(name.clone(), profile);
        self.profiles
            .models
            .insert(model_id, Value::String(name.clone()));
        match self.profiles.save(self.paths) {
            Ok(()) => {
                self.regenerate_swap_async(format!("Created and bound profile {name}"));
            }
            Err(error) => self.notice = format!("✗ {error:#}"),
        }
        self.refresh();
    }
    fn bind_selected_profile(&mut self) {
        let Some(profile) = self.selected_profile() else {
            return;
        };
        let Some(owner) = self.profiles.owner(&profile).map(str::to_owned) else {
            self.notice = "✗ Profile has no owner".into();
            return;
        };
        self.profiles
            .models
            .insert(owner.clone(), Value::String(profile.clone()));
        match self.profiles.save(self.paths) {
            Ok(()) => {
                self.regenerate_swap_async(format!("Bound {owner} → {profile}"));
            }
            Err(error) => self.notice = format!("✗ {error:#}"),
        }
        self.refresh();
    }
    fn profile_clone_selected(&mut self) {
        let Some(source) = self.selected_profile() else {
            return;
        };
        let mut suffix = 2;
        let target = loop {
            let target = format!("{source}-copy-{suffix}");
            if !self.profiles.profiles.contains_key(&target) {
                break target;
            }
            suffix += 1;
        };
        match self
            .profiles
            .clone_profile(&source, &target)
            .and_then(|_| self.profiles.save(self.paths))
        {
            Ok(()) => {
                self.regenerate_swap_async(format!("Cloned {source} → {target}"));
            }
            Err(error) => self.notice = format!("✗ {error:#}"),
        }
        self.refresh();
    }
    fn adjust_profile(&mut self, key: char) {
        let Some(name) = self.selected_profile() else {
            return;
        };
        let Some(profile) = self.profiles.profiles.get_mut(&name) else {
            return;
        };
        match key {
            '+' | '-' => {
                let current = profile
                    .get("ctx-size")
                    .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
                    .unwrap_or(4096) as i64;
                let step = context_step(&self.cfg) as i64;
                let direction = if key == '+' { step } else { -step };
                profile.insert(
                    "ctx-size".into(),
                    Value::from((current + direction).max(1024)),
                );
            }
            '[' | ']' => {
                let current = profile.get("parallel").and_then(Value::as_u64).unwrap_or(1) as i64;
                let direction = if key == ']' { 1 } else { -1 };
                profile.insert(
                    "parallel".into(),
                    Value::from((current + direction).clamp(1, 32)),
                );
            }
            't' => {
                let current = profile
                    .get("split-mode")
                    .and_then(Value::as_str)
                    .unwrap_or("layer");
                let next = match current {
                    "layer" => "tensor",
                    "tensor" => "row",
                    _ => "layer",
                };
                profile.insert("split-mode".into(), Value::String(next.into()));
            }
            'f' => {
                let current = profile
                    .get("flash-attn")
                    .and_then(Value::as_str)
                    .unwrap_or("on");
                profile.insert(
                    "flash-attn".into(),
                    Value::String(if current == "on" { "off" } else { "on" }.into()),
                );
            }
            'k' => {
                let current = profile
                    .get("cache-type-k")
                    .and_then(Value::as_str)
                    .unwrap_or("f16");
                let next = match current {
                    "f16" => "q8_0",
                    "q8_0" => "q4_0",
                    _ => "f16",
                };
                profile.insert("cache-type-k".into(), Value::String(next.into()));
                profile.insert("cache-type-v".into(), Value::String(next.into()));
            }
            _ => return,
        }
        match self.profiles.save(self.paths) {
            Ok(()) => {
                self.regenerate_swap_async(format!("Updated profile {name}"));
            }
            Err(error) => self.notice = format!("✗ {error:#}"),
        }
        self.refresh();
    }
    fn open_profile_editor(&mut self) {
        let Some(name) = self.selected_profile() else {
            return;
        };
        let Some(owner) = self.profiles.owner(&name).map(str::to_owned) else {
            self.notice = "✗ Profile has no owner".into();
            return;
        };
        let Some(model) = self.models.iter().find(|m| m.id == owner) else {
            self.notice = format!("✗ Owner model '{owner}' is not in the library");
            return;
        };
        let profile = self
            .profiles
            .profiles
            .get(&name)
            .cloned()
            .unwrap_or_default();
        let mut settings = editor_settings_from_profile(&profile);
        settings.context_step = context_step(&self.cfg);
        if profile.get("ctx-size").is_none() {
            settings.ctx = self.cfg.ctx_size;
        }
        settings.assign =
            self.profiles.models.get(&owner).and_then(Value::as_str) == Some(name.as_str());
        let editor = ProfileEditor {
            name,
            owner,
            path: model.path.clone(),
            fields: editor_fields(false),
            selected: 0,
            settings,
            prompt: None,
            estimate: EstimateState::Pending,
            estimate_rx: None,
            advanced: false,
            notice: String::new(),
        };
        self.profile_editor = Some(editor);
        self.spawn_editor_estimate();
    }
    fn editor_args(&self, s: &EditorSettings) -> Vec<String> {
        let mut args = process::common_args(&self.cfg);
        args.extend([
            "--ctx-size".into(),
            s.ctx.to_string(),
            "--parallel".into(),
            s.parallel.to_string(),
            "--batch-size".into(),
            s.batch.to_string(),
            "--ubatch-size".into(),
            s.ubatch.to_string(),
            "--cache-type-k".into(),
            s.cache_k.clone(),
            "--cache-type-v".into(),
            s.cache_v.clone(),
            "--flash-attn".into(),
            s.flash.clone(),
            "--n-gpu-layers".into(),
            s.gpu_layers.clone(),
            "--split-mode".into(),
            s.split.clone(),
            "--tensor-split".into(),
            s.tensor_split.clone(),
            "--threads".into(),
            s.threads.to_string(),
            "--threads-batch".into(),
            s.threads_batch.to_string(),
        ]);
        args.push(
            if s.kv_unified {
                "--kv-unified"
            } else {
                "--no-kv-unified"
            }
            .into(),
        );
        if s.spec_type != "none" && !s.spec_type.is_empty() {
            args.extend([
                "--spec-type".into(),
                s.spec_type.clone(),
                "--spec-draft-n-max".into(),
                s.spec_draft_nmax.to_string(),
            ]);
        }
        if !s.spec_draft_model.is_empty() {
            args.extend([
                "--spec-draft-model".into(),
                s.spec_draft_model.clone(),
                "--spec-draft-ngl".into(),
                s.spec_draft_ngl.clone(),
            ]);
            if s.spec_draft_n_cpu_moe != 0 {
                args.extend([
                    "--spec-draft-n-cpu-moe".into(),
                    s.spec_draft_n_cpu_moe.to_string(),
                ]);
            }
        }
        if s.fit == "off" {
            args.extend(["--fit".into(), "off".into()]);
        } else {
            args.extend(["--fit-target".into(), s.fit_target.to_string()]);
        }
        if !s.load_mode.is_empty() {
            args.extend(["--load-mode".into(), s.load_mode.clone()]);
        }
        if s.mlock {
            args.push("--mlock".into());
        }
        if s.direct_io {
            args.push("--direct-io".into());
        }
        if s.numa != "off" {
            args.extend(["--numa".into(), s.numa.clone()]);
        }
        if s.n_cpu_moe != 0 {
            args.extend(["--n-cpu-moe".into(), s.n_cpu_moe.to_string()]);
        }
        if !s.kv_offload {
            args.push("--no-kv-offload".into());
        }
        if !s.override_tensor.is_empty() {
            args.extend(["--override-tensor".into(), s.override_tensor.clone()]);
        }
        if !s.rope_base.is_empty() {
            args.extend(["--rope-freq-base".into(), s.rope_base.clone()]);
        }
        if !s.rope_scale.is_empty() {
            args.extend(["--rope-freq-scale".into(), s.rope_scale.clone()]);
        }
        if !s.seed.is_empty() {
            args.extend(["--seed".into(), s.seed.clone()]);
        }
        args.push(if s.jinja { "--jinja" } else { "--no-jinja" }.into());
        if !s.chat_template_file.is_empty() {
            args.extend(["--chat-template-file".into(), s.chat_template_file.clone()]);
        } else if !s.chat_template.is_empty() {
            args.extend(["--chat-template".into(), s.chat_template.clone()]);
        }
        if !s.chat_template_kwargs.is_empty() {
            args.extend([
                "--chat-template-kwargs".into(),
                s.chat_template_kwargs.clone(),
            ]);
        }
        if !s.reasoning.is_empty() {
            args.extend(["--reasoning".into(), s.reasoning.clone()]);
        }
        args.extend(s.extra.clone());
        args
    }
    fn spawn_editor_estimate(&mut self) {
        let Some((path, settings)) = self
            .profile_editor
            .as_ref()
            .map(|editor| (editor.path.clone(), editor.settings.clone()))
        else {
            return;
        };
        let args = self.editor_args(&settings);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(models::estimate(&path, &args));
        });
        if let Some(editor) = self.profile_editor.as_mut() {
            editor.estimate = EstimateState::Pending;
            editor.estimate_rx = Some(rx);
        }
    }
    fn poll_editor_estimate(&mut self) {
        let result = self
            .profile_editor
            .as_ref()
            .and_then(|editor| editor.estimate_rx.as_ref())
            .and_then(|rx| rx.try_recv().ok());
        if let Some(estimate) = result
            && let Some(editor) = self.profile_editor.as_mut()
        {
            editor.estimate = EstimateState::Ready(estimate);
        }
    }
    fn editor_choices(&self, field: EditorField, s: &EditorSettings) -> Vec<String> {
        let mut list: Vec<String> = match field {
            EditorField::Ctx => {
                let step = s.context_step;
                let limit = self
                    .profile_editor
                    .as_ref()
                    .and_then(|editor| models::context_limit(&editor.path))
                    .unwrap_or_else(|| s.ctx.max(self.cfg.ctx_size).max(1_048_576));
                let mut values = (step..=limit).step_by(step as usize).collect::<Vec<_>>();
                values.extend([0, s.ctx, limit]);
                values.sort_unstable();
                values.dedup();
                values.into_iter().map(|value| value.to_string()).collect()
            }
            EditorField::ContextStep => ["1024", "2048", "4096", "8192", "16384", "32768", "65536"]
                .map(str::to_owned)
                .to_vec(),
            EditorField::Parallel => ["1", "2", "4", "8", "16"].map(str::to_owned).to_vec(),
            EditorField::Batch => ["128", "256", "512", "1024", "2048", "4096", "8192"]
                .map(str::to_owned)
                .to_vec(),
            EditorField::Ubatch => ["64", "128", "256", "512", "1024", "2048"]
                .map(str::to_owned)
                .to_vec(),
            EditorField::CacheK | EditorField::CacheV => ["f16", "q8_0", "q4_0", "q5_0", "q5_1"]
                .map(str::to_owned)
                .to_vec(),
            EditorField::Flash => ["on", "off"].map(str::to_owned).to_vec(),
            EditorField::GpuLayers => ["all", "auto", "0", "12", "24", "38", "52"]
                .map(str::to_owned)
                .to_vec(),
            EditorField::Split => ["none", "layer", "row", "tensor"]
                .map(str::to_owned)
                .to_vec(),
            EditorField::SpecType => [
                "none",
                "draft-mtp",
                "draft-dflash",
                "draft-dspark",
                "draft-eagle3",
            ]
            .map(str::to_owned)
            .to_vec(),
            EditorField::Fit => ["on", "off"].map(str::to_owned).to_vec(),
            EditorField::Numa => ["off", "distribute", "isolate"].map(str::to_owned).to_vec(),
            _ => vec![],
        };
        let current = editor_field_value(field, s);
        if !list.is_empty() && !list.contains(&current) {
            list.push(current);
        }
        list
    }
    fn editor_set_value(&self, field: EditorField, raw: &str, s: &mut EditorSettings) {
        let num = |default: u64| raw.parse::<u64>().unwrap_or(default);
        let on = || raw == "on" || raw == "true";
        match field {
            EditorField::Ctx => s.ctx = num(s.ctx),
            EditorField::ContextStep => s.context_step = num(s.context_step).clamp(1024, 65_536),
            EditorField::Parallel => s.parallel = num(1).clamp(1, 32),
            EditorField::Batch => {
                s.batch = num(2048);
                if s.ubatch > s.batch {
                    s.ubatch = s.batch;
                }
            }
            EditorField::Ubatch => s.ubatch = num(512).min(s.batch),
            EditorField::CacheK => s.cache_k = raw.to_owned(),
            EditorField::CacheV => s.cache_v = raw.to_owned(),
            EditorField::Flash => s.flash = raw.to_owned(),
            EditorField::GpuLayers => s.gpu_layers = raw.to_owned(),
            EditorField::Split => s.split = raw.to_owned(),
            EditorField::TensorSplit => s.tensor_split = raw.to_owned(),
            EditorField::Threads => s.threads = num(32).clamp(1, 512),
            EditorField::ThreadsBatch => s.threads_batch = num(32).clamp(1, 512),
            EditorField::KvUnified => s.kv_unified = on(),
            EditorField::SpecType => s.spec_type = raw.to_owned(),
            EditorField::SpecDraftModel => s.spec_draft_model = raw.to_owned(),
            EditorField::SpecDraftNMax => s.spec_draft_nmax = num(0),
            EditorField::SpecDraftNgl => s.spec_draft_ngl = raw.to_owned(),
            EditorField::Fit => s.fit = raw.to_owned(),
            EditorField::FitTarget => s.fit_target = num(512),
            EditorField::LoadMode => s.load_mode = raw.to_owned(),
            EditorField::Mlock => s.mlock = on(),
            EditorField::DirectIO => s.direct_io = on(),
            EditorField::Numa => s.numa = raw.to_owned(),
            EditorField::KvOffload => s.kv_offload = on(),
            EditorField::CpuMoe => s.n_cpu_moe = num(0),
            EditorField::DraftCpuMoe => s.spec_draft_n_cpu_moe = num(0),
            EditorField::OverrideTensor => s.override_tensor = raw.to_owned(),
            EditorField::RopeBase => s.rope_base = raw.to_owned(),
            EditorField::RopeScale => s.rope_scale = raw.to_owned(),
            EditorField::Seed => s.seed = raw.to_owned(),
            EditorField::Reasoning => s.reasoning = raw.to_owned(),
            EditorField::Jinja => s.jinja = on(),
            EditorField::ChatTemplate => s.chat_template = raw.to_owned(),
            EditorField::ChatTemplateFile => s.chat_template_file = raw.to_owned(),
            EditorField::ChatTemplateKwargs => s.chat_template_kwargs = raw.to_owned(),
            EditorField::Extra => s.extra = raw.split_whitespace().map(str::to_owned).collect(),
            EditorField::Assign | EditorField::Advanced => {}
        }
    }
    fn editor_cycle(&mut self, dir: i32) {
        let field = self
            .profile_editor
            .as_ref()
            .and_then(|editor| editor.fields.get(editor.selected).copied());
        let Some(field) = field else {
            return;
        };
        let Some(mut settings) = self.profile_editor.as_ref().map(|e| e.settings.clone()) else {
            return;
        };
        let mut toggle_advanced = false;
        let changed = match field {
            EditorField::KvUnified => {
                settings.kv_unified = !settings.kv_unified;
                true
            }
            EditorField::Mlock => {
                settings.mlock = !settings.mlock;
                true
            }
            EditorField::DirectIO => {
                settings.direct_io = !settings.direct_io;
                true
            }
            EditorField::KvOffload => {
                settings.kv_offload = !settings.kv_offload;
                true
            }
            EditorField::Jinja => {
                settings.jinja = !settings.jinja;
                true
            }
            EditorField::Assign => {
                settings.assign = !settings.assign;
                true
            }
            EditorField::Advanced => {
                toggle_advanced = true;
                true
            }
            _ => {
                let current = editor_field_value(field, &settings);
                let choices = self.editor_choices(field, &settings);
                let Some(index) = choices.iter().position(|choice| *choice == current) else {
                    return;
                };
                let next =
                    choices[(index as i32 + dir).rem_euclid(choices.len() as i32) as usize].clone();
                self.editor_set_value(field, &next, &mut settings);
                true
            }
        };
        if changed {
            if let Some(editor) = self.profile_editor.as_mut() {
                editor.settings = settings;
                if toggle_advanced {
                    editor.advanced = !editor.advanced;
                    editor.fields = editor_fields(editor.advanced);
                    editor.selected = editor.selected.min(editor.fields.len().saturating_sub(1));
                }
            }
            self.spawn_editor_estimate();
        }
    }
    fn editor_open_typed(&mut self) {
        let Some(editor) = self.profile_editor.as_mut() else {
            return;
        };
        let field = editor
            .fields
            .get(editor.selected)
            .copied()
            .unwrap_or(EditorField::Ctx);
        let (label, current) = match field {
            EditorField::Ctx => (
                "Context tokens (0 = model maximum)",
                editor.settings.ctx.to_string(),
            ),
            EditorField::ContextStep => (
                "Context step tokens (1024–65536)",
                editor.settings.context_step.to_string(),
            ),
            EditorField::Batch => ("Prompt batch size", editor.settings.batch.to_string()),
            EditorField::Ubatch => ("Physical microbatch", editor.settings.ubatch.to_string()),
            EditorField::GpuLayers => (
                "GPU layers (all / auto / N)",
                editor.settings.gpu_layers.clone(),
            ),
            EditorField::TensorSplit => (
                "Tensor split (e.g. 1,1,1,1)",
                editor.settings.tensor_split.clone(),
            ),
            EditorField::Threads => ("CPU threads", editor.settings.threads.to_string()),
            EditorField::ThreadsBatch => (
                "CPU threads batch",
                editor.settings.threads_batch.to_string(),
            ),
            EditorField::SpecDraftModel => {
                ("Draft model path", editor.settings.spec_draft_model.clone())
            }
            EditorField::SpecDraftNMax => (
                "Draft tokens (0 = off)",
                editor.settings.spec_draft_nmax.to_string(),
            ),
            EditorField::SpecDraftNgl => (
                "Draft GPU layers (all / auto / N)",
                editor.settings.spec_draft_ngl.clone(),
            ),
            EditorField::FitTarget => ("Fit target MiB", editor.settings.fit_target.to_string()),
            EditorField::LoadMode => ("Load mode (mmap / none)", editor.settings.load_mode.clone()),
            EditorField::CpuMoe => (
                "CPU experts (n-cpu-moe)",
                editor.settings.n_cpu_moe.to_string(),
            ),
            EditorField::DraftCpuMoe => (
                "Draft CPU experts",
                editor.settings.spec_draft_n_cpu_moe.to_string(),
            ),
            EditorField::OverrideTensor => (
                "Override tensor (pattern=buffer)",
                editor.settings.override_tensor.clone(),
            ),
            EditorField::RopeBase => ("RoPE frequency base", editor.settings.rope_base.clone()),
            EditorField::RopeScale => ("RoPE frequency scale", editor.settings.rope_scale.clone()),
            EditorField::Seed => ("Seed (blank = random)", editor.settings.seed.clone()),
            EditorField::Reasoning => (
                "Reasoning (on / off / auto)",
                editor.settings.reasoning.clone(),
            ),
            EditorField::ChatTemplate => (
                "Inline Jinja chat template",
                editor.settings.chat_template.clone(),
            ),
            EditorField::ChatTemplateFile => (
                "Jinja chat template file",
                editor.settings.chat_template_file.clone(),
            ),
            EditorField::ChatTemplateKwargs => (
                "Chat template kwargs (JSON object)",
                editor.settings.chat_template_kwargs.clone(),
            ),
            _ => return,
        };
        editor.prompt = Some(EditorPrompt {
            label: label.into(),
            text: current.clone(),
            cursor: current.chars().count(),
            field,
        });
    }
    fn editor_open_extra(&mut self) {
        let Some(editor) = self.profile_editor.as_mut() else {
            return;
        };
        let text = editor.settings.extra.join(" ");
        editor.prompt = Some(EditorPrompt {
            label: "Extra llama-server flags (space-separated)".into(),
            text: text.clone(),
            cursor: text.chars().count(),
            field: EditorField::Extra,
        });
    }
    fn editor_commit_prompt(&mut self) {
        let prompt = self
            .profile_editor
            .as_mut()
            .and_then(|editor| editor.prompt.take());
        let Some(prompt) = prompt else {
            return;
        };
        let text = prompt.text.trim().to_owned();
        let mut ok = true;
        if let Some(editor) = self.profile_editor.as_mut() {
            match prompt.field {
                EditorField::Ctx => match text.parse::<u64>() {
                    Ok(value) => {
                        let limit = models::context_limit(&editor.path);
                        if value != 0 && limit.is_some_and(|limit| value > limit) {
                            editor.notice = format!(
                                "✗ Context {value} exceeds model maximum {}",
                                limit.unwrap()
                            );
                            ok = false;
                        } else {
                            editor.settings.ctx = value;
                        }
                    }
                    Err(_) => {
                        editor.notice = "✗ Context must be a number".into();
                        ok = false;
                    }
                },
                EditorField::ContextStep => match text.parse::<u64>() {
                    Ok(value) if (1024..=65_536).contains(&value) => {
                        editor.settings.context_step = value
                    }
                    _ => {
                        editor.notice = "✗ Context step must be between 1024 and 65536".into();
                        ok = false;
                    }
                },
                EditorField::Batch => match text.parse::<u64>() {
                    Ok(value) if value > 0 => {
                        editor.settings.batch = value;
                        if editor.settings.ubatch > value {
                            editor.settings.ubatch = value;
                        }
                    }
                    _ => {
                        editor.notice = "✗ Batch must be a positive number".into();
                        ok = false;
                    }
                },
                EditorField::Ubatch => match text.parse::<u64>() {
                    Ok(value) if value > 0 && value <= editor.settings.batch => {
                        editor.settings.ubatch = value;
                    }
                    Ok(_) => {
                        editor.notice = "✗ Microbatch cannot exceed prompt batch".into();
                        ok = false;
                    }
                    Err(_) => {
                        editor.notice = "✗ Microbatch must be a number".into();
                        ok = false;
                    }
                },
                EditorField::Threads => {
                    editor.settings.threads = text.parse().unwrap_or(editor.settings.threads)
                }
                EditorField::ThreadsBatch => {
                    editor.settings.threads_batch =
                        text.parse().unwrap_or(editor.settings.threads_batch)
                }
                EditorField::SpecDraftNMax => {
                    editor.settings.spec_draft_nmax =
                        text.parse().unwrap_or(editor.settings.spec_draft_nmax)
                }
                EditorField::FitTarget => {
                    editor.settings.fit_target = text.parse().unwrap_or(editor.settings.fit_target)
                }
                EditorField::CpuMoe => {
                    editor.settings.n_cpu_moe = text.parse().unwrap_or(editor.settings.n_cpu_moe)
                }
                EditorField::DraftCpuMoe => {
                    editor.settings.spec_draft_n_cpu_moe =
                        text.parse().unwrap_or(editor.settings.spec_draft_n_cpu_moe)
                }
                EditorField::GpuLayers => {
                    if text == "all" || text == "auto" || text.parse::<u64>().is_ok() {
                        editor.settings.gpu_layers = text;
                    } else {
                        editor.notice = "✗ GPU layers: all, auto, or a number".into();
                        ok = false;
                    }
                }
                EditorField::SpecDraftNgl => {
                    if text == "all" || text == "auto" || text.parse::<u64>().is_ok() {
                        editor.settings.spec_draft_ngl = text;
                    } else {
                        editor.notice = "✗ Draft GPU layers: all, auto, or a number".into();
                        ok = false;
                    }
                }
                EditorField::TensorSplit => editor.settings.tensor_split = text,
                EditorField::SpecDraftModel => editor.settings.spec_draft_model = text,
                EditorField::LoadMode => editor.settings.load_mode = text,
                EditorField::OverrideTensor => editor.settings.override_tensor = text,
                EditorField::RopeBase => editor.settings.rope_base = text,
                EditorField::RopeScale => editor.settings.rope_scale = text,
                EditorField::Seed => editor.settings.seed = text,
                EditorField::Reasoning => editor.settings.reasoning = text,
                EditorField::ChatTemplate => editor.settings.chat_template = text,
                EditorField::ChatTemplateFile => editor.settings.chat_template_file = text,
                EditorField::ChatTemplateKwargs => {
                    if text.is_empty()
                        || serde_json::from_str::<serde_json::Map<String, Value>>(&text).is_ok()
                    {
                        editor.settings.chat_template_kwargs = text;
                    } else {
                        editor.notice = "✗ Template kwargs must be a JSON object".into();
                        ok = false;
                    }
                }
                EditorField::Extra => {
                    editor.settings.extra = text.split_whitespace().map(str::to_owned).collect();
                }
                _ => {}
            }
        }
        if ok {
            self.spawn_editor_estimate();
        }
    }
    fn save_profile_editor(&mut self) {
        let Some(editor) = self.profile_editor.take() else {
            return;
        };
        let name = editor.name.clone();
        let owner = editor.owner.clone();
        let settings = editor.settings.clone();
        let assign = settings.assign;
        if !settings.spec_draft_model.is_empty() {
            let args = self.editor_args(&settings);
            if let Err(error) = process::validate_draft_model(Some(&editor.path), &args) {
                self.profile_editor = Some(editor);
                self.notice = format!("✗ {error:#}");
                return;
            }
        }
        let Some(existing) = self.profiles.profiles.get(&name).cloned() else {
            self.notice = format!("✗ Profile '{name}' no longer exists");
            return;
        };
        let updated = editor_profile_from_profile(&existing, &settings, &owner);
        self.profiles.profiles.insert(name.clone(), updated);
        if assign {
            self.profiles
                .models
                .insert(owner, Value::String(name.clone()));
        } else if self.profiles.models.get(&owner).and_then(Value::as_str) == Some(name.as_str()) {
            self.profiles.models.remove(&owner);
        }
        self.cfg.context_step_scale = settings.context_step as f64 / 4096.0;
        match self
            .profiles
            .save(self.paths)
            .and_then(|_| self.cfg.save(self.paths))
        {
            Ok(()) => {
                self.regenerate_swap_async(format!("Saved profile {name}"));
                self.refresh();
            }
            Err(error) => {
                self.notice = format!("✗ {error:#}");
                self.refresh();
            }
        }
    }
    fn editor_prompt_mut(&mut self) -> Option<&mut EditorPrompt> {
        self.profile_editor
            .as_mut()
            .and_then(|editor| editor.prompt.as_mut())
    }
    fn editor_prompt_cursor(&mut self, delta: i32) {
        if let Some(prompt) = self.editor_prompt_mut() {
            let len = prompt.text.chars().count() as i32;
            prompt.cursor = (prompt.cursor as i32 + delta).clamp(0, len) as usize;
        }
    }
    fn editor_prompt_set(&mut self, pos: usize) {
        if let Some(prompt) = self.editor_prompt_mut() {
            prompt.cursor = pos.min(prompt.text.chars().count());
        }
    }
    fn editor_prompt_end(&mut self) {
        if let Some(prompt) = self.editor_prompt_mut() {
            prompt.cursor = prompt.text.chars().count();
        }
    }
    fn editor_prompt_insert(&mut self, c: char) {
        let Some(prompt) = self.editor_prompt_mut() else {
            return;
        };
        let byte = prompt
            .text
            .char_indices()
            .nth(prompt.cursor)
            .map(|(i, _)| i)
            .unwrap_or(prompt.text.len());
        prompt.text.insert(byte, c);
        prompt.cursor += 1;
    }
    fn editor_prompt_backspace(&mut self) {
        let Some(prompt) = self.editor_prompt_mut() else {
            return;
        };
        if prompt.cursor == 0 {
            return;
        }
        let byte = prompt
            .text
            .char_indices()
            .nth(prompt.cursor - 1)
            .map(|(i, _)| i)
            .unwrap_or(0);
        prompt.text.remove(byte);
        prompt.cursor -= 1;
    }
    fn editor_prompt_delete(&mut self) {
        let Some(prompt) = self.editor_prompt_mut() else {
            return;
        };
        if prompt.cursor >= prompt.text.chars().count() {
            return;
        }
        let byte = prompt
            .text
            .char_indices()
            .nth(prompt.cursor)
            .map(|(i, _)| i)
            .unwrap_or(prompt.text.len());
        prompt.text.remove(byte);
    }
    fn editor_prompt_clear_to_start(&mut self) {
        let Some(prompt) = self.editor_prompt_mut() else {
            return;
        };
        let byte = prompt
            .text
            .char_indices()
            .nth(prompt.cursor)
            .map(|(i, _)| i)
            .unwrap_or(prompt.text.len());
        prompt.text.drain(..byte);
        prompt.cursor = 0;
    }
    fn editor_prompt_clear_to_end(&mut self) {
        let Some(prompt) = self.editor_prompt_mut() else {
            return;
        };
        let byte = prompt
            .text
            .char_indices()
            .nth(prompt.cursor)
            .map(|(i, _)| i)
            .unwrap_or(prompt.text.len());
        prompt.text.truncate(byte);
    }
    fn editor_prompt_delete_word_before(&mut self) {
        let Some(prompt) = self.editor_prompt_mut() else {
            return;
        };
        let chars = prompt.text.chars().collect::<Vec<_>>();
        let mut start = prompt.cursor;
        while start > 0 && !chars[start - 1].is_alphanumeric() {
            start -= 1;
        }
        while start > 0 && chars[start - 1].is_alphanumeric() {
            start -= 1;
        }
        if start != prompt.cursor {
            let start_byte = chars[..start].iter().map(|ch| ch.len_utf8()).sum::<usize>();
            let end_byte = chars[..prompt.cursor]
                .iter()
                .map(|ch| ch.len_utf8())
                .sum::<usize>();
            prompt.text.drain(start_byte..end_byte);
            prompt.cursor = start;
        }
    }
    fn editor_prompt_word(&mut self, dir: i32) {
        let Some(prompt) = self.editor_prompt_mut() else {
            return;
        };
        let chars: Vec<char> = prompt.text.chars().collect();
        let len = chars.len() as i32;
        if dir < 0 {
            let mut pos = prompt.cursor as i32 - 1;
            while pos > 0 && !chars[pos as usize].is_alphanumeric() {
                pos -= 1;
            }
            while pos > 0 && chars[(pos - 1) as usize].is_alphanumeric() {
                pos -= 1;
            }
            prompt.cursor = pos.max(0) as usize;
        } else {
            let mut pos = prompt.cursor as i32;
            while pos < len && !chars[pos as usize].is_alphanumeric() {
                pos += 1;
            }
            while pos < len && chars[pos as usize].is_alphanumeric() {
                pos += 1;
            }
            prompt.cursor = pos.min(len) as usize;
        }
    }
    fn handle_editor_prompt(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => {
                if let Some(editor) = self.profile_editor.as_mut() {
                    editor.prompt = None;
                }
            }
            KeyCode::Enter => self.editor_commit_prompt(),
            KeyCode::Left if ctrl => self.editor_prompt_word(-1),
            KeyCode::Right if ctrl => self.editor_prompt_word(1),
            KeyCode::Left => self.editor_prompt_cursor(-1),
            KeyCode::Right => self.editor_prompt_cursor(1),
            KeyCode::Home => self.editor_prompt_set(0),
            KeyCode::End => self.editor_prompt_end(),
            KeyCode::Backspace => self.editor_prompt_backspace(),
            KeyCode::Delete => self.editor_prompt_delete(),
            KeyCode::Char('h') if ctrl => self.editor_prompt_backspace(),
            KeyCode::Char('a') if ctrl => self.editor_prompt_set(0),
            KeyCode::Char('e') if ctrl => self.editor_prompt_end(),
            KeyCode::Char('u') if ctrl => self.editor_prompt_clear_to_start(),
            KeyCode::Char('k') if ctrl => self.editor_prompt_clear_to_end(),
            KeyCode::Char('w') if ctrl => self.editor_prompt_delete_word_before(),
            KeyCode::Char(c) if !c.is_control() => self.editor_prompt_insert(c),
            _ => {}
        }
    }
    fn handle_profile_editor(&mut self, key: KeyEvent) {
        if self
            .profile_editor
            .as_ref()
            .is_some_and(|editor| editor.prompt.is_some())
        {
            self.handle_editor_prompt(key);
            return;
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.profile_editor = None;
            }
            KeyCode::Enter | KeyCode::Char('s') => self.save_profile_editor(),
            KeyCode::Up | KeyCode::Char('k') => {
                if let Some(editor) = self.profile_editor.as_mut() {
                    editor.selected = editor.selected.saturating_sub(1);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if let Some(editor) = self.profile_editor.as_mut() {
                    editor.selected =
                        (editor.selected + 1).min(editor.fields.len().saturating_sub(1));
                }
            }
            KeyCode::Left | KeyCode::Char('h') => self.editor_cycle(-1),
            KeyCode::Right | KeyCode::Char('l') => self.editor_cycle(1),
            KeyCode::Char('t') => self.editor_open_typed(),
            KeyCode::Char('e') => {
                let is_extra = self.profile_editor.as_ref().is_some_and(|editor| {
                    editor.fields.get(editor.selected).copied() == Some(EditorField::Extra)
                });
                if is_extra {
                    self.editor_open_extra();
                }
            }
            _ => {}
        }
    }
    fn request_profile_delete(&mut self) {
        let Some(profile) = self.selected_profile() else {
            return;
        };
        if self.cfg.scheduler_pinned_models.contains(&profile) {
            self.notice = "✗ Unpin this profile before deleting it".into();
            return;
        }
        self.profile_delete_confirm = Some(profile);
    }
    fn handle_profile_delete_confirm(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter | KeyCode::Char('y') => {
                if let Some(profile) = self.profile_delete_confirm.take() {
                    self.delete_profile(&profile);
                }
            }
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('n') => {
                self.profile_delete_confirm = None;
                self.notice = "Profile deletion cancelled".into();
            }
            _ => {}
        }
    }
    fn delete_profile(&mut self, profile: &str) {
        match self
            .profiles
            .remove(profile)
            .and_then(|_| self.profiles.save(self.paths))
            .and_then(|_| {
                let mut benchmarks = benchmark::ProfileBenchmarks::load(self.paths)?;
                benchmarks.remove(profile);
                benchmarks.save(self.paths)
            }) {
            Ok(()) => self.notice = format!("Deleted profile {profile}"),
            Err(error) => self.notice = format!("✗ {error:#}"),
        }
        self.selected = self.selected.saturating_sub(1);
        self.refresh();
    }
    fn start_profile_rename(&mut self) {
        if let Some(profile) = self.selected_profile() {
            let cursor = profile.chars().count();
            self.rename_input = Some(RenameState {
                original: profile.clone(),
                text: profile,
                cursor,
            });
        }
    }
    fn handle_rename_input(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => {
                self.rename_input = None;
                self.notice = "Rename cancelled".into();
            }
            KeyCode::Enter => {
                let Some(state) = self.rename_input.take() else {
                    return;
                };
                self.apply_rename(&state.original, &state.text);
            }
            KeyCode::Left if ctrl => self.move_cursor_word(-1),
            KeyCode::Right if ctrl => self.move_cursor_word(1),
            KeyCode::Left => self.move_cursor(-1),
            KeyCode::Right => self.move_cursor(1),
            KeyCode::Home => self.set_cursor(0),
            KeyCode::End => self.set_cursor_to_end(),
            KeyCode::Backspace => self.delete_before_cursor(),
            KeyCode::Char('h') if ctrl => self.delete_before_cursor(),
            KeyCode::Delete => self.delete_at_cursor(),
            KeyCode::Char('a') if ctrl => self.set_cursor(0),
            KeyCode::Char('e') if ctrl => self.set_cursor_to_end(),
            KeyCode::Char('u') if ctrl => self.delete_to_start(),
            KeyCode::Char('k') if ctrl => self.delete_to_end(),
            KeyCode::Char('w') if ctrl => self.delete_word_before(),
            KeyCode::Char(c) => self.insert_char(c),
            _ => {}
        }
    }
    fn move_cursor(&mut self, delta: i32) {
        if let Some(state) = self.rename_input.as_mut() {
            let len = state.text.chars().count() as i32;
            state.cursor = (state.cursor as i32 + delta).clamp(0, len) as usize;
        }
    }
    fn set_cursor(&mut self, pos: usize) {
        if let Some(state) = self.rename_input.as_mut() {
            state.cursor = pos.min(state.text.chars().count());
        }
    }
    fn set_cursor_to_end(&mut self) {
        if let Some(state) = self.rename_input.as_mut() {
            state.cursor = state.text.chars().count();
        }
    }
    fn insert_char(&mut self, c: char) {
        let Some(state) = self.rename_input.as_mut() else {
            return;
        };
        let byte = state
            .text
            .char_indices()
            .nth(state.cursor)
            .map(|(i, _)| i)
            .unwrap_or(state.text.len());
        state.text.insert(byte, c);
        state.cursor += 1;
    }
    fn delete_before_cursor(&mut self) {
        let Some(state) = self.rename_input.as_mut() else {
            return;
        };
        if state.cursor == 0 {
            return;
        }
        let byte = state
            .text
            .char_indices()
            .nth(state.cursor - 1)
            .map(|(i, _)| i)
            .unwrap_or(0);
        state.text.remove(byte);
        state.cursor -= 1;
    }
    fn delete_at_cursor(&mut self) {
        let Some(state) = self.rename_input.as_mut() else {
            return;
        };
        if state.cursor >= state.text.chars().count() {
            return;
        }
        let byte = state
            .text
            .char_indices()
            .nth(state.cursor)
            .map(|(i, _)| i)
            .unwrap_or(state.text.len());
        state.text.remove(byte);
    }
    fn delete_to_start(&mut self) {
        if let Some(state) = self.rename_input.as_mut() {
            let byte = state
                .text
                .char_indices()
                .nth(state.cursor)
                .map(|(i, _)| i)
                .unwrap_or(state.text.len());
            state.text.drain(..byte);
            state.cursor = 0;
        }
    }
    fn delete_to_end(&mut self) {
        if let Some(state) = self.rename_input.as_mut() {
            let byte = state
                .text
                .char_indices()
                .nth(state.cursor)
                .map(|(i, _)| i)
                .unwrap_or(state.text.len());
            state.text.truncate(byte);
        }
    }
    fn delete_word_before(&mut self) {
        let Some(state) = self.rename_input.as_mut() else {
            return;
        };
        let chars: Vec<char> = state.text.chars().collect();
        let mut pos = state.cursor;
        while pos > 0 && !chars[pos - 1].is_alphanumeric() {
            pos -= 1;
        }
        while pos > 0 && chars[pos - 1].is_alphanumeric() {
            pos -= 1;
        }
        if pos != state.cursor {
            let start = chars[..pos].iter().map(|c| c.len_utf8()).sum::<usize>();
            let end = chars[..state.cursor]
                .iter()
                .map(|c| c.len_utf8())
                .sum::<usize>();
            state.text.drain(start..end);
            state.cursor = pos;
        }
    }
    fn move_cursor_word(&mut self, dir: i32) {
        let Some(state) = self.rename_input.as_mut() else {
            return;
        };
        let chars: Vec<char> = state.text.chars().collect();
        let len = chars.len() as i32;
        if dir < 0 {
            let mut pos = state.cursor as i32 - 1;
            while pos > 0 && !chars[pos as usize].is_alphanumeric() {
                pos -= 1;
            }
            while pos > 0 && chars[(pos - 1) as usize].is_alphanumeric() {
                pos -= 1;
            }
            state.cursor = pos.max(0) as usize;
        } else {
            let mut pos = state.cursor as i32;
            while pos < len && !chars[pos as usize].is_alphanumeric() {
                pos += 1;
            }
            while pos < len && chars[pos as usize].is_alphanumeric() {
                pos += 1;
            }
            state.cursor = pos.min(len) as usize;
        }
    }
    fn apply_rename(&mut self, old: &str, new: &str) {
        if old == new {
            self.notice = "Name unchanged".into();
            return;
        }
        if let Err(error) = profiles::valid_name(new) {
            self.notice = format!("✗ {error:#}");
            return;
        }
        if self.profiles.profiles.contains_key(new) {
            self.notice = format!("✗ Profile '{new}' already exists");
            return;
        }
        let Some(profile) = self.profiles.profiles.remove(old) else {
            self.notice = format!("✗ Profile '{old}' not found");
            return;
        };
        self.profiles.profiles.insert(new.to_owned(), profile);
        for value in self.profiles.models.values_mut() {
            if value.as_str() == Some(old) {
                *value = Value::String(new.to_owned());
            }
        }
        if let Some(slot) = self
            .cfg
            .scheduler_pinned_models
            .iter()
            .position(|name| name == old)
        {
            self.cfg.scheduler_pinned_models[slot] = new.to_owned();
        }
        match self
            .profiles
            .save(self.paths)
            .and_then(|_| self.cfg.save(self.paths))
            .and_then(|_| {
                let mut benchmarks = benchmark::ProfileBenchmarks::load(self.paths)?;
                benchmarks.rename(old, new);
                benchmarks.save(self.paths)
            }) {
            Ok(()) => {
                self.regenerate_swap_async(format!("Renamed {old} → {new}"));
                self.refresh();
            }
            Err(error) => self.notice = format!("✗ {error:#}"),
        }
    }
    fn show_profile_benchmarks(&mut self) {
        let Some(profile) = self.selected_profile() else {
            return;
        };
        match benchmark::ProfileBenchmarks::load(self.paths) {
            Ok(benchmarks) => {
                let Some(runs) = benchmarks.profiles.get(&profile) else {
                    self.notice = format!("No benchmark results for {profile}");
                    return;
                };
                self.benchmark_view = Some(BenchmarkView { runs: runs.clone() });
            }
            Err(error) => self.notice = format!("✗ {error:#}"),
        }
    }
    fn handle_benchmark_view(&mut self, key: KeyEvent) {
        if matches!(key.code, KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter) {
            self.benchmark_view = None;
        }
    }
    fn request_profile_benchmark(&mut self) {
        if let Some(task) = &self.background {
            self.notice = format!("Busy: {} is still running", task.label);
            return;
        }
        let Some(profile) = self.selected_profile() else {
            return;
        };
        self.benchmark_dialog = Some(BenchmarkDialog::Confirm { profile });
    }
    fn start_profile_benchmark(&mut self, profile: String) {
        let cfg = self.cfg.clone();
        let paths = self.paths.clone();
        let profiles = self.profiles.clone();
        let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_cancelled = cancelled.clone();
        let worker_profile = profile.clone();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = benchmark::run_cancellable_with_progress(
                &cfg,
                &paths,
                &profiles,
                &worker_profile,
                worker_cancelled,
                progress_tx,
            )
            .map(|run| benchmark::summary(&run).replace('\n', " · "));
            let _ = result_tx.send(result);
        });
        self.benchmark_dialog = Some(BenchmarkDialog::Running {
            profile,
            cancelled,
            result_rx,
            progress_rx,
            phase: "Preparing profile…".into(),
            runtime: String::new(),
            effective_context: None,
            load_ms: None,
            completed: Vec::new(),
        });
    }
    fn handle_benchmark_dialog(&mut self, key: KeyEvent) {
        match &self.benchmark_dialog {
            Some(BenchmarkDialog::Confirm { profile }) => match key.code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    let profile = profile.clone();
                    self.start_profile_benchmark(profile);
                }
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('n') | KeyCode::Char('c') => {
                    self.benchmark_dialog = None;
                    self.notice = "Benchmark cancelled".into();
                }
                _ => {}
            },
            Some(BenchmarkDialog::Running { cancelled, .. }) => {
                if matches!(
                    key.code,
                    KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('c')
                ) {
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                    self.notice = "Cancelling benchmark and restoring server…".into();
                }
            }
            None => {}
        }
    }
    fn poll_profile_benchmark(&mut self) {
        if let Some(BenchmarkDialog::Running {
            progress_rx,
            phase,
            runtime,
            effective_context,
            load_ms,
            completed,
            ..
        }) = &mut self.benchmark_dialog
        {
            while let Ok(update) = progress_rx.try_recv() {
                match update {
                    benchmark::BenchmarkProgress::Preparing => *phase = "Preparing profile…".into(),
                    benchmark::BenchmarkProgress::StoppingServer => {
                        *phase = "Stopping active server…".into()
                    }
                    benchmark::BenchmarkProgress::LoadingRuntime => {
                        *phase = "Loading model and runtime…".into()
                    }
                    benchmark::BenchmarkProgress::Ready {
                        runtime: ready_runtime,
                        effective_context: context,
                        load_ms: ready_load_ms,
                    } => {
                        *runtime = ready_runtime;
                        *effective_context = Some(context);
                        *load_ms = Some(ready_load_ms);
                        *phase = "Runtime ready".into();
                    }
                    benchmark::BenchmarkProgress::CaseStarted {
                        name,
                        target_prompt_tokens,
                    } => {
                        *phase =
                            format!("Running {name} case · {target_prompt_tokens} prompt tokens…")
                    }
                    benchmark::BenchmarkProgress::CaseCompleted(case) => {
                        *phase = format!("Completed {} case", case.name);
                        completed.push(case);
                    }
                    benchmark::BenchmarkProgress::RestoringServer => {
                        *phase = "Restoring previous server…".into()
                    }
                }
            }
        }
        let result = match &self.benchmark_dialog {
            Some(BenchmarkDialog::Running { result_rx, .. }) => result_rx.try_recv().ok(),
            _ => None,
        };
        if let Some(result) = result {
            self.benchmark_dialog = None;
            self.notice = match result {
                Ok(summary) => summary,
                Err(error) => format!("✗ {error:#}"),
            };
            self.refresh();
        }
    }
    fn toggle_profile_pin(&mut self) {
        let Some(profile) = self.selected_profile() else {
            return;
        };
        if self.cfg.scheduler_pinned_models.contains(&profile) {
            self.cfg
                .scheduler_pinned_models
                .retain(|name| name != &profile);
            self.notice = format!("Unpinned {profile}");
        } else {
            self.cfg.scheduler_pinned_models.push(profile.clone());
            self.notice = format!("Pinned {profile}");
        }
        self.save_runtime_state();
    }
    fn action(&mut self) {
        match self.page {
            0 => {
                if process::pid(self.paths).is_some() {
                    let paths = self.paths.clone();
                    self.spawn_task("stopping server", move || {
                        process::stop(&paths)?;
                        Ok("Server stopped".into())
                    });
                } else {
                    let cfg = self.cfg.clone();
                    let paths = self.paths.clone();
                    let profiles = self.profiles.clone();
                    self.spawn_task("starting server", move || {
                        let pid = process::start(&cfg, &paths, &profiles, None, &[])?;
                        Ok(format!("Server started · pid {pid}"))
                    });
                }
            }
            1 => {
                if let Some(id) = self
                    .visible_models()
                    .get(self.selected)
                    .map(|m| m.id.clone())
                {
                    if process::pid(self.paths).is_some() && process::swap_mode(self.paths) {
                        let cfg = self.cfg.clone();
                        let profiles = self.profiles.clone();
                        self.spawn_task(format!("loading {id}"), move || {
                            process::swap_load(&cfg, &profiles, &id)?;
                            Ok(format!("Loaded {id} through scheduler"))
                        });
                    } else {
                        let cfg = self.cfg.clone();
                        let paths = self.paths.clone();
                        let profiles = self.profiles.clone();
                        self.spawn_task(format!("starting {id}"), move || {
                            if process::pid(&paths).is_some() {
                                process::stop(&paths)?;
                            }
                            let pid = process::start(&cfg, &paths, &profiles, Some(&id), &[])?;
                            Ok(format!("Started {id} · pid {pid}"))
                        });
                    }
                }
            }
            2 => {
                self.start_exact_profile();
                return;
            }
            5 => match self.selected {
                0 => {
                    self.notice = "… checking updates".into();
                    self.start_update_check();
                }
                1 => {
                    if self
                        .last_check
                        .as_ref()
                        .is_some_and(|(_, changed, _, _)| *changed)
                    {
                        let cfg = self.cfg.clone();
                        let paths = self.paths.clone();
                        self.spawn_task("updating llama.cpp", move || {
                            crate::update::install_llama(&cfg, &paths)?;
                            Ok("llama.cpp updated".into())
                        });
                    } else {
                        self.notice = "llama.cpp is already current".into();
                    }
                }
                2 => {
                    if self
                        .last_check
                        .as_ref()
                        .is_some_and(|(_, _, _, changed)| *changed)
                    {
                        let paths = self.paths.clone();
                        self.spawn_task("updating llama-swap", move || {
                            crate::update::install_swap(&paths)?;
                            Ok("llama-swap updated".into())
                        });
                    } else {
                        self.notice = "llama-swap is already current".into();
                    }
                }
                3 => {
                    let paths = self.paths.clone();
                    self.spawn_task("installing service", move || {
                        crate::install_service(&paths)?;
                        Ok("Service installed".into())
                    });
                }
                _ => self.notice = "Use the arrows and Enter".into(),
            },
            3 => self.toggle_setting(1),
            _ => self.notice = "Use the page-specific shortcut shown in its title".into(),
        }
        self.refresh();
    }
}

#[allow(clippy::collapsible_if)]
pub fn run(cfg: Config, paths: &Paths) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new(cfg, paths)?;
    app.refresh();
    app.start_telemetry(true);
    app.start_update_check();
    let result = (|| -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(60);
        let loading_quit = |k: &KeyEvent| {
            k.kind == KeyEventKind::Press
                && (k.code == KeyCode::Char('q')
                    || (k.code == KeyCode::Char('c')
                        && k.modifiers.contains(KeyModifiers::CONTROL)))
        };
        while app.scan_rx.is_some() && Instant::now() < deadline {
            app.poll_scan();
            app.poll_estimates();
            app.poll_telemetry();
            terminal.draw(|f| draw_loading(f, &app))?;
            if event::poll(Duration::from_millis(0))?
                && let Event::Key(k) = event::read()?
                && loading_quit(&k)
            {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(80));
        }
        app.poll_scan();
        app.refresh();
        while app.estimate_rx.is_some() && Instant::now() < deadline {
            app.poll_estimates();
            terminal.draw(|f| draw_loading(f, &app))?;
            if event::poll(Duration::from_millis(0))?
                && let Event::Key(k) = event::read()?
                && loading_quit(&k)
            {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(80));
        }
        app.poll_estimates();
        loop {
            app.poll_profile_benchmark();
            let benchmark_running =
                matches!(&app.benchmark_dialog, Some(BenchmarkDialog::Running { .. }));
            app.poll_telemetry();
            if !benchmark_running {
                app.poll_editor_estimate();
                app.poll_background();
                app.poll_scan();
                app.poll_estimates();
                app.poll_update_check();
            }
            if app.background.is_some() || benchmark_running {
                app.throbber_state.calc_next();
            }
            terminal.draw(|f| draw(f, &mut app))?;
            if event::poll(Duration::from_millis(250))? {
                if let Event::Key(k) = event::read()? {
                    if k.kind != KeyEventKind::Press {
                        continue;
                    }
                    if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
                        if app.benchmark_dialog.is_some() {
                            app.handle_benchmark_dialog(k);
                        } else {
                            break;
                        }
                        continue;
                    }
                    match k.code {
                        _ if app.benchmark_dialog.is_some() => app.handle_benchmark_dialog(k),
                        _ if app.key_help => {
                            if matches!(
                                k.code,
                                KeyCode::Esc
                                    | KeyCode::Enter
                                    | KeyCode::Char('q')
                                    | KeyCode::Char('?')
                            ) {
                                app.key_help = false;
                            }
                        }
                        _ if app.profile_delete_confirm.is_some() => {
                            app.handle_profile_delete_confirm(k)
                        }
                        _ if app.benchmark_view.is_some() => app.handle_benchmark_view(k),
                        _ if app.runtime_picker.is_some() => app.handle_runtime_picker(k),
                        _ if app.profile_editor.is_some() => app.handle_profile_editor(k),
                        _ if app.rename_input.is_some() => app.handle_rename_input(k),
                        KeyCode::Char('q') => break,
                        KeyCode::Char('?') => app.key_help = true,
                        KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => {
                            app.page = (app.page + 1) % PAGES.len();
                            app.selected = 0
                        }
                        KeyCode::Left | KeyCode::Char('h') | KeyCode::BackTab => {
                            app.page = (app.page + PAGES.len() - 1) % PAGES.len();
                            app.selected = 0
                        }
                        KeyCode::Char(c) if ('1'..='6').contains(&c) => {
                            app.page = c as usize - '1' as usize;
                            app.selected = 0
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            app.selected = app.selected.saturating_sub(1)
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            app.selected = (app.selected + 1).min(app.count().saturating_sub(1))
                        }
                        KeyCode::Home => app.selected = 0,
                        KeyCode::End => app.selected = app.count().saturating_sub(1),
                        KeyCode::Enter => app.action(),
                        KeyCode::Char('R') if app.page == 2 => app.start_profile_rename(),
                        KeyCode::Char('e') if app.page == 2 => app.open_profile_editor(),
                        KeyCode::Char('r') => {
                            app.refresh();
                            app.notice = "Refreshed".into()
                        }
                        KeyCode::Char('c') if app.page == 1 => {
                            app.create_profile_for_selected_model()
                        }
                        KeyCode::Char('u') if app.page == 1 => {
                            if let Some(model) = app
                                .visible_models()
                                .get(app.selected)
                                .map(|model| model.id.clone())
                            {
                                let cfg = app.cfg.clone();
                                let profiles = app.profiles.clone();
                                app.spawn_task(format!("unloading {model}"), move || {
                                    process::swap_unload(&cfg, &profiles, Some(&model))?;
                                    Ok(format!("Unloaded {model}"))
                                });
                            }
                        }
                        KeyCode::Char('m') if app.page == 2 => app.request_profile_benchmark(),
                        KeyCode::Char('v') if app.page == 2 => app.show_profile_benchmarks(),
                        KeyCode::Char('c') if app.page == 2 => app.profile_clone_selected(),
                        KeyCode::Char('d') if app.page == 2 => app.request_profile_delete(),
                        KeyCode::Char('p') if app.page == 2 => app.toggle_profile_pin(),
                        KeyCode::Char('b') if app.page == 2 => app.bind_selected_profile(),
                        KeyCode::Char(c) if app.page == 2 && "+-[]tfk".contains(c) => {
                            app.adjust_profile(c)
                        }
                        KeyCode::Char('=') if app.page == 2 => app.adjust_profile('+'),
                        KeyCode::Char('u') if app.page == 2 => {
                            if let Some(profile) = app.selected_profile() {
                                let cfg = app.cfg.clone();
                                let profiles = app.profiles.clone();
                                app.spawn_task(format!("unloading {profile}"), move || {
                                    process::swap_unload(&cfg, &profiles, Some(&profile))?;
                                    Ok(format!("Unloaded {profile}"))
                                });
                            }
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') if app.page == 3 => {
                            app.toggle_setting(1)
                        }
                        KeyCode::Char('-') if app.page == 3 => app.toggle_setting(-1),
                        _ => {}
                    }
                }
            }
            if app.last_telemetry.elapsed() > Duration::from_secs(2) {
                app.start_telemetry(!benchmark_running);
            }
            if !benchmark_running && app.last_refresh.elapsed() > Duration::from_secs(2) {
                app.refresh();
            }
        }
        Ok(())
    })();
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}
fn draw_loading(frame: &mut ratatui::Frame, app: &App) {
    let area = frame.area();
    let elapsed = app.marquee_started.elapsed().as_secs_f64();
    let spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
    let ch = spinner
        .chars()
        .nth((elapsed * 4.0) as usize % spinner.chars().count())
        .unwrap_or('·');
    let body = format!("\n {ch} Loading model library…\n\n elapsed {elapsed:.0}s");
    let center = if area.width > 32 && area.height > 9 {
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Length(6),
                Constraint::Min(0),
            ])
            .split(area);
        let horizontal = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(30),
                Constraint::Min(40),
                Constraint::Percentage(30),
            ])
            .split(vertical[1]);
        horizontal[1]
    } else {
        area
    };
    frame.render_widget(
        Paragraph::new(body)
            .style(Style::default().fg(Color::Cyan))
            .block(title("llamactl NEO")),
        center,
    );
}

fn page_keys(app: &App) -> Vec<(&'static str, &'static str)> {
    if let Some(dialog) = &app.benchmark_dialog {
        return match dialog {
            BenchmarkDialog::Confirm { .. } => {
                vec![("Enter/y", "run"), ("Esc/q/n", "cancel")]
            }
            BenchmarkDialog::Running { .. } => vec![("Esc/q/c", "cancel benchmark")],
        };
    }
    if app.key_help {
        return vec![("Enter/Esc/q/?", "close controls")];
    }
    if app.profile_delete_confirm.is_some() {
        return vec![("Enter/y", "delete"), ("Esc/q/n", "cancel")];
    }
    if app.benchmark_view.is_some() {
        return vec![("Enter/Esc", "close")];
    }
    if app.runtime_picker.is_some() {
        return vec![
            ("↑↓/jk", "select"),
            ("Home/End", "first/last"),
            ("Enter", "confirm"),
            ("Esc/q", "cancel"),
        ];
    }
    if let Some(editor) = &app.profile_editor {
        if editor.prompt.is_some() {
            return vec![
                ("Enter", "confirm"),
                ("Esc", "cancel"),
                ("←→", "move"),
                ("Backspace", "delete"),
            ];
        }
        return vec![
            ("↑↓/jk", "field"),
            ("←→/hl", "value"),
            ("t", "exact"),
            ("e", "flags"),
            ("Enter/s", "save"),
            ("Esc/q", "cancel"),
        ];
    }
    if app.rename_input.is_some() {
        return vec![("Enter", "confirm"), ("Esc", "cancel"), ("←→", "cursor")];
    }
    match app.page {
        0 => vec![
            ("Enter", "start/stop"),
            ("r", "refresh"),
            ("?", "controls"),
            ("q", "quit"),
        ],
        1 => vec![
            ("Enter", "load model"),
            ("c", "create profile"),
            ("u", "unload"),
            ("r", "refresh"),
            ("?", "controls"),
            ("q", "quit"),
        ],
        2 => vec![
            ("Enter", "load"),
            ("m/v", "benchmark/results"),
            ("e/R/c/d", "edit/rename/clone/delete"),
            ("b/p/u", "bind/pin/unload"),
            ("+/-/[]", "context/slots"),
            ("t/f/k", "split/flash/cache"),
            ("r", "refresh"),
            ("?", "controls"),
            ("q", "quit"),
        ],
        3 => vec![
            ("Enter/+/-", "change"),
            ("r", "refresh"),
            ("?", "controls"),
            ("q", "quit"),
        ],
        4 => vec![("r", "refresh"), ("?", "controls"), ("q", "quit")],
        5 => vec![
            ("Enter", "run action"),
            ("r", "refresh"),
            ("?", "controls"),
            ("q", "quit"),
        ],
        _ => vec![("Enter", "act"), ("?", "controls"), ("q", "quit")],
    }
}
fn legend_line(app: &App) -> Line<'static> {
    let mut spans = Vec::new();
    for (i, (key, desc)) in page_keys(app).iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ", Style::default().fg(Color::DarkGray)));
        }
        spans.push(Span::styled(
            *key,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(" {desc}"),
            Style::default().fg(Color::DarkGray),
        ));
    }
    Line::from(spans)
}

fn notice_color(notice: &str) -> Color {
    if notice.starts_with('✗') {
        Color::Red
    } else {
        Color::DarkGray
    }
}

fn draw(frame: &mut ratatui::Frame, app: &mut App) {
    let area = frame.area();
    if area.width < 72 || area.height < 18 {
        frame.render_widget(
            Paragraph::new(
                "llamactl NEO\n\nResize terminal to at least 72×18.\nq quit · r refresh",
            )
            .style(Style::default().fg(Color::Yellow)),
            area,
        );
        return;
    }
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
    let (status_line, status_color) = if let Some(editor) = &app.profile_editor {
        if let Some(prompt) = &editor.prompt {
            let mut text = prompt.text.clone();
            let byte = text
                .char_indices()
                .nth(prompt.cursor)
                .map(|(i, _)| i)
                .unwrap_or(text.len());
            text.insert(byte, '▏');
            (
                format!(" {}: {text} · Enter confirm · Esc cancel", prompt.label),
                Color::DarkGray,
            )
        } else if editor.notice.is_empty() {
            (format!(" {}", app.notice), notice_color(&app.notice))
        } else {
            (format!(" {}", editor.notice), notice_color(&editor.notice))
        }
    } else if let Some(state) = &app.rename_input {
        let mut text = state.text.clone();
        let byte = text
            .char_indices()
            .nth(state.cursor)
            .map(|(i, _)| i)
            .unwrap_or(text.len());
        text.insert(byte, '▏');
        (
            format!(" RENAME PROFILE → {text} · Enter confirm · Esc cancel"),
            Color::DarkGray,
        )
    } else if let Some(task) = &app.background {
        (format!("{} …", task.label), Color::DarkGray)
    } else {
        (format!(" {}", app.notice), notice_color(&app.notice))
    };
    if app.background.is_some() {
        let spinner_area = Rect {
            x: outer[2].x + 1,
            width: outer[2].width.saturating_sub(1),
            ..outer[2]
        };
        frame.render_stateful_widget(
            Throbber::default()
                .label(status_line)
                .style(Style::default().fg(status_color))
                .throbber_style(Style::default().fg(Color::Cyan)),
            spinner_area,
            &mut app.throbber_state,
        );
    } else {
        frame.render_widget(
            Paragraph::new(status_line).style(Style::default().fg(status_color)),
            outer[2],
        );
    }
    frame.render_widget(Paragraph::new(legend_line(app)), outer[3]);
    draw_telemetry_strip(frame, app, outer[0]);
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(23), Constraint::Min(45)])
        .split(outer[1]);
    let items = PAGES
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let mut line = Line::from(format!("{}. {n}", i + 1));
            if i == 5
                && app
                    .last_check
                    .as_ref()
                    .is_some_and(|(_, lc, _, sc)| *lc || *sc)
            {
                line.push_span(Span::styled(" ↑", Style::default().fg(Color::Yellow)));
            }
            ListItem::new(line)
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected(Some(app.page));
    frame.render_stateful_widget(
        List::new(items)
            .block(
                Block::default()
                    .title(Span::styled(
                        " WORKSPACE ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::RIGHT)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("› "),
        body[0],
        &mut state,
    );
    match app.page {
        0 => dashboard(frame, app, body[1]),
        1 => model_page(frame, app, body[1]),
        2 => {
            if let Some(editor) = &app.profile_editor {
                profile_editor_view(frame, app, editor, body[1]);
            } else {
                profile_page(frame, app, body[1]);
            }
        }
        3 => settings_page(frame, app, body[1]),
        4 => logs(frame, app, body[1]),
        _ => system(frame, app, body[1]),
    }
    if let Some(picker) = &app.runtime_picker {
        runtime_picker_modal(frame, picker, area);
    }
    if let Some(view) = &app.benchmark_view {
        benchmark_modal(frame, view, area);
    }
    if let Some(profile) = &app.profile_delete_confirm {
        profile_delete_modal(frame, profile, area);
    }
    if let Some(state) = &app.rename_input {
        rename_modal(frame, state, area);
    }
    if let Some(dialog) = &app.benchmark_dialog {
        benchmark_dialog_modal(frame, dialog, area, &mut app.throbber_state);
    }
    if app.key_help {
        keyboard_help_modal(frame, area);
    }
}
fn keyboard_help_modal(frame: &mut ratatui::Frame, area: Rect) {
    let width = area.width.saturating_sub(8).min(88);
    let height = 17.min(area.height.saturating_sub(2));
    let modal = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    let rows = [
        ("↑/↓ or j/k", "select previous/next item"),
        ("Home / End", "select first/last item"),
        ("←/→ or h/l", "switch workspace"),
        ("Tab / Shift+Tab", "switch workspace"),
        ("1–6", "jump directly to a workspace"),
        ("Enter", "run the primary action"),
        ("r", "refresh the current state"),
        ("?", "open or close this control reference"),
        ("q / Ctrl+C", "quit; modal q cancels instead"),
        ("Profiles: m / v", "run benchmark / view results"),
        ("Profiles: e / R / c / d", "edit / rename / clone / delete"),
        ("Profiles: b / p / u", "bind / pin / unload"),
        ("Profiles: +/- / [/]", "context size / parallel slots"),
        (
            "Profiles: t / f / k",
            "split mode / flash attention / KV cache",
        ),
    ]
    .map(|(key, action)| Row::new([key, action]));
    frame.render_widget(Clear, modal);
    frame.render_widget(
        Table::new(rows, [Constraint::Length(27), Constraint::Min(1)])
            .header(
                Row::new(["KEY", "ACTION"]).style(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            )
            .block(title("KEYBOARD CONTROLS").padding(Padding::horizontal(1))),
        modal,
    );
}
fn profile_delete_modal(frame: &mut ratatui::Frame, profile: &str, area: Rect) {
    let width = area.width.saturating_sub(12).min(80);
    let height = 7.min(area.height.saturating_sub(2));
    let modal = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, modal);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                format!("Permanently delete profile {profile}?"),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("This also deletes its retained benchmark results."),
            Line::from(""),
            Line::from(Span::styled(
                "Enter/y delete · Esc/q/n cancel",
                Style::default().fg(Color::DarkGray),
            )),
        ])
        .block(title("DELETE PROFILE").padding(Padding::horizontal(1)))
        .wrap(Wrap { trim: false }),
        modal,
    );
}
fn rename_modal(frame: &mut ratatui::Frame, state: &RenameState, area: Rect) {
    let width = area.width.saturating_sub(12).min(80);
    let height = 5;
    let modal = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    let mut text = state.text.clone();
    let byte = text
        .char_indices()
        .nth(state.cursor)
        .map(|(index, _)| index)
        .unwrap_or(text.len());
    text.insert(byte, '▏');
    frame.render_widget(Clear, modal);
    frame.render_widget(
        Paragraph::new(format!("\n{text}"))
            .block(title("RENAME PROFILE").padding(Padding::horizontal(1))),
        modal,
    );
}
fn benchmark_dialog_modal(
    frame: &mut ratatui::Frame,
    dialog: &BenchmarkDialog,
    area: Rect,
    throbber_state: &mut ThrobberState,
) {
    let width = area.width.saturating_sub(8).min(100);
    let wanted_height = match dialog {
        BenchmarkDialog::Confirm { .. } => 8,
        BenchmarkDialog::Running { .. } => 12,
    };
    let height = wanted_height.min(area.height.saturating_sub(2));
    let modal = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, modal);
    let block = title("PROFILE BENCHMARK").padding(Padding::horizontal(1));
    let inner = block.inner(modal);
    frame.render_widget(block, modal);
    match dialog {
        BenchmarkDialog::Confirm { profile } => frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    format!("Run the full benchmark for {profile}?"),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from("The active server will be stopped and restored afterward."),
                Line::from("Loading or interacting with models would invalidate the results."),
                Line::from(""),
                Line::from(Span::styled(
                    "Enter/y run · Esc/q/n cancel",
                    Style::default().fg(Color::DarkGray),
                )),
            ])
            .wrap(Wrap { trim: false }),
            inner,
        ),
        BenchmarkDialog::Running {
            profile,
            cancelled,
            phase,
            runtime,
            effective_context,
            load_ms,
            completed,
            ..
        } => {
            let cancelling = cancelled.load(std::sync::atomic::Ordering::Relaxed);
            frame.render_stateful_widget(
                Throbber::default()
                    .label(if cancelling {
                        format!("Cancelling {profile} and restoring server…")
                    } else {
                        phase.clone()
                    })
                    .style(Style::default().fg(Color::White))
                    .throbber_style(Style::default().fg(Color::Cyan)),
                Rect { height: 1, ..inner },
                throbber_state,
            );
            let metadata = if let Some(context) = effective_context {
                benchmark_metadata_line(runtime, *context, load_ms.unwrap_or_default())
            } else {
                Line::from(Span::styled(
                    "Resolving profile and runtime…",
                    Style::default().fg(Color::DarkGray),
                ))
            };
            let mut metadata_spans = vec![
                Span::styled("Profile ", Style::default().fg(Color::DarkGray)),
                Span::raw(profile.clone()),
                Span::raw("   "),
            ];
            metadata_spans.extend(metadata.spans);
            frame.render_widget(
                Paragraph::new(Line::from(metadata_spans)),
                Rect {
                    y: inner.y + 2,
                    height: 1,
                    ..inner
                },
            );
            frame.render_widget(
                benchmark_case_table(completed),
                Rect {
                    y: inner.y + 4,
                    height: inner.height.saturating_sub(5).min(4),
                    ..inner
                },
            );
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    "Controls locked · Esc/q/c cancels",
                    Style::default().fg(Color::DarkGray),
                ))),
                Rect {
                    y: inner.bottom().saturating_sub(1),
                    height: 1,
                    ..inner
                },
            );
        }
    }
}
fn benchmark_modal(frame: &mut ratatui::Frame, view: &BenchmarkView, area: Rect) {
    let width = area.width.saturating_sub(8).min(110);
    let height = (view.runs.len() as u16 * 5 + 2)
        .max(7)
        .min(area.height.saturating_sub(2));
    let modal = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, modal);
    let block = title("PROFILE BENCHMARK").padding(Padding::horizontal(1));
    let inner = block.inner(modal);
    frame.render_widget(block, modal);
    for (index, run) in view.runs.iter().rev().enumerate() {
        let y = inner.y + index as u16 * 5;
        let partial = if run.cases.len() < 3 {
            " · PARTIAL"
        } else {
            ""
        };
        let mut heading = vec![Span::styled(
            format!(
                "RUN {}{partial} · {}   ",
                view.runs.len() - index,
                run.profile
            ),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
        heading.extend(
            benchmark_metadata_line(&run.runtime, run.effective_context, run.load_ms).spans,
        );
        frame.render_widget(
            Paragraph::new(Line::from(heading)),
            Rect {
                y,
                height: 1,
                ..inner
            },
        );
        frame.render_widget(
            benchmark_case_table(&run.cases),
            Rect {
                y: y + 1,
                height: 4,
                ..inner
            },
        );
    }
}
fn benchmark_metadata_line(runtime: &str, context: u64, load_ms: u64) -> Line<'static> {
    Line::from(vec![
        Span::styled("Runtime ", Style::default().fg(Color::DarkGray)),
        Span::raw(runtime.to_owned()),
        Span::styled("   Context ", Style::default().fg(Color::DarkGray)),
        Span::raw(context.to_string()),
        Span::styled("   Load ", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("{:.2}s", load_ms as f64 / 1000.0)),
    ])
}
fn benchmark_case_table(cases: &[benchmark::BenchmarkCase]) -> Table<'static> {
    let rows =
        [("small", "SMALL"), ("medium", "MEDIUM"), ("long", "LARGE")].map(|(name, label)| {
            let case = cases.iter().find(|case| case.name == name);
            let placeholder =
                || Line::from(Span::styled("--", Style::default().fg(Color::DarkGray)));
            let metric = |value: Option<f64>, color| {
                value
                    .map(|value| {
                        Line::from(Span::styled(
                            format!("{value:.1}"),
                            Style::default().fg(color),
                        ))
                    })
                    .unwrap_or_else(&placeholder)
            };
            Row::new(vec![
                Line::from(Span::styled(
                    label,
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
                case.map(|case| Line::raw(case.actual_prompt_tokens.to_string()))
                    .unwrap_or_else(&placeholder),
                metric(
                    case.map(|case| case.prompt_tokens_per_second),
                    Color::Yellow,
                ),
                metric(case.map(|case| case.decode_tokens_per_second), Color::Green),
                metric(
                    case.map(|case| case.decode_peak_tokens_per_second),
                    Color::Green,
                ),
                metric(
                    case.map(|case| case.decode_median_tokens_per_second),
                    Color::Green,
                ),
                case.map(|case| {
                    Line::from(Span::styled(
                        format!("{:.2}s", case.time_to_first_response_ms / 1000.0),
                        Style::default().fg(Color::Yellow),
                    ))
                })
                .unwrap_or_else(&placeholder),
            ])
        });
    Table::new(
        rows,
        [
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
        ],
    )
    .header(
        Row::new([
            "CASE", "TOKENS", "PP T/S", "DEC T/S", "PEAK", "MEDIAN", "FIRST",
        ])
        .style(
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
    )
}
fn runtime_picker_modal(frame: &mut ratatui::Frame, picker: &RuntimePicker, area: Rect) {
    let content_width = picker
        .options
        .iter()
        .map(|runtime| runtime.chars().count())
        .max()
        .unwrap_or(0) as u16;
    let width = (content_width + 6).clamp(48, area.width.saturating_sub(4));
    let height = (picker.options.len() as u16 + 2).clamp(5, area.height.saturating_sub(4));
    let modal = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, modal);
    let items = picker
        .options
        .iter()
        .map(|runtime| ListItem::new(runtime.as_str()))
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected(Some(picker.selected));
    frame.render_stateful_widget(
        List::new(items)
            .block(title("SELECT RUNTIME").padding(Padding::horizontal(1)))
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("› "),
        modal,
        &mut state,
    );
}
fn draw_telemetry_strip(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let cards = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 5),
            Constraint::Ratio(1, 5),
            Constraint::Ratio(1, 5),
            Constraint::Ratio(2, 5),
        ])
        .split(area);
    let t = &app.telemetry;
    residence_card(
        frame,
        cards[0],
        "VRAM",
        t.vram_used,
        t.vram_total,
        Color::Green,
    );
    residence_card(frame, cards[1], "RAM", t.ram_used, t.ram_total, Color::Cyan);
    let temps = if t.gpu_temps.is_empty() {
        "—".into()
    } else {
        t.gpu_temps
            .iter()
            .enumerate()
            .map(|(i, temp)| format!("GPU {i}: {temp:.0}°C"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let temps = marquee_text(
        &temps,
        cards[2].width.saturating_sub(2) as usize,
        app.marquee_started.elapsed(),
    );
    frame.render_widget(
        Paragraph::new(format!(" {temps}"))
            .style(
                Style::default()
                    .fg(temperature_color(&t.gpu_temps))
                    .add_modifier(Modifier::BOLD),
            )
            .block(compact_block("GPU TEMPS")),
        cards[2],
    );
    let prompt = if t.prompt_total > 0 {
        format!(
            "{:>3.0}%",
            t.prompt_done as f64 * 100.0 / t.prompt_total as f64
        )
    } else if t.active_requests > 0 {
        "  …".into()
    } else {
        " —".into()
    };
    let rate = t
        .tokens_per_second
        .map(|v| format!("{v:.1}"))
        .unwrap_or("—".into());
    let online = process::pid(app.paths).is_some();
    let (card_title, title_color) = if !online {
        ("○ STOPPED".to_owned(), Color::Red)
    } else if t.model_state == ModelState::None {
        ("✓ SERVING".to_owned(), Color::Green)
    } else {
        let (icon, status, color) = match t.model_state {
            ModelState::Loading => ("◐ ", "· LOADING", Color::Yellow),
            _ => ("● ", "", Color::Green),
        };

        let title_width = cards[3].width.saturating_sub(2) as usize;
        let suffix_width = if status.is_empty() {
            0
        } else {
            status.chars().count() + 1
        };
        let name_width = title_width.saturating_sub(icon.chars().count() + suffix_width);
        let names = if name_width == 0 {
            t.model_name.clone()
        } else {
            marquee_text(&t.model_name, name_width, app.marquee_started.elapsed())
        };
        let suffix = if status.is_empty() {
            String::new()
        } else {
            format!(" {status}")
        };
        (format!("{icon}{names}{suffix}"), color)
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" P ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                prompt,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  TOK ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                t.generated.to_string(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  T/S ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                rate,
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {} active", t.active_requests),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .block(
            compact_block(card_title).title_style(
                Style::default()
                    .fg(title_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ),
        cards[3],
    );
}

fn compact_block(title: impl Into<String>) -> Block<'static> {
    Block::default()
        .title(format!(" {} ", title.into()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray))
}

fn tree_slot_block(slot_id: usize, has_next: bool) -> Block<'static> {
    let tree_border = border::Set {
        top_left: "├",
        bottom_left: if has_next { "├" } else { "╰" },
        ..border::ROUNDED
    };
    Block::default()
        .title(format!("─ SLOT {slot_id} "))
        .borders(Borders::ALL)
        .border_set(tree_border)
        .border_style(Style::default().fg(Color::DarkGray))
}

fn estimate_card(
    frame: &mut ratatui::Frame,
    area: Rect,
    title: &'static str,
    estimate: Option<u64>,
    available: u64,
    color: Color,
) {
    let ratio = estimate
        .filter(|_| available > 0)
        .map(|estimate| estimate as f64 / available as f64)
        .unwrap_or_default();
    let label = match estimate {
        Some(estimate) if available > 0 => format!(
            "{:.0}/{:.0}MiB - {:.1}%",
            estimate as f64 / (1u64 << 20) as f64,
            available as f64 / (1u64 << 20) as f64,
            ratio * 100.0,
        ),
        Some(estimate) => format!("{:.0}MiB - --", estimate as f64 / (1u64 << 20) as f64),
        None => "Estimating…".into(),
    };
    frame.render_widget(
        Gauge::default()
            .block(compact_block(title))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(label)
            .gauge_style(Style::default().fg(color)),
        area,
    );
}

fn residence_card(
    frame: &mut ratatui::Frame,
    area: Rect,
    title: &'static str,
    used: u64,
    total: u64,
    color: Color,
) {
    let ratio = if total > 0 {
        used as f64 / total as f64
    } else {
        0.0
    };
    let label = if total > 0 {
        format!(
            "{:.1}/{:.1}G - {:.0}%",
            used as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64,
            ratio * 100.0
        )
    } else {
        "—".into()
    };
    frame.render_widget(
        Gauge::default()
            .block(compact_block(title))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(label)
            .gauge_style(Style::default().fg(color)),
        area,
    );
}

fn marquee_text(text: &str, width: usize, elapsed: Duration) -> String {
    let width = width.saturating_sub(1);
    let chars = text.chars().collect::<Vec<_>>();
    if width == 0 || chars.len() <= width {
        return text.into();
    }

    let gap = 4;
    let cycle = chars.len() + gap;
    let offset = ((elapsed.as_millis() / 250) as usize) % cycle;
    (0..width)
        .map(|index| {
            let position = (offset + index) % cycle;
            chars.get(position).copied().unwrap_or(' ')
        })
        .collect()
}

fn temperature_color(temperatures: &[f64]) -> Color {
    let hottest = temperatures.iter().copied().fold(0.0, f64::max);
    if hottest >= 85.0 {
        Color::Red
    } else if hottest >= 75.0 {
        Color::Yellow
    } else {
        Color::Green
    }
}

fn title(name: &str) -> Block<'static> {
    Block::default()
        .title(Line::from(Span::styled(
            format!(" {} ", name.to_uppercase()),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan))
}
fn dashboard(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),
            Constraint::Length(3),
            Constraint::Min(1),
        ])
        .split(area);

    let pid = process::pid(app.paths);
    let (status_icon, status_text, status_color) = if pid.is_some() {
        ("●", "ONLINE", Color::Green)
    } else {
        ("○", "OFFLINE", Color::Red)
    };
    let version = if app.telemetry.llama_cpp_version.is_empty() {
        "unknown".into()
    } else {
        app.telemetry.llama_cpp_version.clone()
    };
    let server_lines: Vec<Line> = vec![
        Line::from(vec![Span::styled(
            format!("{status_icon} {status_text} · pid {}", pid.unwrap_or(0)),
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Endpoint  ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("http://{}:{}/v1", app.cfg.host, app.cfg.port)),
        ]),
        Line::from(vec![
            Span::styled("Auth      ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{} API key(s)", app.cfg.keys().len())),
        ]),
        Line::from(vec![
            Span::styled("Runtime   ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("llama.cpp {}", version)),
        ]),
        Line::from(vec![
            Span::styled("Models    ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{} discovered", app.models.len())),
        ]),
        Line::from(vec![
            Span::styled("Profiles  ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{} loaded", app.profiles.profiles.len())),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(server_lines).block(title("SERVER").padding(Padding::horizontal(1))),
        chunks[0],
    );

    let stats = &app.telemetry;
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" REQUESTS ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                stats.total_requests.to_string(),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("   INPUT ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                stats.total_input_tokens.to_string(),
                Style::default().fg(Color::White),
            ),
            Span::styled("   OUTPUT ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                stats.total_output_tokens.to_string(),
                Style::default().fg(Color::Green),
            ),
            Span::styled("   CACHE ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                stats.total_cache_tokens.to_string(),
                Style::default().fg(Color::Yellow),
            ),
        ]))
        .block(compact_block("PERFORMANCE TOTALS")),
        chunks[1],
    );

    let is_loading = app.telemetry.model_state == ModelState::Loading;

    let model_line = if !app.telemetry.model_name.is_empty() {
        Line::from(vec![Span::styled(
            format!(
                "{} ({:?})",
                app.telemetry.model_name, app.telemetry.model_state
            ),
            Style::default()
                .fg(if is_loading {
                    Color::Yellow
                } else {
                    Color::Cyan
                })
                .add_modifier(Modifier::BOLD),
        )])
    } else {
        Line::from(Span::styled(
            "— (no model)",
            Style::default().fg(Color::DarkGray),
        ))
    };

    let telemetry_area = if let Some(request) = &app.telemetry.last_request {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(1)])
            .split(chunks[2]);
        let acceptance = if request.draft_tokens > 0 {
            format!(
                "{:.0}% ({}/{})",
                request.draft_accepted as f64 * 100.0 / request.draft_tokens as f64,
                request.draft_accepted,
                request.draft_tokens
            )
        } else {
            "—".into()
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    format!(" {} ", request.model),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(" PP ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1} tok/s", request.prompt_tok_s),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled("   GEN ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1} tok/s", request.generation_tok_s),
                    Style::default().fg(Color::Green),
                ),
                Span::styled("   TOKENS ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!(
                    "{}→{}",
                    request.prompt_tokens, request.output_tokens
                )),
                Span::styled("   CACHE ", Style::default().fg(Color::DarkGray)),
                Span::raw(request.cache_tokens.to_string()),
                Span::styled("   DRAFT ", Style::default().fg(Color::DarkGray)),
                Span::raw(acceptance),
                Span::styled("   TTFT ", Style::default().fg(Color::DarkGray)),
                Span::raw(
                    request
                        .ttft_ms
                        .map(|value| format!("{:.2}s", value / 1000.0))
                        .unwrap_or_else(|| "--".into()),
                ),
                Span::styled("   TIME ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!("{:.2}s", request.duration_ms as f64 / 1000.0)),
            ]))
            .block(compact_block("LAST REQUEST")),
            split[0],
        );
        split[1]
    } else {
        chunks[2]
    };
    let telemetry_block = title("METRICS").padding(Padding::horizontal(1));
    let telemetry_inner = telemetry_block.inner(telemetry_area);
    frame.render_widget(telemetry_block, telemetry_area);

    let fallback_model = if app.telemetry.model_name.is_empty() {
        "MODEL".to_owned()
    } else {
        app.telemetry.model_name.clone()
    };
    let mut groups: Vec<(String, Vec<&SlotDetail>)> = Vec::new();
    for slot in &app.telemetry.slot_details {
        let name = if slot.model_name.is_empty() {
            fallback_model.clone()
        } else {
            slot.model_name.clone()
        };
        if let Some((_, slots)) = groups.iter_mut().find(|(model, _)| *model == name) {
            slots.push(slot);
        } else {
            groups.push((name, vec![slot]));
        }
    }
    if groups.is_empty() {
        frame.render_widget(Paragraph::new(model_line), telemetry_inner);
    } else {
        let heights = groups
            .iter()
            .map(|(_, slots)| Constraint::Length(1 + slots.len() as u16 * 3))
            .collect::<Vec<_>>();
        let group_rects = Layout::default()
            .direction(Direction::Vertical)
            .constraints(heights)
            .split(telemetry_inner);
        for ((model, slots), group_rect) in groups.iter().zip(group_rects.iter()) {
            let mut header_spans = vec![Span::styled(
                model.to_owned(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )];
            if app.telemetry.model_state == ModelState::Loading {
                header_spans.push(Span::styled(
                    " · LOADING",
                    Style::default().fg(Color::DarkGray),
                ));
            }
            frame.render_widget(
                Paragraph::new(Line::from(header_spans)),
                Rect {
                    height: 1,
                    ..*group_rect
                },
            );
            let slot_area = Rect {
                y: group_rect.y + 1,
                height: group_rect.height.saturating_sub(1),
                ..*group_rect
            };
            let rects = Layout::default()
                .direction(Direction::Vertical)
                .constraints(std::iter::repeat_n(Constraint::Length(3), slots.len()))
                .split(slot_area);
            let mut lines = Vec::with_capacity(slots.len());
            for slot in slots {
                let state = if slot.is_processing { "active" } else { "idle" };
                let color = if slot.is_processing {
                    Color::Green
                } else {
                    Color::DarkGray
                };

                let prompt = if slot.is_processing {
                    format!("{:>3.0}%", slot.prompt_progress * 100.0)
                } else {
                    " —".into()
                };
                let generated = format!("{:>7}", slot.decoded);

                let rate = match (slot.pp_tok_s, slot.td_tok_s) {
                    (Some(pp), Some(td)) => Some(pp + td),
                    (Some(pp), None) => Some(pp),
                    (None, Some(td)) => Some(td),
                    (None, None) => None,
                };

                let rate = rate
                    .filter(|value| *value > 0.0)
                    .map_or_else(|| "—".into(), |value| format!("{value:.1}"));

                let line = Line::from(vec![
                    Span::styled(" P ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        prompt,
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("  TOK ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        generated,
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("  T/S ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        rate,
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  {state} "),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                ]);
                lines.push((slot.slot_id, line));
            }

            let content_width = lines
                .iter()
                .map(|(_, line)| line.width())
                .max()
                .unwrap_or(0) as u16;
            let card_width = content_width.saturating_add(2);
            let slot_count = lines.len();
            for (index, ((slot_id, line), rect)) in lines.into_iter().zip(rects.iter()).enumerate()
            {
                frame.render_widget(
                    Paragraph::new(line).block(tree_slot_block(slot_id, index + 1 < slot_count)),
                    Rect {
                        x: rect.x + 1,
                        width: card_width.min(rect.width.saturating_sub(1)),
                        ..*rect
                    },
                );
            }
        }
    }
}
fn model_page(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let visible = app.visible_models();
    let mut mains = Vec::new();
    let mut drafts = Vec::new();
    for model in &visible {
        if model.kind == models::ModelKind::Main {
            mains.push(model);
        } else {
            drafts.push(model);
        }
    }
    let show_separator = !mains.is_empty() && !drafts.is_empty();
    let mut rows = Vec::with_capacity(visible.len() + usize::from(show_separator));
    for model in &mains {
        rows.push(Row::new(vec![
            Span::raw(format!(" {}", model.id)),
            Span::raw(format!("{:.1}G", model.bytes as f64 / (1u64 << 30) as f64)),
            Span::raw(if model.vision { "vision" } else { "text" }),
            Span::raw(model.relative.clone()),
        ]));
    }
    if show_separator {
        rows.push(
            Row::new(vec![
                Span::styled("DRAFT MODELS ", Style::default().fg(Color::DarkGray)),
                Span::raw(""),
                Span::raw(""),
                Span::raw(""),
            ])
            .style(
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            ),
        );
    }
    for model in &drafts {
        rows.push(Row::new(vec![
            Span::raw(format!(" {}", model.id)),
            Span::raw(format!("{:.1}G", model.bytes as f64 / (1u64 << 30) as f64)),
            Span::styled("draft", Style::default().fg(Color::Yellow)),
            Span::raw(model.relative.clone()),
        ]));
    }
    let selected_row = if show_separator && app.selected >= mains.len() {
        app.selected + 1
    } else {
        app.selected
    };
    let table = Table::new(
        rows,
        [
            Constraint::Percentage(30),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Percentage(50),
        ],
    )
    .header(
        Row::new(["MODEL", "SIZE", "FEATURES", "PATH"]).style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .row_highlight_style(
        Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )
    .highlight_symbol("› ")
    .block(title("MODELS"));
    let mut state = ratatui::widgets::TableState::default().with_selected(Some(selected_row));
    frame.render_stateful_widget(table, area, &mut state);
}
fn profile_editor_view(frame: &mut ratatui::Frame, app: &App, editor: &ProfileEditor, area: Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Min(1),
        ])
        .split(area);
    let estimate_cards = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
        .split(layout[1]);
    let free_vram = app
        .telemetry
        .vram_total
        .saturating_sub(app.telemetry.vram_used);
    let free_ram = app
        .telemetry
        .ram_total
        .saturating_sub(app.telemetry.ram_used);
    let estimate = match &editor.estimate {
        EstimateState::Pending => None,
        EstimateState::Ready(estimate) => Some(estimate),
    };
    let vram_color = estimate.map_or(Color::DarkGray, |estimate| {
        if free_vram > 0 && estimate.vram as f64 <= free_vram as f64 * 0.85 {
            Color::Green
        } else if free_vram > 0 && estimate.vram <= free_vram {
            Color::Yellow
        } else if free_vram > 0 {
            Color::Red
        } else {
            Color::DarkGray
        }
    });
    estimate_card(
        frame,
        estimate_cards[0],
        "EST. VRAM",
        estimate.map(|estimate| estimate.vram),
        free_vram,
        vram_color,
    );
    estimate_card(
        frame,
        estimate_cards[1],
        "EST. RAM",
        estimate.map(|estimate| estimate.ram),
        free_ram,
        if estimate.is_some() {
            Color::Cyan
        } else {
            Color::DarkGray
        },
    );
    let filename = editor
        .path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| editor.path.display().to_string());
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" Profile ", Style::default().fg(Color::DarkGray)),
            Span::raw(editor.name.clone()),
            Span::styled(" · Model ", Style::default().fg(Color::DarkGray)),
            Span::raw(editor.owner.clone()),
            Span::styled(" · File ", Style::default().fg(Color::DarkGray)),
            Span::raw(filename),
        ])),
        layout[0],
    );
    let mut items = Vec::new();
    for row in editor_rows(editor.advanced) {
        match row {
            EditorRow::Header(label) => items.push(ListItem::new(Span::styled(
                label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ))),
            EditorRow::Field(field) => {
                let value = if field == EditorField::Advanced {
                    if editor.advanced {
                        "on".to_owned()
                    } else {
                        "off".to_owned()
                    }
                } else {
                    editor_field_value(field, &editor.settings)
                };
                items.push(ListItem::new(format!(
                    "{:<26} {}",
                    editor_field_label(field),
                    value
                )));
            }
        }
    }
    let selected_item = editor_selected_row(editor.selected, editor.advanced);
    let mut state = ListState::default().with_selected(Some(selected_item));
    frame.render_stateful_widget(
        List::new(items)
            .block(title("PROFILE EDITOR"))
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("› "),
        layout[2],
        &mut state,
    );
}

struct EditorCategory {
    label: &'static str,
    fields: &'static [EditorField],
}

const EDITOR_BASIC_CATEGORIES: &[EditorCategory] = &[
    EditorCategory {
        label: "PERFORMANCE",
        fields: &[
            EditorField::Ctx,
            EditorField::ContextStep,
            EditorField::Parallel,
            EditorField::Batch,
            EditorField::Ubatch,
            EditorField::Threads,
            EditorField::ThreadsBatch,
            EditorField::Flash,
        ],
    },
    EditorCategory {
        label: "KV CACHE",
        fields: &[
            EditorField::CacheK,
            EditorField::CacheV,
            EditorField::KvUnified,
            EditorField::KvOffload,
        ],
    },
    EditorCategory {
        label: "GPU OFFLOAD",
        fields: &[
            EditorField::GpuLayers,
            EditorField::Split,
            EditorField::TensorSplit,
            EditorField::Numa,
            EditorField::CpuMoe,
        ],
    },
    EditorCategory {
        label: "MEMORY",
        fields: &[
            EditorField::Fit,
            EditorField::FitTarget,
            EditorField::LoadMode,
            EditorField::Mlock,
            EditorField::DirectIO,
        ],
    },
    EditorCategory {
        label: "SPECULATIVE",
        fields: &[
            EditorField::SpecType,
            EditorField::SpecDraftModel,
            EditorField::SpecDraftNMax,
            EditorField::SpecDraftNgl,
            EditorField::DraftCpuMoe,
        ],
    },
];
const EDITOR_ADVANCED_CATEGORY: EditorCategory = EditorCategory {
    label: "ADVANCED",
    fields: &[
        EditorField::OverrideTensor,
        EditorField::RopeBase,
        EditorField::RopeScale,
        EditorField::Seed,
        EditorField::Reasoning,
        EditorField::Jinja,
        EditorField::ChatTemplate,
        EditorField::ChatTemplateFile,
        EditorField::ChatTemplateKwargs,
    ],
};
const EDITOR_PROFILE_CATEGORY: EditorCategory = EditorCategory {
    label: "PROFILE",
    fields: &[EditorField::Extra, EditorField::Assign],
};

enum EditorRow {
    Header(&'static str),
    Field(EditorField),
}
fn editor_rows(advanced: bool) -> Vec<EditorRow> {
    let mut rows = Vec::new();
    for category in EDITOR_BASIC_CATEGORIES {
        rows.push(EditorRow::Header(category.label));
        rows.extend(category.fields.iter().copied().map(EditorRow::Field));
    }
    rows.push(EditorRow::Field(EditorField::Advanced));
    if advanced {
        rows.push(EditorRow::Header(EDITOR_ADVANCED_CATEGORY.label));
        rows.extend(
            EDITOR_ADVANCED_CATEGORY
                .fields
                .iter()
                .copied()
                .map(EditorRow::Field),
        );
    }
    rows.push(EditorRow::Header(EDITOR_PROFILE_CATEGORY.label));
    rows.extend(
        EDITOR_PROFILE_CATEGORY
            .fields
            .iter()
            .copied()
            .map(EditorRow::Field),
    );
    rows
}

fn editor_fields(advanced: bool) -> Vec<EditorField> {
    let mut fields = Vec::new();
    for category in EDITOR_BASIC_CATEGORIES {
        fields.extend_from_slice(category.fields);
    }
    fields.push(EditorField::Advanced);
    if advanced {
        fields.extend_from_slice(EDITOR_ADVANCED_CATEGORY.fields);
    }
    fields.extend_from_slice(EDITOR_PROFILE_CATEGORY.fields);
    fields
}

fn editor_selected_row(selected: usize, advanced: bool) -> usize {
    let mut field = 0;
    for (row_index, row) in editor_rows(advanced).iter().enumerate() {
        if let EditorRow::Field(_) = row {
            if field == selected {
                return row_index;
            }
            field += 1;
        }
    }
    0
}
fn editor_field_label(field: EditorField) -> &'static str {
    match field {
        EditorField::Ctx => "Context size",
        EditorField::ContextStep => "Context step",
        EditorField::Parallel => "Parallel slots",
        EditorField::Batch => "Prompt batch",
        EditorField::Ubatch => "Microbatch",
        EditorField::CacheK => "K cache",
        EditorField::CacheV => "V cache",
        EditorField::Flash => "Flash attention",
        EditorField::GpuLayers => "GPU layers",
        EditorField::Split => "Split mode",
        EditorField::TensorSplit => "Tensor split",
        EditorField::Threads => "Threads",
        EditorField::ThreadsBatch => "Threads batch",
        EditorField::KvUnified => "Unified KV cache",
        EditorField::SpecType => "Speculative type",
        EditorField::SpecDraftModel => "Draft model",
        EditorField::SpecDraftNMax => "Draft tokens",
        EditorField::SpecDraftNgl => "Draft GPU layers",
        EditorField::Extra => "Extra flags",
        EditorField::Assign => "Default profile for model",
        EditorField::Advanced => "Advanced options",
        EditorField::Fit => "Fit",
        EditorField::FitTarget => "Fit target (MiB)",
        EditorField::LoadMode => "Load mode",
        EditorField::Mlock => "Keep model in memory",
        EditorField::DirectIO => "Direct I/O",
        EditorField::Numa => "NUMA",
        EditorField::KvOffload => "KV cache on GPU",
        EditorField::CpuMoe => "CPU experts",
        EditorField::DraftCpuMoe => "Draft CPU experts",
        EditorField::OverrideTensor => "Override tensor",
        EditorField::RopeBase => "RoPE frequency base",
        EditorField::RopeScale => "RoPE frequency scale",
        EditorField::Seed => "Seed",
        EditorField::Reasoning => "Reasoning",
        EditorField::Jinja => "Jinja templates",
        EditorField::ChatTemplate => "Chat template",
        EditorField::ChatTemplateFile => "Chat template file",
        EditorField::ChatTemplateKwargs => "Template kwargs",
    }
}
fn editor_field_value(field: EditorField, settings: &EditorSettings) -> String {
    let onoff = |value: bool| if value { "on" } else { "off" }.to_owned();
    match field {
        EditorField::Ctx => settings.ctx.to_string(),
        EditorField::ContextStep => settings.context_step.to_string(),
        EditorField::Parallel => settings.parallel.to_string(),
        EditorField::Batch => settings.batch.to_string(),
        EditorField::Ubatch => settings.ubatch.to_string(),
        EditorField::CacheK => settings.cache_k.clone(),
        EditorField::CacheV => settings.cache_v.clone(),
        EditorField::Flash => settings.flash.clone(),
        EditorField::GpuLayers => settings.gpu_layers.clone(),
        EditorField::Split => settings.split.clone(),
        EditorField::TensorSplit => settings.tensor_split.clone(),
        EditorField::Threads => settings.threads.to_string(),
        EditorField::ThreadsBatch => settings.threads_batch.to_string(),
        EditorField::KvUnified => onoff(settings.kv_unified),
        EditorField::SpecType => settings.spec_type.clone(),
        EditorField::SpecDraftModel => {
            if settings.spec_draft_model.is_empty() {
                "—".into()
            } else {
                settings.spec_draft_model.clone()
            }
        }
        EditorField::SpecDraftNMax => {
            if settings.spec_draft_nmax == 0 {
                "off".into()
            } else {
                settings.spec_draft_nmax.to_string()
            }
        }
        EditorField::SpecDraftNgl => settings.spec_draft_ngl.clone(),
        EditorField::Extra => {
            if settings.extra.is_empty() {
                "—".into()
            } else {
                settings.extra.join(" ")
            }
        }
        EditorField::Assign => {
            if settings.assign {
                "yes".into()
            } else {
                "no".into()
            }
        }
        EditorField::Advanced => String::new(),
        EditorField::Fit => settings.fit.clone(),
        EditorField::FitTarget => settings.fit_target.to_string(),
        EditorField::LoadMode => {
            if settings.load_mode.is_empty() {
                "(default)".into()
            } else {
                settings.load_mode.clone()
            }
        }
        EditorField::Mlock => onoff(settings.mlock),
        EditorField::DirectIO => onoff(settings.direct_io),
        EditorField::Numa => settings.numa.clone(),
        EditorField::KvOffload => onoff(settings.kv_offload),
        EditorField::CpuMoe => settings.n_cpu_moe.to_string(),
        EditorField::DraftCpuMoe => settings.spec_draft_n_cpu_moe.to_string(),
        EditorField::OverrideTensor => {
            if settings.override_tensor.is_empty() {
                "—".into()
            } else {
                settings.override_tensor.clone()
            }
        }
        EditorField::RopeBase => {
            if settings.rope_base.is_empty() {
                "(default)".into()
            } else {
                settings.rope_base.clone()
            }
        }
        EditorField::RopeScale => {
            if settings.rope_scale.is_empty() {
                "(default)".into()
            } else {
                settings.rope_scale.clone()
            }
        }
        EditorField::Seed => {
            if settings.seed.is_empty() {
                "(random)".into()
            } else {
                settings.seed.clone()
            }
        }
        EditorField::Reasoning => {
            if settings.reasoning.is_empty() {
                "(default)".into()
            } else {
                settings.reasoning.clone()
            }
        }
        EditorField::Jinja => onoff(settings.jinja),
        EditorField::ChatTemplate => {
            if settings.chat_template.is_empty() {
                "(model default)".into()
            } else {
                settings.chat_template.clone()
            }
        }
        EditorField::ChatTemplateFile => {
            if settings.chat_template_file.is_empty() {
                "—".into()
            } else {
                settings.chat_template_file.clone()
            }
        }
        EditorField::ChatTemplateKwargs => {
            if settings.chat_template_kwargs.is_empty() {
                "—".into()
            } else {
                settings.chat_template_kwargs.clone()
            }
        }
    }
}
fn editor_settings_from_profile(profile: &serde_json::Map<String, Value>) -> EditorSettings {
    let u = |key: &str, default: u64| {
        profile
            .get(key)
            .and_then(|value| {
                value
                    .as_u64()
                    .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
            })
            .unwrap_or(default)
    };
    let s = |key: &str, default: &str| {
        profile
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or(default)
            .to_owned()
    };
    let b = |key: &str| profile.get(key).and_then(Value::as_bool).unwrap_or(false);
    let mut extra: Vec<String> = profile
        .get("_extra_args")
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let mut kv_unified = !profile.contains_key("no-kv-unified");
    let mut jinja = !profile.contains_key("no-jinja");
    if extra.iter().any(|flag| flag == "--kv-unified") {
        kv_unified = true;
    } else if extra.iter().any(|flag| flag == "--no-kv-unified") {
        kv_unified = false;
    }
    if extra.iter().any(|flag| flag == "--jinja") {
        jinja = true;
    } else if extra.iter().any(|flag| flag == "--no-jinja") {
        jinja = false;
    }
    extra.retain(|flag| {
        flag != "--kv-unified"
            && flag != "--no-kv-unified"
            && flag != "--jinja"
            && flag != "--no-jinja"
    });
    let mut settings = EditorSettings {
        ctx: u("ctx-size", 4096),
        context_step: 4096,
        parallel: u("parallel", 1),
        batch: u("batch-size", 2048),
        ubatch: u("ubatch-size", 512),
        cache_k: s("cache-type-k", "f16"),
        cache_v: s("cache-type-v", "f16"),
        flash: s("flash-attn", "on"),
        gpu_layers: s("n-gpu-layers", "all"),
        split: s("split-mode", "layer"),
        tensor_split: s("tensor-split", "1,1,1,1"),
        threads: u("threads", 32),
        threads_batch: u("threads-batch", 32),
        kv_unified,
        spec_type: s("spec-type", "none"),
        spec_draft_model: s("spec-draft-model", ""),
        spec_draft_nmax: u("spec-draft-n-max", 0),
        spec_draft_ngl: s("spec-draft-ngl", "99"),
        fit: s("fit", "on"),
        fit_target: u("fit-target", 512),
        load_mode: s("load-mode", ""),
        mlock: b("mlock"),
        direct_io: b("direct-io"),
        numa: s("numa", "off"),
        n_cpu_moe: u("n-cpu-moe", 0),
        spec_draft_n_cpu_moe: u("spec-draft-n-cpu-moe", 0),
        kv_offload: !profile.contains_key("no-kv-offload"),
        override_tensor: s("override-tensor", ""),
        rope_base: s("rope-freq-base", ""),
        rope_scale: s("rope-freq-scale", ""),
        seed: s("seed", ""),
        reasoning: s("reasoning", ""),
        jinja,
        chat_template: s("chat-template", ""),
        chat_template_file: s("chat-template-file", ""),
        chat_template_kwargs: s("chat-template-kwargs", ""),
        extra,
        assign: false,
    };
    if settings.ubatch > settings.batch {
        settings.ubatch = settings.batch;
    }
    settings
}
fn editor_profile_from_profile(
    original: &serde_json::Map<String, Value>,
    settings: &EditorSettings,
    owner: &str,
) -> serde_json::Map<String, Value> {
    const MANAGED: &[&str] = &[
        "_model",
        "ctx-size",
        "parallel",
        "batch-size",
        "ubatch-size",
        "cache-type-k",
        "cache-type-v",
        "flash-attn",
        "n-gpu-layers",
        "split-mode",
        "tensor-split",
        "threads",
        "threads-batch",
        "kv-unified",
        "no-kv-unified",
        "spec-type",
        "spec-draft-model",
        "spec-draft-n-max",
        "spec-draft-ngl",
        "spec-draft-n-cpu-moe",
        "fit",
        "fit-target",
        "load-mode",
        "mlock",
        "direct-io",
        "numa",
        "n-cpu-moe",
        "kv-offload",
        "no-kv-offload",
        "override-tensor",
        "rope-freq-base",
        "rope-freq-scale",
        "seed",
        "reasoning",
        "jinja",
        "no-jinja",
        "chat-template",
        "chat-template-file",
        "chat-template-kwargs",
        "_extra_args",
    ];
    let mut profile = original.clone();
    for key in MANAGED {
        profile.remove(*key);
    }
    profile.insert("_model".into(), Value::String(owner.into()));
    profile.insert("ctx-size".into(), Value::from(settings.ctx));
    profile.insert("parallel".into(), Value::from(settings.parallel));
    profile.insert("batch-size".into(), Value::from(settings.batch));
    profile.insert("ubatch-size".into(), Value::from(settings.ubatch));
    profile.insert(
        "cache-type-k".into(),
        Value::String(settings.cache_k.clone()),
    );
    profile.insert(
        "cache-type-v".into(),
        Value::String(settings.cache_v.clone()),
    );
    profile.insert("flash-attn".into(), Value::String(settings.flash.clone()));
    profile.insert(
        "n-gpu-layers".into(),
        Value::String(settings.gpu_layers.clone()),
    );
    profile.insert("split-mode".into(), Value::String(settings.split.clone()));
    profile.insert(
        "tensor-split".into(),
        Value::String(settings.tensor_split.clone()),
    );
    profile.insert("threads".into(), Value::from(settings.threads));
    profile.insert("threads-batch".into(), Value::from(settings.threads_batch));
    if settings.kv_unified {
        profile.insert("kv-unified".into(), Value::Bool(true));
    } else {
        profile.insert("no-kv-unified".into(), Value::Bool(true));
    }
    if settings.spec_type != "none" && !settings.spec_type.is_empty() {
        profile.insert(
            "spec-type".into(),
            Value::String(settings.spec_type.clone()),
        );
        profile.insert(
            "spec-draft-n-max".into(),
            Value::from(settings.spec_draft_nmax),
        );
    }
    if !settings.spec_draft_model.is_empty() {
        profile.insert(
            "spec-draft-model".into(),
            Value::String(settings.spec_draft_model.clone()),
        );
        profile.insert(
            "spec-draft-ngl".into(),
            Value::String(settings.spec_draft_ngl.clone()),
        );
        if settings.spec_draft_n_cpu_moe != 0 {
            profile.insert(
                "spec-draft-n-cpu-moe".into(),
                Value::from(settings.spec_draft_n_cpu_moe),
            );
        }
    }
    if settings.fit == "off" {
        profile.insert("fit".into(), Value::String("off".into()));
    } else {
        profile.insert("fit-target".into(), Value::from(settings.fit_target));
    }
    if !settings.load_mode.is_empty() {
        profile.insert(
            "load-mode".into(),
            Value::String(settings.load_mode.clone()),
        );
    }
    if settings.mlock {
        profile.insert("mlock".into(), Value::Bool(true));
    }
    if settings.direct_io {
        profile.insert("direct-io".into(), Value::Bool(true));
    }
    if settings.numa != "off" {
        profile.insert("numa".into(), Value::String(settings.numa.clone()));
    }
    if settings.n_cpu_moe != 0 {
        profile.insert("n-cpu-moe".into(), Value::from(settings.n_cpu_moe));
    }
    if !settings.kv_offload {
        profile.insert("no-kv-offload".into(), Value::Bool(true));
    }
    if !settings.override_tensor.is_empty() {
        profile.insert(
            "override-tensor".into(),
            Value::String(settings.override_tensor.clone()),
        );
    }
    if !settings.rope_base.is_empty() {
        profile.insert(
            "rope-freq-base".into(),
            Value::String(settings.rope_base.clone()),
        );
    }
    if !settings.rope_scale.is_empty() {
        profile.insert(
            "rope-freq-scale".into(),
            Value::String(settings.rope_scale.clone()),
        );
    }
    if !settings.seed.is_empty() {
        profile.insert("seed".into(), Value::String(settings.seed.clone()));
    }
    if settings.jinja {
        profile.insert("jinja".into(), Value::Bool(true));
    } else {
        profile.insert("no-jinja".into(), Value::Bool(true));
    }
    if !settings.chat_template_file.is_empty() {
        profile.insert(
            "chat-template-file".into(),
            Value::String(settings.chat_template_file.clone()),
        );
    } else if !settings.chat_template.is_empty() {
        profile.insert(
            "chat-template".into(),
            Value::String(settings.chat_template.clone()),
        );
    }
    if !settings.chat_template_kwargs.is_empty() {
        profile.insert(
            "chat-template-kwargs".into(),
            Value::String(settings.chat_template_kwargs.clone()),
        );
    }
    if !settings.reasoning.is_empty() {
        profile.insert(
            "reasoning".into(),
            Value::String(settings.reasoning.clone()),
        );
    }
    if settings.extra.is_empty() {
        profile.remove("_extra_args");
    } else {
        profile.insert(
            "_extra_args".into(),
            Value::Array(
                settings
                    .extra
                    .iter()
                    .map(|flag| Value::String(flag.clone()))
                    .collect(),
            ),
        );
    }
    profile
}

fn profile_page(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let wide = area.width >= 140;
    let show_memory = area.width >= 70;
    let gib = (1u64 << 30) as f64;
    let free_vram = (app
        .telemetry
        .vram_total
        .saturating_sub(app.telemetry.vram_used)) as f64
        / gib;
    let rows = app
        .profiles
        .profiles
        .keys()
        .map(|name| {
            let estimate = app.profile_estimates.get(name).copied();
            let (vram, ram) = estimate.unwrap_or_default();
            let owner = app.profiles.owner(name);
            let pinned = app.cfg.scheduler_pinned_models.iter().any(|p| p == name);
            let is_default = owner
                .and_then(|owner| app.profiles.models.get(owner).and_then(Value::as_str))
                == Some(name.as_str());
            let rate = app
                .last_tok
                .get(name)
                .copied()
                .or_else(|| owner.and_then(|owner| app.last_tok.get(owner)).copied());
            let mut name_spans = vec![Span::raw(format!(" {name}"))];
            if pinned {
                name_spans.push(Span::styled(" ◆", Style::default().fg(Color::Yellow)));
            }
            if is_default {
                name_spans.push(Span::styled(" ★", Style::default().fg(Color::Cyan)));
            }
            let vram_color = if vram <= 0.0 {
                Color::DarkGray
            } else if free_vram > 0.0 && vram <= free_vram * 0.85 {
                Color::Green
            } else if free_vram > 0.0 && vram <= free_vram {
                Color::Yellow
            } else {
                Color::Red
            };
            let (tok_text, tok_color) = match rate {
                Some(rate) if rate > 0.0 => (format!("{rate:.1}"), Color::Green),
                _ => ("--".into(), Color::DarkGray),
            };
            let latest = app
                .benchmarks
                .profiles
                .get(name)
                .and_then(|runs| runs.last());
            let benchmark_decode = latest.and_then(benchmark_decode_median);
            let mut cells = vec![Line::from(name_spans)];
            if show_memory {
                cells.push(Line::from(Span::styled(
                    estimate
                        .map(|_| format!("{vram:.1}G"))
                        .unwrap_or_else(|| "…".into()),
                    Style::default().fg(vram_color),
                )));
                cells.push(Line::from(Span::styled(
                    estimate
                        .map(|_| format!("{ram:.1}G"))
                        .unwrap_or_else(|| "…".into()),
                    Style::default().fg(if estimate.is_some() {
                        Color::White
                    } else {
                        Color::DarkGray
                    }),
                )));
            }
            if wide {
                let context = app
                    .profiles
                    .profiles
                    .get(name)
                    .and_then(|profile| profile.get("ctx-size"))
                    .and_then(|value| {
                        value
                            .as_u64()
                            .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
                    })
                    .unwrap_or(app.cfg.ctx_size);
                cells.push(Line::raw(context.to_string()));
                for case_name in ["small", "medium", "long"] {
                    let case =
                        latest.and_then(|run| run.cases.iter().find(|case| case.name == case_name));
                    cells.push(benchmark_metric_cell(
                        case.map(|case| case.prompt_tokens_per_second),
                        Color::Yellow,
                    ));
                    cells.push(benchmark_metric_cell(
                        case.map(|case| case.decode_median_tokens_per_second),
                        Color::Green,
                    ));
                }
            } else {
                cells.push(benchmark_metric_cell(benchmark_decode, Color::Green));
            }
            cells.push(Line::from(Span::styled(
                tok_text,
                Style::default().fg(tok_color),
            )));
            Row::new(cells)
        })
        .collect::<Vec<_>>();
    let (headers, widths) = if wide {
        (
            vec![
                "PROFILE", "VRAM", "RAM", "CTX", "PP-S", "T/S-S", "PP-M", "T/S-M", "PP-L", "T/S-L",
                "LIVE T/S",
            ],
            vec![
                Constraint::Percentage(30),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(9),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(11),
            ],
        )
    } else if show_memory {
        (
            vec!["PROFILE", "VRAM", "RAM", "BENCH T/S", "LIVE T/S"],
            vec![
                Constraint::Percentage(30),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(11),
            ],
        )
    } else {
        (
            vec!["PROFILE", "BENCH T/S", "LIVE T/S"],
            vec![
                Constraint::Percentage(50),
                Constraint::Length(10),
                Constraint::Length(11),
            ],
        )
    };
    let table = Table::new(rows, widths)
        .header(
            Row::new(headers).style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        )
        .row_highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ")
        .block(title("PROFILES"));
    let mut state = ratatui::widgets::TableState::default().with_selected(Some(app.selected));
    frame.render_stateful_widget(table, area, &mut state);
}
fn benchmark_decode_median(run: &benchmark::BenchmarkRun) -> Option<f64> {
    let mut values = run
        .cases
        .iter()
        .map(|case| case.decode_median_tokens_per_second)
        .filter(|value| *value > 0.0)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    match values.len() {
        0 => None,
        count if count % 2 == 1 => Some(values[count / 2]),
        count => Some((values[count / 2 - 1] + values[count / 2]) / 2.0),
    }
}
fn benchmark_metric_cell(value: Option<f64>, color: Color) -> Line<'static> {
    match value {
        Some(value) => Line::from(Span::styled(
            format!("{value:.1}"),
            Style::default().fg(color),
        )),
        None => Line::from(Span::styled("--", Style::default().fg(Color::DarkGray))),
    }
}
fn settings_page(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let block = title("SETTINGS");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = settings(&app.cfg);
    let offset = app
        .selected
        .saturating_add(1)
        .saturating_sub(inner.height as usize);
    for (visible_index, (label, value)) in rows
        .iter()
        .enumerate()
        .skip(offset)
        .take(inner.height as usize)
    {
        let row = Rect {
            y: inner.y + (visible_index - offset) as u16,
            height: 1,
            ..inner
        };
        let selected = visible_index == app.selected;
        let row_style = if selected {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        frame.render_widget(Block::default().style(row_style), row);
        match value {
            SettingValue::Text(value) => frame.render_widget(
                Paragraph::new(format!("  {label:<26} {value}")).style(row_style),
                row,
            ),
            SettingValue::Boolean { checked, detail } => {
                let indicator_style = if selected {
                    row_style
                } else if *checked {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                frame.render_widget(
                    Paragraph::new(format!("  {label:<26} ")).style(row_style),
                    row,
                );
                let indicator_x = (row.x + 29).min(row.right());
                frame.render_widget(
                    Checkbox::new("", *checked)
                        .checked_symbol("●")
                        .unchecked_symbol("○")
                        .style(row_style)
                        .checkbox_style(indicator_style),
                    Rect {
                        x: indicator_x,
                        width: row.right().saturating_sub(indicator_x).min(2),
                        ..row
                    },
                );
                if !detail.is_empty() {
                    let detail_x = (indicator_x + 2).min(row.right());
                    frame.render_widget(
                        Paragraph::new(detail.clone()).style(row_style),
                        Rect {
                            x: detail_x,
                            width: row.right().saturating_sub(detail_x),
                            ..row
                        },
                    );
                }
            }
        }
    }
}
fn logs(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let lines = app.log.lines().map(log_line).collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lines)
            .block(title("LOGS"))
            .wrap(Wrap { trim: false }),
        area,
    );
}
fn log_line(line: &str) -> Line<'static> {
    let Some((timestamp, message)) = line.split_once(' ').filter(|(timestamp, _)| {
        let parts = timestamp.split('.').collect::<Vec<_>>();
        parts.len() == 4
            && parts
                .iter()
                .all(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_digit()))
    }) else {
        return Line::from(Span::styled(line.to_owned(), log_line_style(line)));
    };
    Line::from(vec![
        Span::styled(
            format!("[{timestamp}] "),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(message.to_owned(), log_line_style(line)),
    ])
}
fn log_line_style(line: &str) -> Style {
    let lower = line.to_ascii_lowercase();
    if lower.starts_with("[error]") || line.contains(" E ") {
        Style::default().fg(Color::Red)
    } else if lower.starts_with("[warn]") || line.contains(" W ") {
        Style::default().fg(Color::Yellow)
    } else if lower.starts_with("[info]") || line.contains(" I ") {
        Style::default().fg(Color::Green)
    } else if lower.contains("error") || lower.contains("assert") || lower.contains("✗") {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}
fn lan_ip() -> &'static str {
    static IP: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    IP.get_or_init(|| {
        std::process::Command::new("hostname")
            .arg("-I")
            .output()
            .ok()
            .and_then(|out| {
                String::from_utf8_lossy(&out.stdout)
                    .split_whitespace()
                    .find(|ip| !ip.starts_with("127.") && !ip.contains(':'))
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "unknown".into())
    })
    .as_str()
}
fn system(frame: &mut ratatui::Frame, app: &App, area: Rect) {
    let llama_label = match &app.last_check {
        Some((tag, changed, _, _)) => {
            if *changed {
                format!("Update llama.cpp → {tag}")
            } else {
                format!("llama.cpp — current ({tag})")
            }
        }
        None => "Update llama.cpp".into(),
    };
    let swap_label = match &app.last_check {
        Some((_, _, tag, changed)) => {
            if *changed {
                format!("Update llama-swap → {tag}")
            } else {
                format!("llama-swap — current ({tag})")
            }
        }
        None => "Update llama-swap".into(),
    };
    let values = [
        "Check for updates".to_owned(),
        llama_label,
        swap_label,
        if service_installed() {
            "Service installed".into()
        } else {
            "Install systemd service".into()
        },
    ];
    let mut state = ListState::default().with_selected(Some(app.selected));
    frame.render_stateful_widget(
        List::new(values.iter().map(|value| ListItem::new(value.as_str())))
            .block(title("MAINTENANCE"))
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("› "),
        area,
        &mut state,
    );
}
fn service_installed() -> bool {
    directories::BaseDirs::new()
        .map(|base| {
            base.home_dir()
                .join(".config/systemd/user/llamactl.service")
                .is_file()
        })
        .unwrap_or(false)
}
fn available_runtimes(paths: &Paths, configured: &str) -> Vec<String> {
    let mut runtimes = vec!["managed".to_owned()];
    if let Ok(entries) = fs::read_dir(&paths.versions) {
        let mut managed = entries
            .filter_map(Result::ok)
            .filter(|entry| entry.path().join("llama-server").is_file())
            .map(|entry| format!("managed:{}", entry.file_name().to_string_lossy()))
            .collect::<Vec<_>>();
        managed.sort();
        runtimes.extend(managed);
    }
    if let Some(home) = directories::BaseDirs::new().map(|base| base.home_dir().to_owned()) {
        let backends = home.join(".lmstudio/extensions/backends");
        if let Ok(entries) = fs::read_dir(backends) {
            let mut lmstudio = entries
                .filter_map(Result::ok)
                .filter(|entry| {
                    let path = entry.path();
                    path.join("llama-server").is_file()
                        && path.join("backend-manifest.json").is_file()
                        && entry
                            .file_name()
                            .to_string_lossy()
                            .starts_with("llama.cpp-")
                })
                .map(|entry| format!("lmstudio:{}", entry.file_name().to_string_lossy()))
                .collect::<Vec<_>>();
            lmstudio.sort();
            runtimes.extend(lmstudio);
        }
    }
    if !runtimes.iter().any(|runtime| runtime == configured) {
        runtimes.push(configured.to_owned());
    }
    runtimes
}
enum SettingValue {
    Text(String),
    Boolean { checked: bool, detail: String },
}

fn context_step(c: &Config) -> u64 {
    (4096.0 * c.context_step_scale)
        .round()
        .clamp(1024.0, 65_536.0) as u64
}

fn settings(c: &Config) -> Vec<(String, SettingValue)> {
    vec![
        (
            "LAN access".into(),
            SettingValue::Boolean {
                checked: c.host != "127.0.0.1",
                detail: if c.host == "127.0.0.1" {
                    "localhost only".into()
                } else {
                    format!("- {}", lan_ip())
                },
            },
        ),
        ("API port".into(), SettingValue::Text(c.port.to_string())),
        (
            "Telemetry sidecar port".into(),
            SettingValue::Text(c.telemetry_port.to_string()),
        ),
        ("Runtime".into(), SettingValue::Text(c.runtime.clone())),
        ("Target".into(), SettingValue::Text(c.backend.clone())),
        (
            "Context size".into(),
            SettingValue::Text(c.ctx_size.to_string()),
        ),
        (
            "Scheduler".into(),
            SettingValue::Boolean {
                checked: c.scheduler_enabled,
                detail: String::new(),
            },
        ),
        (
            "Scheduler VRAM fraction".into(),
            SettingValue::Text(format!("{:.0}%", c.scheduler_vram_fraction * 100.0)),
        ),
        (
            "Pinned models".into(),
            SettingValue::Text(if c.scheduler_pinned_models.is_empty() {
                "—".into()
            } else {
                c.scheduler_pinned_models.join(", ")
            }),
        ),
        (
            "Advertise base models".into(),
            SettingValue::Boolean {
                checked: c.advertise_base_models,
                detail: String::new(),
            },
        ),
        (
            "Advertise profiles".into(),
            SettingValue::Boolean {
                checked: c.advertise_profiles,
                detail: String::new(),
            },
        ),
        (
            "Start on boot".into(),
            SettingValue::Boolean {
                checked: crate::service_enabled(),
                detail: String::new(),
            },
        ),
    ]
}
fn compute_profile_estimates(
    cfg: &Config,
    profiles: &Profiles,
    models: &[models::Model],
    tx: &std::sync::mpsc::Sender<ProfileEstimateUpdate>,
) {
    let gib = (1u64 << 30) as f64;
    // Tensor-placement overrides require scanning large tensor directories. Do
    // the inexpensive profiles first and stream every result to the UI instead
    // of withholding the whole table until the slowest model is finished.
    let mut names = profiles.profiles.keys().collect::<Vec<_>>();
    names.sort_by_key(|name| profiles.profiles[*name].contains_key("override-tensor"));
    for name in names {
        let Some(owner) = profiles.owner(name) else {
            continue;
        };
        let Some(model) = models.iter().find(|m| m.id == owner) else {
            continue;
        };
        let Ok(args) = profiles.args(name) else {
            continue;
        };
        let mut full = process::common_args(cfg);
        full.extend(args);
        let est = models::estimate(&model.path, &full);
        if tx
            .send(ProfileEstimateUpdate::Estimate(
                name.clone(),
                (est.vram as f64 / gib, est.ram as f64 / gib),
            ))
            .is_err()
        {
            return;
        }
    }
    let _ = tx.send(ProfileEstimateUpdate::Done);
}

fn latest_gpu_samples(payload: &Value) -> Vec<&Value> {
    // The llamactl sidecar already returns one current sample per device.
    if let Some(gpus) = payload.get("gpus").and_then(Value::as_array) {
        return gpus.iter().collect();
    }
    // llama-swap's /api/performance response is a time series. Keep only the
    // newest sample for each device instead of treating every sample as a GPU.
    let mut latest = BTreeMap::new();
    for sample in payload
        .get("gpu_stats")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(id) = sample.get("id").and_then(Value::as_u64) {
            latest.insert(id, sample);
        }
    }
    latest.into_values().collect()
}

/// Runtime-reported (total, free) device memory, cached briefly.
///
/// Unlike capacity this genuinely changes, so it cannot be cached for the
/// process lifetime — but it costs a subprocess, and telemetry is polled on a
/// timer, so re-running it on every tick would fork `llama-server` several
/// times a second. A few seconds of staleness is the right trade for a number
/// that is only consulted while idle.
fn runtime_device_memory() -> Option<(u64, u64)> {
    use std::sync::Mutex;
    static CACHE: Mutex<Option<(std::time::Instant, Option<(u64, u64)>)>> = Mutex::new(None);
    const TTL: Duration = Duration::from_secs(5);

    let mut guard = CACHE.lock().ok()?;
    if let Some((stamp, value)) = guard.as_ref()
        && stamp.elapsed() < TTL
    {
        return *value;
    }
    let fresh = crate::process::device_memory_bytes();
    *guard = Some((std::time::Instant::now(), fresh));
    fresh
}

/// Card capacity in bytes, cached for the process lifetime.
///
/// fdinfo reports usage but never capacity — the DRM usage-stats interface has
/// no such field. amdgpu publishes `mem_info_vram_total` in sysfs; xe and i915
/// publish nothing equivalent, so on Intel we fall back to asking the runtime
/// itself via `llama-server --list-devices`, whose `(TOTAL MiB, FREE MiB free)`
/// format is emitted by llama.cpp's common code and is therefore identical
/// across CUDA, ROCm, Vulkan and SYCL.
///
/// Cached because that fallback spawns a subprocess and telemetry is polled on
/// a timer. Capacity is unlikely to change while a runtime is loaded — though
/// it is not impossible: PCIe hotplug, SR-IOV VFs appearing or disappearing,
/// and passthrough changes inside a container can all move it.
///
/// Note: It *is* backend-dependent, though: the same Arc B70 reports 32656 MiB
/// under Vulkan and 31023 MiB under SYCL. Switching backend therefore needs a
/// restart before the figure is right again.
fn vram_capacity_bytes() -> u64 {
    static CAPACITY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAPACITY.get_or_init(|| {
        let sysfs = crate::drm::device_total_bytes();
        if sysfs > 0 {
            return sysfs;
        }
        crate::process::installed_vram_bytes()
    })
}

fn system_telemetry(cfg: &Config, probe_api: bool) -> Telemetry {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mem = |name: &str| {
        text.lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
    };
    let ram_total = mem("MemTotal:");
    let ram_used = ram_total.saturating_sub(mem("MemAvailable:"));
    let mut telemetry = Telemetry {
        ram_used,
        ram_total,
        ..Telemetry::default()
    };
    if probe_api
        && let Ok(client) = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()
    {
        let api_key = cfg.keys().into_iter().next();
        let endpoints = [
            format!(
                "http://{}:{}/api/llamactl/telemetry",
                cfg.host, cfg.telemetry_port
            ),
            format!("http://{}:{}/api/performance", cfg.host, cfg.port),
        ];
        for endpoint in endpoints {
            let mut request = client.get(endpoint);
            if let Some(key) = &api_key {
                request = request.bearer_auth(key);
            }
            let Ok(payload) = request
                .send()
                .and_then(|response| response.error_for_status())
                .and_then(|response| response.json::<Value>())
            else {
                continue;
            };
            let gpus = latest_gpu_samples(&payload);
            if gpus.is_empty() {
                continue;
            }
            for gpu in gpus {
                let used = gpu
                    .get("mem_used_mb")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let total = gpu
                    .get("mem_total_mb")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                telemetry.vram_used += (used * 1024.0 * 1024.0) as u64;
                telemetry.vram_total += (total * 1024.0 * 1024.0) as u64;
                if let Some(temp) = gpu.get("temp_c").and_then(Value::as_f64) {
                    telemetry.gpu_temps.push(temp);
                }
            }
            return telemetry;
        }
    }
    // DRM fdinfo: the only cross-vendor memory interface in the kernel, and on
    // Intel the only VRAM source that exists at all — sysfs publishes no counter,
    // the xe PMU exposes engines and clocks only, and debugfs is unavailable in a
    // container. Tried before nvidia-smi so non-NVIDIA hardware reports something
    // instead of zero; on an NVIDIA box this yields nothing and we fall through.

    // Tier 1 — DRM fdinfo. Exact, per-process, no subprocess, vendor-neutral.
    // Only sees clients that exist, so it reads zero before anything is loaded.
    let drm = crate::drm::read();
    let drm_used = drm.total_allocated();
    if drm_used > 0 {
        telemetry.vram_used += drm_used;
        telemetry.vram_total += vram_capacity_bytes();
        telemetry.gpu_temps.extend(crate::drm::temperatures());
        return telemetry;
    }

    // Tier 2 — ask the runtime. This is the idle case: nothing of ours is
    // loaded, so fdinfo has nothing to report, but the card is not necessarily
    // empty. The driver's own view is what a fit calculation needs here.
    if let Some((total, free)) = runtime_device_memory() {
        telemetry.vram_used += total.saturating_sub(free);
        telemetry.vram_total += total;
        telemetry.gpu_temps.extend(crate::drm::temperatures());
        return telemetry;
    }

    // Tier 3 — nvidia-smi, below.
    if let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let values = line
                .split(',')
                .filter_map(|value| value.trim().parse::<f64>().ok())
                .collect::<Vec<_>>();
            if values.len() == 3 {
                telemetry.vram_used += (values[0] * 1024.0 * 1024.0) as u64;
                telemetry.vram_total += (values[1] * 1024.0 * 1024.0) as u64;
                telemetry.gpu_temps.push(values[2]);
            }
        }
    }
    telemetry
}

fn runtime_version(paths: &Paths) -> String {
    let manifest = paths.current.join(".llamactl-build.json");
    std::fs::read_to_string(manifest)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|json| json.get("tag").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_default()
}

fn serving_telemetry(cfg: &Config, paths: &Paths) -> Telemetry {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(700))
        .build()
    {
        Ok(client) => client,
        Err(_) => return Telemetry::default(),
    };
    let api_key = cfg.keys().into_iter().next();
    let headers = |request: reqwest::blocking::RequestBuilder| match &api_key {
        Some(key) => request.bearer_auth(key),
        None => request,
    };
    let base = format!("http://{}:{}", cfg.host, cfg.port);
    let mut telemetry = Telemetry {
        llama_cpp_version: runtime_version(paths),
        ..Telemetry::default()
    };
    let mut targets = vec![(base.clone(), telemetry.model_name.clone())];
    let mut swap_detected = false;
    if let Ok(response) = headers(client.get(format!("{base}/running"))).send()
        && response.status().is_success()
        && let Ok(payload) = response.json::<Value>()
    {
        swap_detected = true;
        let running = payload
            .get("running")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        targets = running
            .iter()
            .filter_map(|item| {
                let proxy = item.get("proxy").and_then(Value::as_str)?;
                let model = item
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                Some((proxy.replace("localhost", "127.0.0.1"), model.to_owned()))
            })
            .collect();
        let models = running
            .iter()
            .filter_map(|item| item.get("model").and_then(Value::as_str))
            .collect::<Vec<_>>();
        telemetry.model_name = models.join(", ");
        if !models.is_empty() {
            telemetry.model_state = if running.iter().all(|item| {
                item.get("state")
                    .and_then(Value::as_str)
                    .is_some_and(|state| state.eq_ignore_ascii_case("ready"))
            }) {
                ModelState::Loaded
            } else {
                ModelState::Loading
            };
        }
    }
    if !swap_detected
        && let Ok(response) = headers(client.get(format!("{base}/v1/models"))).send()
        && response.status().is_success()
        && let Ok(payload) = response.json::<Value>()
    {
        let names = payload
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("id").and_then(Value::as_str))
            .collect::<Vec<_>>();
        if !names.is_empty() {
            telemetry.model_name = names.join(", ");
            telemetry.model_state = ModelState::Loaded;
        }
    }
    if swap_detected
        && let Ok(response) = headers(client.get(format!(
            "{base}/api/metrics/activity?page=1&limit=999&order=desc"
        )))
        .send()
        && response.status().is_success()
        && let Ok(activity) = response.json::<Value>()
        && let Some(last) = activity
            .get("data")
            .and_then(Value::as_array)
            .and_then(|data| data.first())
    {
        let tokens = last.get("tokens").unwrap_or(last);
        telemetry.last_request = Some(RequestPerformance {
            model: last
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            prompt_tokens: tokens
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            output_tokens: tokens
                .get("output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            cache_tokens: tokens
                .get("cache_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            draft_tokens: tokens
                .get("draft_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            draft_accepted: tokens
                .get("draft_acc_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            prompt_tok_s: tokens
                .get("prompt_per_second")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            generation_tok_s: tokens
                .get("tokens_per_second")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            ttft_ms: last
                .get("time_to_first_token_ms")
                .or_else(|| {
                    last.get("metadata")
                        .and_then(|metadata| metadata.get("time_to_first_token_ms"))
                })
                .and_then(Value::as_f64),
            duration_ms: last.get("duration_ms").and_then(Value::as_u64).unwrap_or(0),
        });
        // llama-swap retains activity across model unloads and manager restarts.
        // Keep the newest measured generation rate for every advertised model.
        for request in activity
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(model) = request.get("model").and_then(Value::as_str) else {
                continue;
            };
            let tokens = request.get("tokens").unwrap_or(request);
            let Some(rate) = tokens.get("tokens_per_second").and_then(Value::as_f64) else {
                continue;
            };
            if rate > 0.0 {
                telemetry
                    .historical_tok_s
                    .entry(model.to_owned())
                    .or_insert(rate);
            }
        }
    }
    if swap_detected
        && let Ok(response) = headers(client.get(format!("{base}/api/metrics/stats"))).send()
        && response.status().is_success()
        && let Ok(stats) = response.json::<Value>()
    {
        telemetry.total_requests = stats
            .get("total_requests")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        telemetry.total_input_tokens = stats
            .get("total_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        telemetry.total_output_tokens = stats
            .get("total_output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        telemetry.total_cache_tokens = stats
            .get("total_cache_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
    }
    if telemetry.model_state == ModelState::None
        && process::pid(paths).is_some()
        && let Ok(text) = std::fs::read_to_string(&paths.launch)
        && let Ok(spec) = serde_json::from_str::<process::LaunchSpec>(&text)
        && let Some(model) = spec.model
    {
        telemetry.model_name = model;
        telemetry.model_state = ModelState::Loading;
    }

    if telemetry.llama_cpp_version.is_empty() {
        for (target, _) in &targets {
            if let Ok(response) =
                headers(client.get(format!("{}/props", target.trim_end_matches('/')))).send()
                && response.status().is_success()
                && let Ok(props) = response.json::<Value>()
            {
                telemetry.llama_cpp_version = props
                    .pointer("/metadata/general")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| {
                        props
                            .pointer("/metadata/llama.cpp.version")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                    })
                    .to_string();
                if !telemetry.llama_cpp_version.is_empty() {
                    break;
                }
            }
        }
    }
    for (target, model_name) in targets {
        let Ok(response) =
            headers(client.get(format!("{}/slots", target.trim_end_matches('/')))).send()
        else {
            continue;
        };
        let Ok(slots) = response.json::<Value>() else {
            continue;
        };
        for slot in slots.as_array().into_iter().flatten() {
            let slot_id = slot.get("id").and_then(Value::as_u64).unwrap_or(0) as usize;
            let is_processing = slot
                .get("is_processing")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let total = slot
                .get("n_prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let processed = slot
                .get("n_prompt_tokens_processed")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let decoded = slot
                .get("next_token")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|token| token.get("n_decoded").and_then(Value::as_u64))
                .sum::<u64>();
            let prompt_total = processed.max(total.saturating_sub(decoded));
            let prompt_progress = if prompt_total > 0 {
                (processed.min(prompt_total) as f64) / prompt_total as f64
            } else {
                0.0
            };
            telemetry.slot_details.push(SlotDetail {
                model_name: model_name.clone(),
                slot_id,
                prompt_progress,
                prompt_done: processed.min(prompt_total),
                decoded,
                pp_tok_s: None,
                td_tok_s: None,
                is_processing,
            });
            if is_processing {
                telemetry.prompt_done += processed.min(prompt_total);
                telemetry.prompt_total += prompt_total;
                telemetry.generated += decoded;
                telemetry.active_requests += 1;
            }
        }
    }
    telemetry
}

#[cfg(test)]
mod telemetry_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn performance_history_keeps_only_latest_sample_per_gpu() {
        let payload = json!({"gpu_stats": [
            {"id": 0, "timestamp": "old", "mem_total_mb": 100, "temp_c": 40},
            {"id": 1, "timestamp": "old", "mem_total_mb": 200, "temp_c": 41},
            {"id": 0, "timestamp": "new", "mem_total_mb": 100, "temp_c": 50},
            {"id": 1, "timestamp": "new", "mem_total_mb": 200, "temp_c": 51}
        ]});
        let samples = latest_gpu_samples(&payload);
        assert_eq!(samples.len(), 2);
        assert_eq!(
            samples[0].get("timestamp").and_then(Value::as_str),
            Some("new")
        );
        assert_eq!(
            samples[1].get("timestamp").and_then(Value::as_str),
            Some("new")
        );
    }

    #[test]
    fn compact_sidecar_gpu_samples_are_used_directly() {
        let payload = json!({"gpus": [
            {"id": 0, "mem_total_mb": 100},
            {"id": 1, "mem_total_mb": 200}
        ]});
        let samples = latest_gpu_samples(&payload);
        assert_eq!(samples.len(), 2);
        assert_eq!(
            samples
                .iter()
                .filter_map(|gpu| gpu.get("mem_total_mb").and_then(Value::as_u64))
                .sum::<u64>(),
            300
        );
    }
}

#[cfg(test)]
mod rename_editing_tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn app() -> App<'static> {
        let paths = Paths::discover().expect("paths");
        let cfg = Config::load(&paths).expect("config");

        let paths: &'static Paths = Box::leak(Box::new(paths));
        App::new(cfg, paths).expect("app")
    }

    #[test]
    fn editing_supports_arrows_home_end_and_insert_at_cursor() {
        let mut app = app();
        app.page = 2;
        app.selected = 0;
        app.start_profile_rename();
        let original = app.rename_input.as_ref().unwrap().original.clone();

        app.handle_rename_input(key(KeyCode::Home));
        app.handle_rename_input(key(KeyCode::Char('z')));
        app.handle_rename_input(key(KeyCode::Char('-')));
        assert!(app.rename_input.as_ref().unwrap().text.starts_with("z-"));
        assert_eq!(app.rename_input.as_ref().unwrap().cursor, 2);

        app.handle_rename_input(key(KeyCode::End));
        assert_eq!(
            app.rename_input.as_ref().unwrap().cursor,
            app.rename_input.as_ref().unwrap().text.chars().count()
        );
        let before = app.rename_input.as_ref().unwrap().text.clone();
        app.handle_rename_input(key(KeyCode::Backspace));
        assert_eq!(
            app.rename_input.as_ref().unwrap().text,
            before[..before.len() - 1]
        );

        let cursor_before = app.rename_input.as_ref().unwrap().cursor;
        app.handle_rename_input(key(KeyCode::Left));
        app.handle_rename_input(key(KeyCode::Left));
        app.handle_rename_input(key(KeyCode::Delete));
        assert_eq!(app.rename_input.as_ref().unwrap().cursor, cursor_before - 2);

        app.handle_rename_input(ctrl('u'));
        let state = app.rename_input.as_ref().unwrap();
        assert_eq!(state.cursor, 0);

        app.handle_rename_input(key(KeyCode::Esc));
        assert!(app.rename_input.is_none());
        assert!(app.profiles.profiles.contains_key(&original));
    }

    #[test]
    fn ctrl_edits_delete_word_line_and_jump_words() {
        let mut app = app();
        app.page = 2;
        app.selected = 0;
        app.start_profile_rename();
        let state = app.rename_input.as_ref().unwrap().clone();
        let len = state.text.chars().count();

        app.handle_rename_input(ctrl('a'));
        assert_eq!(app.rename_input.as_ref().unwrap().cursor, 0);
        app.handle_rename_input(ctrl('e'));
        assert_eq!(
            app.rename_input.as_ref().unwrap().cursor,
            app.rename_input.as_ref().unwrap().text.chars().count()
        );

        app.handle_rename_input(KeyEvent::new(KeyCode::Left, KeyModifiers::CONTROL));
        let after_jump = app.rename_input.as_ref().unwrap().cursor;
        assert!(after_jump < len);

        let before = app.rename_input.as_ref().unwrap().text.clone();
        app.handle_rename_input(ctrl('w'));
        assert!(app.rename_input.as_ref().unwrap().text.len() < before.len());

        app.handle_rename_input(ctrl('k'));
        let state = app.rename_input.as_ref().unwrap();
        assert_eq!(state.cursor, state.text.chars().count());
        app.handle_rename_input(key(KeyCode::Esc));
    }
}

#[cfg(test)]
mod profile_editor_tests {
    use super::*;
    use serde_json::json;

    fn map(value: serde_json::Value) -> serde_json::Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn settings_roundtrip_preserves_unmanaged_keys() {
        let profile = map(json!({
            "_model": "m",
            "ctx-size": 65536,
            "parallel": 4,
            "spec-draft-model": "/x/draft.gguf",
            "chat-template-file": "/x/chat.jinja",
            "chat-template-kwargs": "{\"tools\":true}",
            "override-tensor": "blk.*=CPU",
            "split-mode": "tensor"
        }));
        let settings = editor_settings_from_profile(&profile);
        assert_eq!(settings.ctx, 65536);
        assert_eq!(settings.parallel, 4);
        assert_eq!(settings.split, "tensor");
        assert_eq!(settings.chat_template_file, "/x/chat.jinja");
        assert_eq!(settings.chat_template_kwargs, "{\"tools\":true}");
        let out = editor_profile_from_profile(&profile, &settings, "m");
        assert_eq!(out.get("ctx-size").and_then(Value::as_u64), Some(65536));
        assert_eq!(
            out.get("spec-draft-model").and_then(Value::as_str),
            Some("/x/draft.gguf")
        );
        assert_eq!(
            out.get("override-tensor").and_then(Value::as_str),
            Some("blk.*=CPU")
        );
        assert_eq!(
            out.get("chat-template-file").and_then(Value::as_str),
            Some("/x/chat.jinja")
        );
    }

    #[test]
    fn external_draft_spec_type_is_preserved() {
        let profile = map(json!({
            "_model": "m",
            "spec-type": "draft-dflash",
            "spec-draft-n-max": 10,
            "spec-draft-model": "/x/d.gguf"
        }));
        let settings = editor_settings_from_profile(&profile);
        assert_eq!(settings.spec_type, "draft-dflash");
        assert_eq!(settings.spec_draft_nmax, 10);
        assert_eq!(settings.spec_draft_model, "/x/d.gguf");
        let out = editor_profile_from_profile(&profile, &settings, "m");
        assert_eq!(
            out.get("spec-type").and_then(Value::as_str),
            Some("draft-dflash")
        );
        assert_eq!(
            out.get("spec-draft-n-max").and_then(Value::as_u64),
            Some(10)
        );
        assert_eq!(
            out.get("spec-draft-model").and_then(Value::as_str),
            Some("/x/d.gguf")
        );
    }

    #[test]
    fn mtp_on_emits_draft_mtp_and_off_strips_it() {
        let profile = map(json!({
            "_model": "m",
            "spec-type": "draft-mtp",
            "spec-draft-n-max": 3
        }));
        let mut settings = editor_settings_from_profile(&profile);
        assert_eq!(settings.spec_type, "draft-mtp");
        assert_eq!(settings.spec_draft_nmax, 3);
        settings.spec_draft_nmax = 5;
        let out = editor_profile_from_profile(&profile, &settings, "m");
        assert_eq!(
            out.get("spec-type").and_then(Value::as_str),
            Some("draft-mtp")
        );
        assert_eq!(out.get("spec-draft-n-max").and_then(Value::as_u64), Some(5));
        settings.spec_type = "none".into();
        settings.spec_draft_nmax = 0;
        let out = editor_profile_from_profile(&profile, &settings, "m");
        assert!(out.get("spec-type").is_none());
        assert!(out.get("spec-draft-n-max").is_none());
    }

    #[test]
    fn string_ctx_values_are_parsed() {
        let profile = map(json!({
            "_model": "m",
            "ctx-size": "262144",
            "batch-size": "2048"
        }));
        let settings = editor_settings_from_profile(&profile);
        assert_eq!(settings.ctx, 262144);
        assert_eq!(settings.batch, 2048);
    }
}
