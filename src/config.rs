use anyhow::{Context, Result, bail};
use directories::BaseDirs;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub const BACKENDS: &[&str] = &[
    "cuda",
    "vulkan",
    "cpu",
    "rocm",
    "sycl-fp16",
    "sycl-fp32",
    "openvino",
];

#[derive(Debug, Clone)]
pub struct Paths {
    pub data_dir: PathBuf,
    pub state_dir: PathBuf,
    pub config: PathBuf,
    pub profiles: PathBuf,
    pub versions: PathBuf,
    pub current: PathBuf,
    pub pid: PathBuf,
    pub launch: PathBuf,
    pub log: PathBuf,
    pub swap_bin: PathBuf,
    pub swap_config: PathBuf,
    pub metadata_cache: PathBuf,
    pub profile_benchmarks: PathBuf,
}

impl Paths {
    pub fn discover() -> Result<Self> {
        let base = BaseDirs::new().context("cannot determine home directory")?;
        let config_dir = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| base.home_dir().join(".config"))
            .join("llamactl");
        let data_dir = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| base.home_dir().join(".local/share"))
            .join("llamactl");
        let state_dir = std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| base.home_dir().join(".local/state"))
            .join("llamactl");
        Ok(Self {
            config: config_dir.join("config.json"),
            profiles: config_dir.join("profiles.json"),
            versions: data_dir.join("versions"),
            current: data_dir.join("current"),
            pid: state_dir.join("server.pid"),
            launch: state_dir.join("launch.json"),
            log: state_dir.join("server.log"),
            swap_bin: data_dir.join("llama-swap/llama-swap"),
            swap_config: state_dir.join("llama-swap.yaml"),
            metadata_cache: state_dir.join("gguf-metadata-cache.json"),
            profile_benchmarks: state_dir.join("profile-benchmarks.json"),
            data_dir,
            state_dir,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub backend: String,
    pub runtime: String,
    pub update_backends: Vec<String>,
    pub host: String,
    pub port: u16,
    pub telemetry_port: u16,
    pub api_key: String,
    pub api_keys: Vec<String>,
    pub models_dirs: Vec<PathBuf>,
    pub default_model: String,
    pub gpu_layers: Value,
    pub ctx_size: u64,
    pub extra_args: Vec<String>,
    /// Shell script sourced before the runtime binary is exec'd.
    ///
    /// Some backends cannot run from a bare exec: a SYCL build needs oneAPI's
    /// `setvars.sh` on the environment, and without it `llama-server` aborts
    /// with "No device of requested type available" — which looks like absent
    /// hardware rather than a missing environment. LD_LIBRARY_PATH alone is not
    /// enough; setvars sets considerably more than that.
    ///
    /// Empty means exec the binary directly, which is the behaviour every
    /// self-contained build wants.
    pub runtime_env_file: String,
    pub auto_update: bool,
    pub keep_versions: usize,
    pub swap_ttl: u64,
    pub advertise_base_models: bool,
    pub advertise_profiles: bool,
    pub scheduler_enabled: bool,
    pub scheduler_vram_fraction: f64,
    pub scheduler_max_models: usize,
    pub scheduler_pinned_models: Vec<String>,
    pub context_step_scale: f64,
}

impl Default for Config {
    fn default() -> Self {
        let home = BaseDirs::new()
            .map(|b| b.home_dir().to_path_buf())
            .unwrap_or_default();
        let mut models_dirs = [
            home.join(".lmstudio/models"),
            home.join(".cache/lm-studio/models"),
        ]
        .into_iter()
        .filter(|p| p.is_dir())
        .collect::<Vec<_>>();
        if let Ok(p) = Paths::discover() {
            models_dirs.push(p.data_dir.join("models"));
        }
        Self {
            backend: "vulkan".into(),
            runtime: "managed".into(),
            update_backends: vec!["vulkan".into(), "cuda".into()],
            host: "127.0.0.1".into(),
            port: 1234,
            telemetry_port: 1235,
            api_key: String::new(),
            api_keys: vec![],
            models_dirs,
            default_model: String::new(),
            gpu_layers: Value::String("all".into()),
            ctx_size: 4096,
            extra_args: vec![],
            runtime_env_file: String::new(),
            auto_update: true,
            keep_versions: 2,
            swap_ttl: 600,
            advertise_base_models: true,
            advertise_profiles: false,
            scheduler_enabled: true,
            scheduler_vram_fraction: 0.95,
            scheduler_max_models: 4,
            scheduler_pinned_models: vec![],
            context_step_scale: 1.0,
        }
    }
}

impl Config {
    pub fn load(paths: &Paths) -> Result<Self> {
        if !paths.config.exists() {
            let cfg = Self::default();
            cfg.save(paths)?;
            return Ok(cfg);
        }
        let text = fs::read_to_string(&paths.config)
            .with_context(|| format!("read {}", paths.config.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("invalid JSON in {}", paths.config.display()))
    }
    pub fn save(&self, paths: &Paths) -> Result<()> {
        atomic_json(&paths.config, self)
    }
    pub fn keys(&self) -> Vec<String> {
        let mut out = vec![];
        for key in self
            .api_key
            .split(',')
            .chain(self.api_keys.iter().map(String::as_str))
        {
            let key = key.trim();
            if !key.is_empty() && !out.iter().any(|v| v == key) {
                out.push(key.to_owned());
            }
        }
        out
    }
    pub fn set(&mut self, key: &str, raw: &str) -> Result<()> {
        let mut value = serde_json::to_value(self.clone())?;
        let map = value.as_object_mut().unwrap();
        let old = map
            .get(key)
            .with_context(|| format!("unknown config key '{key}'"))?;
        let parsed = match old {
            Value::String(_) => Value::String(raw.to_owned()),
            Value::Bool(_) => Value::Bool(raw.parse().context("expected true or false")?),
            Value::Number(n) if n.is_u64() => Value::Number(raw.parse::<u64>()?.into()),
            Value::Number(_) => serde_json::Number::from_f64(raw.parse()?)
                .map(Value::Number)
                .context("invalid number")?,
            Value::Array(_) => serde_json::from_str(raw).context("expected a JSON array")?,
            _ => serde_json::from_str(raw).context("invalid JSON value")?,
        };
        if key == "backend" && !BACKENDS.contains(&raw) {
            bail!("invalid backend '{raw}'\nvalid: {}", BACKENDS.join(", "));
        }
        map.insert(key.to_owned(), parsed);
        let candidate: Self = serde_json::from_value(value)?;
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        if !BACKENDS.contains(&self.backend.as_str()) {
            bail!("invalid backend '{}'", self.backend)
        }
        if !(0.1..=1.0).contains(&self.scheduler_vram_fraction) {
            bail!("scheduler_vram_fraction must be between 0.1 and 1.0")
        }
        if self.scheduler_max_models == 0 {
            bail!("scheduler_max_models must be at least 1")
        }
        if self.port == 0 || self.telemetry_port == 0 {
            bail!("ports must be between 1 and 65535")
        }
        if self.port == self.telemetry_port {
            bail!("telemetry_port must differ from the OpenAI API port")
        }
        if self.keep_versions == 0 {
            bail!("keep_versions must be at least 1")
        }
        Ok(())
    }
}

pub fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("path has no parent")?;
    fs::create_dir_all(parent)?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temp, value)?;
    writeln!(temp)?;
    temp.flush()?;
    temp.persist(path).map_err(|e| e.error)?;
    Ok(())
}
