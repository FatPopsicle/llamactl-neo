use crate::{
    config::{Config, Paths},
    models,
    profiles::Profiles,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{Read, Write},
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::LazyLock,
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchSpec {
    pub model: Option<String>,
    pub extra: Vec<String>,
    pub swap: bool,
}

pub fn pid(paths: &Paths) -> Option<u32> {
    let value = fs::read_to_string(&paths.pid).ok()?.trim().parse().ok()?;
    let stat = fs::read_to_string(format!("/proc/{value}/stat")).ok()?;

    let state = stat.rsplit_once(") ")?.1.chars().next()?;
    if state == 'Z' {
        let _ = fs::remove_file(&paths.pid);
        return None;
    }
    Some(value)
}

fn runtime_command(binary: &Path, paths: &Paths) -> Command {
    // A runtime that needs an environment script cannot simply be exec'd. Wrap
    // it in a shell that sources the script first, then execs the binary so no
    // extra process is left in the tree and signals/exit codes pass through
    // unchanged. Arguments appended by callers land in "$@" after the shift.
    let env_file = Config::load(paths)
        .ok()
        .map(|cfg| cfg.runtime_env_file)
        .filter(|path| !path.is_empty());
    let mut command = match env_file {
        Some(env_file) => {
            let mut shell = Command::new("bash");
            shell
                .arg("-c")
                .arg(r#"source "$1" >/dev/null 2>&1; shift; exec "$@""#)
                .arg("llamactl")
                .arg(env_file)
                .arg(binary);
            shell
        }
        None => Command::new(binary),
    };
    let runtime_binary = if binary == paths.swap_bin || binary == Path::new("/usr/bin/env") {
        server_binary(paths).unwrap_or_else(|| paths.current.join("llama-server"))
    } else {
        binary.to_owned()
    };
    let runtime = runtime_binary
        .parent()
        .and_then(|parent| parent.canonicalize().ok())
        .unwrap_or_else(|| paths.current.clone());
    let inherited = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
    let mut library_path = runtime.clone().into_os_string();
    let manifest = runtime.join("backend-manifest.json");
    if let Ok(text) = fs::read_to_string(manifest)
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(packages) = value
            .get("vendor_lib_package_names")
            .and_then(|item| item.as_array())
        && let Some(backends) = runtime.parent()
    {
        for package in packages.iter().filter_map(|item| item.as_str()) {
            let vendor = backends.join("vendor").join(package);
            if vendor.is_dir() {
                library_path.push(":");
                library_path.push(vendor);
            }
        }
    }
    if !inherited.is_empty() {
        library_path.push(":");
        library_path.push(inherited);
    }
    command.env("LD_LIBRARY_PATH", library_path);
    command
}
pub fn server_binary(paths: &Paths) -> Option<PathBuf> {
    let cfg = Config::load(paths).ok();
    if let Some(runtime) = cfg.as_ref().map(|config| config.runtime.as_str())
        && runtime != "managed"
    {
        if let Some(name) = runtime.strip_prefix("managed:") {
            let path = paths.versions.join(name).join("llama-server");
            if path.is_file() {
                return Some(path);
            }
        }
        if let Some(name) = runtime.strip_prefix("lmstudio:") {
            let home = directories::BaseDirs::new()?.home_dir().to_owned();
            let path = home
                .join(".lmstudio/extensions/backends")
                .join(name)
                .join("llama-server");
            if path.is_file() {
                return Some(path);
            }
        }
    }
    let path = paths.current.join("llama-server");
    path.is_file().then_some(path)
}
pub fn server_help(paths: &Paths) -> Result<String> {
    let binary = server_binary(paths).context("llama-server is not installed")?;
    let output = runtime_command(&binary, paths).arg("--help").output()?;
    Ok(format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}
static FLAG_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"--([A-Za-z0-9][A-Za-z0-9-]*)").unwrap());
static VRAM_DEVICE_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\((\d+)\s+MiB,\s+(\d+)\s+MiB free\)").unwrap());

pub fn known_flags(paths: &Paths) -> Option<BTreeSet<String>> {
    let text = server_help(paths).ok()?;
    Some(
        FLAG_RE
            .captures_iter(&text)
            .map(|capture| capture[1].to_owned())
            .collect(),
    )
}

fn flag_value(args: &[String], flags: &[&str]) -> Option<String> {
    args.windows(2)
        .rev()
        .find(|pair| flags.contains(&pair[0].as_str()))
        .map(|pair| pair[1].clone())
}

pub fn validate_draft_model(main: Option<&Path>, args: &[String]) -> Result<()> {
    let Some(main) = main else { return Ok(()) };
    let Some(raw) = flag_value(args, &["--spec-draft-model", "--model-draft", "-md"]) else {
        return Ok(());
    };
    let draft = PathBuf::from(raw);
    if !draft.is_file() {
        bail!("draft model '{}' does not exist", draft.display())
    }
    match models::draft_compatibility(main, &draft) {
        models::DraftCompatibility::Compatible => Ok(()),
        models::DraftCompatibility::Incompatible(reason) => bail!(
            "draft model '{}' is incompatible with '{}': {reason}",
            draft.display(),
            main.display()
        ),
        models::DraftCompatibility::Unknown(_) => Ok(()),
    }
}

pub fn common_args(cfg: &Config) -> Vec<String> {
    let ngl = cfg
        .gpu_layers
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| cfg.gpu_layers.to_string());
    let mut out = vec!["-ngl".into(), ngl, "-c".into(), cfg.ctx_size.to_string()];
    out.extend(cfg.extra_args.clone());
    out
}
pub fn build_command(
    cfg: &Config,
    paths: &Paths,
    profiles: &Profiles,
    model: Option<&str>,
    extra: &[String],
) -> Result<(PathBuf, Vec<String>, bool)> {
    if model.is_none() {
        if !paths.swap_bin.is_file() {
            bail!("llama-swap is not installed — run 'llamactl update'")
        }
        write_swap_config(cfg, paths, profiles)?;
        return Ok((
            paths.swap_bin.clone(),
            vec![
                "-config".into(),
                paths.swap_config.display().to_string(),
                "-watch-config".into(),
                "-listen".into(),
                format!("{}:{}", cfg.host, cfg.port),
            ],
            true,
        ));
    }
    let binary =
        server_binary(paths).context("llama-server is not installed — run 'llamactl update'")?;
    let spec = model.unwrap();
    let (query, profile_name) = spec
        .split_once('@')
        .map_or((spec, None), |(a, b)| (a, Some(b)));
    let profile_name =
        profile_name.or_else(|| profiles.profiles.contains_key(spec).then_some(spec));
    let query = profile_name
        .and_then(|p| profiles.owner(p))
        .unwrap_or(query);
    let (mut args, main_path, id) = models::resolve(cfg, Some(query))?;
    args.extend(common_args(cfg));
    if let Some(profile) = profile_name {
        args.extend(profiles.args(profile)?);
        args.extend(["--alias".into(), profile.into()]);
    } else {
        args.extend(["--alias".into(), id]);
    }
    args.extend([
        "--host".into(),
        cfg.host.clone(),
        "--port".into(),
        cfg.port.to_string(),
    ]);
    for key in cfg.keys() {
        args.extend(["--api-key".into(), key]);
    }
    args.extend(extra.to_owned());
    validate_draft_model(main_path.as_deref(), &args)?;
    if let Some(profile) = profile_name {
        let environment = profiles.environment(profile)?;
        if !environment.is_empty() {
            let mut wrapped = environment
                .into_iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>();
            wrapped.push(binary.display().to_string());
            wrapped.extend(args);
            return Ok((PathBuf::from("/usr/bin/env"), wrapped, false));
        }
    }
    Ok((binary, args, false))
}

pub fn serve(
    cfg: &Config,
    paths: &Paths,
    profiles: &Profiles,
    model: Option<&str>,
    extra: &[String],
) -> Result<()> {
    let (binary, args, _) = build_command(cfg, paths, profiles, model, extra)?;
    let status = runtime_command(&binary, paths)
        .args(&args)
        .status()
        .with_context(|| format!("run {}", binary.display()))?;
    if !status.success() {
        bail!("server exited with {status}")
    }
    Ok(())
}
pub fn start(
    cfg: &Config,
    paths: &Paths,
    profiles: &Profiles,
    model: Option<&str>,
    extra: &[String],
) -> Result<u32> {
    if let Some(existing) = pid(paths) {
        bail!("server already running (pid {existing})")
    }
    fs::create_dir_all(&paths.state_dir)?;
    let (binary, args, swap) = build_command(cfg, paths, profiles, model, extra)?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log)?;
    let err = log.try_clone()?;
    let mut child = runtime_command(&binary, paths)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(err)
        .process_group(0)
        .spawn()
        .with_context(|| format!("start {}", binary.display()))?;

    thread::sleep(Duration::from_millis(250));
    if let Some(status) = child.try_wait()? {
        let _ = fs::remove_file(&paths.pid);
        bail!(
            "server exited during startup with {status}; see {}",
            paths.log.display()
        )
    }
    fs::write(&paths.pid, format!("{}\n", child.id()))?;
    crate::config::atomic_json(
        &paths.launch,
        &LaunchSpec {
            model: model.map(str::to_owned),
            extra: extra.to_owned(),
            swap,
        },
    )?;
    Ok(child.id())
}
pub fn stop(paths: &Paths) -> Result<bool> {
    let Some(id) = pid(paths) else {
        let _ = fs::remove_file(&paths.pid);
        return Ok(false);
    };
    Command::new("kill")
        .args(["-TERM", &format!("-{id}")])
        .status()
        .context("send SIGTERM")?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if !Path::new(&format!("/proc/{id}")).exists() {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = fs::remove_file(&paths.pid);
    Ok(true)
}
pub fn restart(cfg: &Config, paths: &Paths, profiles: &Profiles) -> Result<u32> {
    let spec = fs::read_to_string(&paths.launch)
        .ok()
        .and_then(|s| serde_json::from_str::<LaunchSpec>(&s).ok())
        .unwrap_or(LaunchSpec {
            model: None,
            extra: vec![],
            swap: true,
        });
    stop(paths)?;
    start(cfg, paths, profiles, spec.model.as_deref(), &spec.extra)
}

#[derive(Clone)]
struct SwapEntry {
    id: String,
    physical: String,
    path: PathBuf,
    profile: Option<String>,
    overrides: serde_json::Map<String, serde_json::Value>,
    estimate: u64,
    environment: Vec<(String, String)>,
}

pub fn write_swap_config(cfg: &Config, paths: &Paths, profiles: &Profiles) -> Result<usize> {
    let binary = server_binary(paths).context("llama-server is not installed")?;
    let known_flags = known_flags(paths);
    fs::create_dir_all(&paths.state_dir)?;
    let mut entries = vec![];
    for model in models::scan(cfg) {
        if cfg.advertise_base_models {
            let binding = profiles.binding(&model.id, &model.relative)?;
            let (profile, overrides) = binding
                .map(|(profile, overrides)| (Some(profile), overrides))
                .unwrap_or_default();
            let mut estimate_args = common_args(cfg);
            if let Some(profile) = &profile {
                let mut resolved = profiles.clone();
                if let Some(target) = resolved.profiles.get_mut(profile) {
                    for (key, value) in &overrides {
                        target.insert(key.clone(), value.clone());
                    }
                }
                estimate_args.extend(resolved.args_checked(profile, known_flags.as_ref())?);
            }
            let environment = profile
                .as_deref()
                .map(|profile| profiles.environment(profile))
                .transpose()?
                .unwrap_or_default();
            entries.push(SwapEntry {
                id: model.id.clone(),
                physical: model.id.clone(),
                path: model.path.clone(),
                profile,
                overrides,
                estimate: models::estimate_vram(&model.path, &estimate_args),
                environment,
            });
        }
        if cfg.advertise_profiles {
            for name in profiles
                .profiles
                .keys()
                .filter(|n| profiles.owner(n) == Some(model.id.as_str()))
            {
                if !profiles
                    .profiles
                    .get(name)
                    .and_then(|profile| profile.get("_hidden"))
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    entries.push(SwapEntry {
                        id: name.clone(),
                        physical: model.id.clone(),
                        path: model.path.clone(),
                        profile: Some(name.clone()),
                        overrides: serde_json::Map::new(),
                        estimate: {
                            let mut args = common_args(cfg);
                            args.extend(profiles.args_checked(name, known_flags.as_ref())?);
                            models::estimate_vram(&model.path, &args)
                        },
                        environment: profiles.environment(name)?,
                    });
                }
            }
        }
    }

    let mut unique = BTreeMap::<String, SwapEntry>::new();
    for entry in entries {
        match unique.get(&entry.id) {
            Some(existing) if existing.profile.is_some() && entry.profile.is_none() => {}
            _ => {
                unique.insert(entry.id.clone(), entry);
            }
        }
    }
    let mut entries = unique.into_values().collect::<Vec<_>>();
    if cfg.advertise_profiles {
        let qualified = entries
            .iter()
            .filter(|entry| entry.profile.is_some() && entry.id != entry.physical)
            .map(|entry| {
                let mut qualified = entry.clone();
                qualified.id = format!("{}@{}", entry.physical, entry.id);
                qualified
            })
            .collect::<Vec<_>>();
        entries.extend(qualified);
    }
    entries.sort_by(|left, right| left.id.cmp(&right.id));
    if entries.is_empty() {
        bail!("model advertisement is empty; enable advertise_base_models or advertise_profiles")
    }
    let known = entries
        .iter()
        .map(|e| e.id.as_str())
        .collect::<BTreeSet<_>>();
    for pin in &cfg.scheduler_pinned_models {
        if !known.contains(pin.as_str()) {
            bail!("unknown pinned model '{pin}'")
        }
    }
    let mut text = format!("healthCheckTimeout: 300\nstartPort: {}\n", cfg.port + 1);
    if !cfg.keys().is_empty() {
        text.push_str("apiKeys:\n");
        for key in cfg.keys() {
            text.push_str(&format!("  - {}\n", serde_json::to_string(&key)?));
        }
    }
    text.push_str("models:\n");
    for entry in &entries {
        let mut args = vec![binary.display().to_string()];
        args.extend(models::model_args(&entry.path));
        args.extend(common_args(cfg));
        if let Some(profile) = &entry.profile {
            let mut resolved = profiles.clone();
            if let Some(target) = resolved.profiles.get_mut(profile) {
                for (key, value) in &entry.overrides {
                    target.insert(key.clone(), value.clone());
                }
            }
            args.extend(resolved.args_checked(profile, known_flags.as_ref())?);
            let external_draft = args
                .iter()
                .any(|arg| arg == "--spec-draft-model" || arg == "--model-draft");
            if args.iter().any(|arg| arg == "draft-mtp")
                && !external_draft
                && !models::has_mtp(&entry.path)
            {
                bail!(
                    "profile '{profile}' requests internal MTP but its GGUF lacks compatible NextN tensors"
                )
            }
        }
        validate_draft_model(Some(&entry.path), &args)
            .with_context(|| format!("model '{}'", entry.id))?;
        if !args
            .iter()
            .any(|arg| arg == "--embeddings" || arg == "--reranking")
        {
            match models::serving_mode(&entry.path) {
                Some("rerank") => args.push("--reranking".into()),
                Some("embed") => args.push("--embeddings".into()),
                _ => {}
            }
        }
        args.extend([
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            "${PORT}".into(),
            "--alias".into(),
            entry.id.clone(),
        ]);
        let ttl = if cfg.scheduler_pinned_models.contains(&entry.id) {
            0
        } else {
            cfg.swap_ttl
        };
        if !entry.environment.is_empty() {
            args.splice(
                0..0,
                std::iter::once("env".into()).chain(
                    entry
                        .environment
                        .iter()
                        .map(|(key, value)| format!("{key}={value}")),
                ),
            );
        }
        text.push_str(&format!(
            "  {}:\n    cmd: {}\n    ttl: {ttl}\n",
            serde_json::to_string(&entry.id)?,
            shell_join(&args),
        ));
    }
    let installed_vram = if cfg.scheduler_enabled || !cfg.scheduler_pinned_models.is_empty() {
        installed_vram_bytes()
    } else {
        0
    };
    if cfg.scheduler_enabled {
        append_scheduler_matrix(&mut text, cfg, &entries, installed_vram)?;
    }
    if !cfg.scheduler_pinned_models.is_empty() {
        let pinned = cfg
            .scheduler_pinned_models
            .iter()
            .filter_map(|pin| entries.iter().find(|entry| &entry.id == pin))
            .collect::<Vec<_>>();
        let unique = pinned
            .iter()
            .map(|entry| entry.physical.as_str())
            .collect::<BTreeSet<_>>();
        if unique.len() != pinned.len() {
            bail!("pinned models cannot contain two profiles of the same physical GGUF")
        }
        let budget = (installed_vram as f64 * cfg.scheduler_vram_fraction.clamp(0.1, 1.0)) as u64;
        let needed = pinned.iter().map(|entry| entry.estimate).sum::<u64>();
        if pinned.len() > cfg.scheduler_max_models || budget > 0 && needed > budget {
            bail!(
                "pinned models need {:.1} G but scheduler budget is {:.1} G",
                needed as f64 / (1u64 << 30) as f64,
                budget as f64 / (1u64 << 30) as f64
            )
        }
        text.push_str("hooks:\n  on_startup:\n    preload:\n");
        for pin in &cfg.scheduler_pinned_models {
            text.push_str(&format!("      - {}\n", serde_json::to_string(pin)?));
        }
    }
    let mut file = tempfile::NamedTempFile::new_in(&paths.state_dir)?;
    file.write_all(text.as_bytes())?;
    file.persist(&paths.swap_config).map_err(|e| e.error)?;
    Ok(entries.len())
}

fn maximal_fitting_sets(estimates: &[u64], budget: u64, max_size: usize) -> Vec<Vec<usize>> {
    fn visit(
        estimates: &[u64],
        budget: u64,
        max_size: usize,
        start: usize,
        total: u64,
        selected: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if selected.len() >= 2 {
            let can_extend = selected.len() < max_size
                && estimates.iter().enumerate().any(|(index, estimate)| {
                    !selected.contains(&index) && total.saturating_add(*estimate) <= budget
                });
            if !can_extend {
                output.push(selected.clone());
            }
        }
        if selected.len() == max_size {
            return;
        }
        for index in start..estimates.len() {
            let next = total.saturating_add(estimates[index]);
            if next > budget {
                continue;
            }
            selected.push(index);
            visit(
                estimates,
                budget,
                max_size,
                index + 1,
                next,
                selected,
                output,
            );
            selected.pop();
        }
    }

    if max_size < 2 {
        return vec![];
    }
    let mut output = vec![];
    visit(
        estimates,
        budget,
        max_size.min(estimates.len()),
        0,
        0,
        &mut vec![],
        &mut output,
    );
    output
}

fn append_scheduler_matrix(
    text: &mut String,
    cfg: &Config,
    entries: &[SwapEntry],
    installed: u64,
) -> Result<()> {
    if installed == 0 {
        return Ok(());
    }
    let budget = (installed as f64 * cfg.scheduler_vram_fraction.clamp(0.1, 1.0)) as u64;
    let mut physical = BTreeMap::<String, (u64, Vec<&SwapEntry>)>::new();
    for entry in entries {
        let group = physical
            .entry(entry.physical.clone())
            .or_insert((entry.estimate, vec![]));
        group.0 = group.0.max(entry.estimate);
        group.1.push(entry);
    }
    let groups = physical.into_iter().collect::<Vec<_>>();
    let estimates = groups
        .iter()
        .map(|(_, (estimate, _))| *estimate)
        .collect::<Vec<_>>();
    let maximal = maximal_fitting_sets(&estimates, budget, cfg.scheduler_max_models);
    if maximal.is_empty() {
        return Ok(());
    }
    let mut participating = vec![false; groups.len()];
    for index in maximal.iter().flatten() {
        participating[*index] = true;
    }
    let matrix_entries = groups
        .iter()
        .enumerate()
        .filter(|(index, _)| participating[*index])
        .flat_map(|(_, (_, (_, variants)))| variants.iter().copied())
        .collect::<Vec<_>>();
    let vars = matrix_entries
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.id.as_str(), format!("m{index}")))
        .collect::<BTreeMap<_, _>>();
    text.push_str("matrix:\n  vars:\n");
    for entry in &matrix_entries {
        text.push_str(&format!(
            "    {}: {}\n",
            vars[entry.id.as_str()],
            serde_json::to_string(&entry.id)?
        ));
    }
    text.push_str("  evict_costs:\n");
    for entry in &matrix_entries {
        text.push_str(&format!(
            "    {}: {}\n",
            vars[entry.id.as_str()],
            (entry.estimate / 1_000_000_000).max(1)
        ));
    }
    text.push_str("  sets:\n");
    for (set_index, set) in maximal.iter().enumerate() {
        let expression = set
            .iter()
            .map(|index| {
                let variants = &groups[*index].1.1;
                let names = variants
                    .iter()
                    .filter_map(|entry| vars.get(entry.id.as_str()))
                    .cloned()
                    .collect::<Vec<_>>();
                if names.len() == 1 {
                    names[0].clone()
                } else {
                    format!("({})", names.join(" | "))
                }
            })
            .collect::<Vec<_>>()
            .join(" & ");
        text.push_str(&format!(
            "    fit{set_index}: {}\n",
            serde_json::to_string(&expression)?
        ));
    }
    Ok(())
}

/// Total and free device memory, as reported by the runtime itself.
///
/// `llama-server --list-devices` prints `(TOTAL MiB, FREE MiB free)` from
/// llama.cpp's common code, so the format is identical across CUDA, ROCm,
/// Vulkan and SYCL. Both numbers are summed across devices.
///
/// This is the only usage source that works with *nothing loaded*: DRM fdinfo
/// can only report clients that exist, so before the first model starts it
/// legitimately reads zero — which is indistinguishable from "the card is
/// empty" even when another process holds memory we cannot see.
///
/// ⚠️ `free` is backend-dependent and not directly comparable to fdinfo. On the
/// same idle Arc B70 pair, SYCL reports the cards as entirely free while Vulkan
/// reports ~3.2 GiB already in use. Treat it as a fallback, not a cross-check.
pub fn device_memory_bytes() -> Option<(u64, u64)> {
    let paths = Paths::discover().ok()?;
    let binary = server_binary(&paths).unwrap_or_else(|| paths.current.join("llama-server"));
    let output = runtime_command(&binary, &paths)
        .arg("--list-devices")
        .output()
        .ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let mut total = 0u64;
    let mut free = 0u64;
    for capture in VRAM_DEVICE_RE.captures_iter(&text) {
        total += capture[1].parse::<u64>().ok()? * 1024 * 1024;
        free += capture[2].parse::<u64>().ok()? * 1024 * 1024;
    }
    (total > 0).then_some((total, free))
}

pub fn installed_vram_bytes() -> u64 {
    if let Ok(paths) = Paths::discover()
        && let Some(binary) = server_binary(&paths)
        && let Ok(output) = runtime_command(&binary, &paths)
            .arg("--list-devices")
            .output()
    {
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let total = VRAM_DEVICE_RE
            .captures_iter(&text)
            .filter_map(|capture| capture[1].parse::<u64>().ok())
            .sum::<u64>();
        if total > 0 {
            return total * 1024 * 1024;
        }
    }
    Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter_map(|line| line.trim().parse::<u64>().ok())
                .sum::<u64>()
                * 1024
                * 1024
        })
        .unwrap_or(0)
}
fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|a| {
            if a == "${PORT}"
                || a.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "/._-:=,".contains(c))
            {
                a.clone()
            } else {
                format!("'{}'", a.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn swap_mode(paths: &Paths) -> bool {
    fs::read_to_string(&paths.launch)
        .ok()
        .and_then(|text| serde_json::from_str::<LaunchSpec>(&text).ok())
        .is_some_and(|spec| spec.swap)
}

pub fn wait_swap_ready(cfg: &Config, timeout: Duration) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;
    let deadline = Instant::now() + timeout;
    loop {
        let mut request = client.get(format!("http://{}:{}/running", cfg.host, cfg.port));
        if let Some(key) = cfg.keys().first() {
            request = request.bearer_auth(key);
        }
        if request
            .send()
            .is_ok_and(|response| response.status().is_success())
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!(
                "scheduler did not become ready within {}s",
                timeout.as_secs()
            )
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn swap_entries(cfg: &Config) -> Result<Vec<String>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let mut request = client.get(format!("http://{}:{}/v1/models", cfg.host, cfg.port));
    if let Some(key) = cfg.keys().first() {
        request = request.bearer_auth(key);
    }
    let response = request.send()?;
    if !response.status().is_success() {
        bail!(
            "could not list scheduled models: HTTP {}",
            response.status()
        )
    }
    let payload = response.json::<serde_json::Value>()?;
    Ok(payload
        .get("data")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("id").and_then(serde_json::Value::as_str))
        .map(str::to_owned)
        .collect())
}

fn swap_running_models(cfg: &Config) -> Result<Vec<String>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let mut request = client.get(format!("http://{}:{}/running", cfg.host, cfg.port));
    if let Some(key) = cfg.keys().first() {
        request = request.bearer_auth(key);
    }
    let payload = request
        .send()?
        .error_for_status()?
        .json::<serde_json::Value>()?;
    Ok(payload
        .get("running")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("model").and_then(serde_json::Value::as_str))
        .map(str::to_owned)
        .collect())
}

fn resolve_swap_entry(cfg: &Config, profiles: &Profiles, requested: &str) -> Result<String> {
    let names = swap_entries(cfg)?;
    if names.iter().any(|name| name == requested) {
        return Ok(requested.to_owned());
    }

    if let Some((profile, _)) = profiles.binding(requested, requested)?
        && names.iter().any(|name| name == &profile)
    {
        return Ok(profile);
    }
    let lower = requested.to_lowercase();
    let hits = names
        .iter()
        .filter(|name| name.to_lowercase().contains(&lower))
        .cloned()
        .collect::<Vec<_>>();
    match hits.len() {
        0 => bail!("no scheduled model matching '{requested}' — see 'llamactl models'"),
        1 => Ok(hits[0].clone()),
        _ => bail!(
            "'{requested}' is ambiguous, matches:\n{}",
            hits.iter()
                .map(|name| format!("  {name}"))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    }
}

pub fn swap_load(cfg: &Config, profiles: &Profiles, model: &str) -> Result<()> {
    let model = resolve_swap_entry(cfg, profiles, model)?;
    if swap_running_models(cfg)?.iter().any(|name| name == &model) {
        return Ok(());
    }
    let encoded = percent_encode(&model);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;
    let mut request = client
        .get(format!(
            "http://{}:{}/upstream/{encoded}/",
            cfg.host, cfg.port
        ))
        .header("Accept", "text/html");
    if let Some(key) = cfg.keys().first() {
        request = request.bearer_auth(key);
    }
    let response = request.send()?;
    if !response.status().is_success()
        && !swap_running_models(cfg)?.iter().any(|name| name == &model)
    {
        bail!("could not load '{model}': HTTP {}", response.status())
    }
    Ok(())
}

pub fn swap_unload(cfg: &Config, profiles: &Profiles, model: Option<&str>) -> Result<()> {
    let resolved = model
        .map(|model| resolve_swap_entry(cfg, profiles, model))
        .transpose()?;
    let route = resolved
        .as_deref()
        .map(|model| format!("/api/models/unload/{}", percent_encode(model)))
        .unwrap_or_else(|| "/api/models/unload".into());
    let client = reqwest::blocking::Client::new();
    let mut request = client.post(format!("http://{}:{}{route}", cfg.host, cfg.port));
    if let Some(key) = cfg.keys().first() {
        request = request.bearer_auth(key);
    }
    request.send()?.error_for_status()?;
    Ok(())
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
                (byte as char).to_string()
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
}

pub fn upstream_log(cfg: &Config, max_lines: usize) -> String {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(client) => client,
        Err(_) => return String::new(),
    };
    let mut request = client
        .get(format!("http://{}:{}/api/events", cfg.host, cfg.port))
        .header("Accept", "text/event-stream");
    if let Some(key) = cfg.keys().first() {
        request = request.bearer_auth(key);
    }
    let Ok(mut response) = request.send() else {
        return String::new();
    };
    if !response.status().is_success() {
        return String::new();
    }
    let mut text = String::new();
    let mut chunk = [0u8; 64 * 1024];
    while text.len() < 512 * 1024 {
        match response.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(read) => text.push_str(&String::from_utf8_lossy(&chunk[..read])),
        }
        if complete_event_contains_upstream(&text) {
            break;
        }
    }
    upstream_log_from_events(&text, max_lines)
}

fn complete_event_contains_upstream(events: &str) -> bool {
    events
        .split_inclusive("\n\n")
        .any(|event| event.contains("\\\"source\\\":\\\"upstream\\\""))
}

fn upstream_log_from_events(events: &str, max_lines: usize) -> String {
    let mut upstream = String::new();
    for line in events.lines().filter_map(|line| line.strip_prefix("data:")) {
        let Ok(event) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
            continue;
        };
        if event.get("type").and_then(serde_json::Value::as_str) != Some("logData") {
            continue;
        }
        let Some(payload) = event.get("data").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if payload.get("source").and_then(serde_json::Value::as_str) == Some("upstream")
            && let Some(data) = payload.get("data").and_then(serde_json::Value::as_str)
        {
            upstream.push_str(data);
        }
    }
    let lines = upstream.lines().collect::<Vec<_>>();
    lines[lines.len().saturating_sub(max_lines)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::{maximal_fitting_sets, upstream_log_from_events};
    use crate::config::Config;

    #[test]
    fn advertisement_defaults_to_assigned_base_models() {
        let cfg = Config::default();
        assert!(cfg.advertise_base_models);
        assert!(!cfg.advertise_profiles);
    }

    #[test]
    fn event_log_keeps_only_upstream_output_and_limits_lines() {
        let events = concat!(
            "data:{\"type\":\"logData\",\"data\":\"{\\\"source\\\":\\\"proxy\\\",\\\"data\\\":\\\"proxy\\\\n\\\"}\"}\n",
            "data:{\"type\":\"logData\",\"data\":\"{\\\"source\\\":\\\"upstream\\\",\\\"data\\\":\\\"one\\\\ntwo\\\\nthree\\\\n\\\"}\"}\n"
        );
        assert_eq!(upstream_log_from_events(events, 2), "two\nthree");
    }

    #[test]
    fn scheduler_returns_only_maximal_sets() {
        let sets = maximal_fitting_sets(&[12, 15, 20], 30, 4);
        assert_eq!(sets, vec![vec![0, 1]]);
    }

    #[test]
    fn scheduler_supports_more_than_one_machine_word_of_models() {
        let estimates = vec![10; usize::BITS as usize + 1];
        let sets = maximal_fitting_sets(&estimates, 20, 2);
        assert_eq!(sets.len(), estimates.len() * (estimates.len() - 1) / 2);
        assert!(sets.iter().all(|set| set.len() == 2));
    }
}

#[cfg(test)]
mod capacity_live {
    /// Manual check — runs the real runtime binary, so #[ignore]d.
    /// `cargo test --release -- --ignored --nocapture capacity_live`
    #[test]
    #[ignore]
    fn reports_installed_vram() {
        let bytes = super::installed_vram_bytes();
        println!("device_memory_bytes: {:?}", super::device_memory_bytes().map(|(t,f)| (t >> 30, f >> 30)));
        println!(
            "installed_vram_bytes: {} bytes = {:.2} GiB",
            bytes,
            bytes as f64 / (1u64 << 30) as f64
        );
    }
}
