mod benchmark;
mod config;
mod drm;
mod gguf;
mod huggingface;
mod models;
mod process;
mod profiles;
mod templates;
mod ui;
mod update;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use config::{Config, Paths};
use profiles::Profiles;
use std::{
    fs,
    io::{IsTerminal, Read},
    path::PathBuf,
    process::Command,
    sync::{Arc, LazyLock, atomic::{AtomicBool, Ordering}},
};

#[derive(Parser)]
#[command(
    name = "llamactl",
    version,
    about = "Native llama.cpp runtime manager",
    arg_required_else_help = false
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}
#[derive(Subcommand)]
enum Commands {
    Ui,

    Update {
        #[arg(long)]
        check: bool,
        #[arg(long)]
        restart: bool,
    },

    Build {
        backend: Option<String>,
        #[arg(long)]
        restart: bool,
    },

    Start {
        model: Option<String>,
        #[arg(last = true)]
        extra: Vec<String>,
    },

    Serve {
        model: Option<String>,
        #[arg(last = true)]
        extra: Vec<String>,
    },

    Load {
        model: String,
        #[arg(long)]
        pin: bool,
    },

    Rm {
        model: String,
        #[arg(short, long)]
        yes: bool,
    },

    Graft {
        target: String,
        donor: String,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        profile: Option<String>,
    },
    Stop,
    Restart,
    Unload {
        model: Option<String>,
    },
    Status,
    Reload,
    Models,
    Builds,

    Profiles {
        #[command(subcommand)]
        action: Option<ProfileAction>,
    },

    Fit {
        model: Option<String>,
        #[arg(long)]
        contexts: bool,
        #[arg(long)]
        compact: bool,
        #[arg(last = true)]
        extra: Vec<String>,
    },

    Flags {
        pattern: Option<String>,
    },

    Keys {
        action: Option<String>,
        value: Option<String>,
    },

    Scheduler {
        action: Option<String>,
        model: Option<String>,
    },
    InstallService,

    Config {
        key: Option<String>,
        value: Option<String>,
    },

    Search {
        query: Option<String>,
        #[arg(long)]
        templates: bool,
    },

    Repo {
        id: String,
    },

    Card {
        id: String,
    },

    Download {
        id: String,
        quant: Option<String>,
    },

    TemplateFetch {
        id: String,
        #[arg(long)]
        save: bool,
    },

    Logs {
        #[arg(long, default_value_t = 300)]
        lines: usize,
    },

    Templates {
        #[command(subcommand)]
        action: Option<TemplateAction>,
    },
}
#[derive(Subcommand)]
enum ProfileAction {
    Clone { source: String, new_name: String },
    Rename { source: String, new_name: String },
    Delete { name: String },
    Benchmark {
        name: String,
        #[arg(long, value_name = "CASE")]
        max_case: Option<String>,
    },
    Benchmarks { name: String },
    Create {
        name: String,
        #[arg(long)]
        model: String,
    },
    Load { name: String },
    Bind { name: String },
    Set {
        name: String,
        key: String,
        value: String,
    },
    SetModel {
        name: String,
        model: String,
    },
}
#[derive(Subcommand)]
enum TemplateAction {
    List,
    Show { name: String },
    Add {
        name: String,
        #[arg(long)]
        file: Option<PathBuf>,
    },
    Edit {
        name: String,
        #[arg(long)]
        file: Option<PathBuf>,
    },
    Rename { old: String, new: String },
    Delete { name: String },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("✗ {e:#}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let cli = Cli::parse();
    let paths = Paths::discover()?;
    models::init_metadata_cache(&paths);
    let mut cfg = Config::load(&paths)?;
    let launch_command = matches!(
        cli.command,
        None | Some(Commands::Ui | Commands::Start { .. } | Commands::Serve { .. })
    );
    if launch_command && process::server_binary(&paths).is_none() {
        update::install(&cfg, &paths)?;
    }
    match cli.command {
        None | Some(Commands::Ui) => {
            if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
                bail!(
                    "the interactive UI needs a terminal\nrun 'llamactl --help' for non-interactive commands"
                )
            }
            ui::run(cfg, &paths)?
        }
        Some(Commands::Update { check, restart }) => {
            if check {
                let (l, lc, s, sc) = update::check(&paths)?;
                println!(
                    "llama.cpp {l}: {}",
                    if lc { "update available" } else { "up to date" }
                );
                println!(
                    "llama-swap {s}: {}",
                    if sc { "update available" } else { "up to date" }
                );
            } else {
                let running = process::pid(&paths).is_some();
                update::install(&cfg, &paths)?;
                println!("✓ updates installed");
                if restart && running {
                    let p = Profiles::load(&paths)?;
                    println!("✓ restarted - pid {}", process::restart(&cfg, &paths, &p)?);
                }
            }
        }
        Some(Commands::Build { backend, restart }) => {
            let backend = backend.unwrap_or_else(|| cfg.backend.clone());
            if !config::BACKENDS.contains(&backend.as_str()) {
                bail!("invalid backend '{backend}'")
            }
            let was_running = process::pid(&paths).is_some();
            update::build_source(&cfg, &paths, &backend)?;
            cfg.backend = backend;
            cfg.runtime = "managed".into();
            cfg.save(&paths)?;
            println!("✓ source build installed");
            if restart && was_running {
                let profiles = Profiles::load(&paths)?;
                println!(
                    "✓ restarted - pid {}",
                    process::restart(&cfg, &paths, &profiles)?
                );
            }
        }
        Some(Commands::Start { model, extra }) => {
            let p = Profiles::load(&paths)?;
            println!(
                "✓ server started - pid {}",
                process::start(&cfg, &paths, &p, model.as_deref(), &extra)?
            )
        }
        Some(Commands::Serve { model, extra }) => {
            let p = Profiles::load(&paths)?;
            process::serve(&cfg, &paths, &p, model.as_deref(), &extra)?
        }
        Some(Commands::Stop) => println!(
            "{}",
            if process::stop(&paths)? {
                "✓ server stopped"
            } else {
                "- server is not running"
            }
        ),
        Some(Commands::Restart) => {
            let p = Profiles::load(&paths)?;
            println!(
                "✓ server restarted - pid {}",
                process::restart(&cfg, &paths, &p)?
            )
        }
        Some(Commands::Status) => status(&cfg, &paths),
        Some(Commands::Models) => list_models(&cfg),
        Some(Commands::Builds) => list_builds(&paths)?,
        Some(Commands::Reload) => {
            let p = Profiles::load(&paths)?;
            let n = process::write_swap_config(&cfg, &paths, &p)?;
            println!("✓ generated {} swap entries", n)
        }
        Some(Commands::Profiles { action }) => profile_command(&cfg, &paths, action)?,
        Some(Commands::Rm { model, yes }) => {
            if !yes {
                bail!("refusing to delete without --yes")
            };
            ensure_model_not_loaded(&cfg, &paths, &model)?;
            let files = models::delete(&cfg, &model)?;
            let mut profiles = Profiles::load(&paths)?;
            let removed = profiles
                .profiles
                .iter()
                .filter(|(_, profile)| {
                    profile.get("_model").and_then(serde_json::Value::as_str) == Some(&model)
                })
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>();
            for profile in &removed {
                profiles.remove(profile)?;
            }
            profiles.models.remove(&model);
            profiles.save(&paths)?;
            refresh_swap(&cfg, &paths)?;
            println!(
                "✓ deleted {} file(s) and {} profile(s) for {model}",
                files.len(),
                removed.len()
            )
        }
        Some(Commands::Graft {
            target,
            donor,
            output,
            profile,
        }) => {
            let (_, target_path, _) = models::resolve(&cfg, Some(&target))?;
            let target_path = target_path.context("target is not a local model")?;
            let (_, donor_path, _) = models::resolve(&cfg, Some(&donor))?;
            let donor_path = donor_path.context("donor is not a local model")?;
            let output = output.unwrap_or_else(|| {
                let stem = target_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy();
                let mut out = target_path.clone();
                out.set_file_name(format!("{stem}-MTP.gguf"));
                out
            });
            println!(
                "grafting MTP from {} into {}…",
                donor_path.display(),
                target_path.display()
            );
            let mut last_tenth = u64::MAX;
            let report = gguf::graft_mtp(&target_path, &donor_path, &output, |copied, total| {
                if total > 0 {
                    let tenth = copied * 10 / total;
                    if tenth != last_tenth {
                        eprintln!("  {}%", tenth * 10);
                        last_tenth = tenth;
                    }
                }
            })?;
            println!(
                "✓ grafted {} MTP tensor(s) into {} ({} tensors, {} blocks, nextn {}, {:.2} GiB)",
                report.grafted_tensors,
                output.display(),
                report.total_tensors,
                report.block_count,
                report.nextn_layers,
                report.output_bytes as f64 / (1u64 << 30) as f64
            );
            if let Some(name) = profile {
                let mut profiles = Profiles::load(&paths)?;
                let query = output.to_string_lossy().into_owned();
                let (id, _, removed_mtp) = retarget_profile(&cfg, &mut profiles, &name, &query)?;
                profiles.save(&paths)?;
                refresh_swap(&cfg, &paths)?;
                println!("✓ profile {name} now uses model {id}");
                if removed_mtp {
                    println!("⚠ removed draft-mtp speculation - model '{id}' lacks MTP tensors");
                }
            }
        }
        Some(Commands::Config { key, value }) => match (key, value) {
            (None, None) => println!("{}", serde_json::to_string_pretty(&cfg)?),
            (Some(k), None) => {
                let v = serde_json::to_value(&cfg)?;
                println!(
                    "{}",
                    v.get(&k)
                        .with_context(|| format!("unknown config key '{k}'"))?
                )
            }
            (Some(k), Some(v)) => {
                cfg.set(&k, &v)?;
                cfg.save(&paths)?;
                refresh_swap(&cfg, &paths)?;
                println!("✓ set {k}")
            }
            _ => bail!("a value requires a key"),
        },
        Some(Commands::Search { query, templates }) => search_command(query.as_deref(), templates)?,
        Some(Commands::Repo { id }) => repo_command(&id)?,
        Some(Commands::Card { id }) => card_command(&id)?,
        Some(Commands::Download { id, quant }) => {
            download_command(&paths, &id, quant.as_deref())?
        }
        Some(Commands::TemplateFetch { id, save }) => template_fetch_command(&paths, &id, save)?,
        Some(Commands::Logs { lines }) => logs_command(&cfg, &paths, lines)?,
        Some(Commands::Templates { action }) => templates_command(&paths, action)?,
        Some(Commands::Keys { action, value }) => keys(&mut cfg, &paths, action.as_deref(), value)?,
        Some(Commands::Scheduler { action, model }) => {
            scheduler(&mut cfg, &paths, action.as_deref(), model)?
        }
        Some(Commands::Flags { pattern }) => flags(&paths, pattern.as_deref())?,
        Some(Commands::Fit {
            model,
            contexts,
            compact,
            extra,
        }) => fit(&cfg, &paths, model.as_deref(), contexts, compact, &extra)?,
        Some(Commands::Load { model, pin }) => {
            let profiles = Profiles::load(&paths)?;
            if pin && !cfg.scheduler_pinned_models.contains(&model) {
                cfg.scheduler_pinned_models.push(model.clone());
                cfg.save(&paths)?;
                refresh_swap(&cfg, &paths)?;
            }
            process::swap_load(&cfg, &profiles, &model)?;
            println!("✓ loaded - {model}")
        }
        Some(Commands::Unload { model }) => {
            let profiles = Profiles::load(&paths)?;
            process::swap_unload(&cfg, &profiles, model.as_deref())?;
            println!("✓ unload requested")
        }
        Some(Commands::InstallService) => {
            install_service(&paths)?;
            println!("✓ installed systemd user units");
            println!(
                "  enable with: systemctl --user enable --now llamactl.service llamactl-update.timer"
            );
        }
    }
    Ok(())
}
fn status(cfg: &Config, p: &Paths) {
    println!("llamactl {}", env!("CARGO_PKG_VERSION"));
    println!("  backend     {}", cfg.backend);
    println!("  endpoint    http://{}:{}/v1", cfg.host, cfg.port);
    println!(
        "  server      {}",
        process::pid(p)
            .map(|x| format!("running (pid {x})"))
            .unwrap_or("stopped".into())
    );
    println!("  models      {}", models::scan(cfg).len());
    println!(
        "  llama-swap  {}",
        if p.swap_bin.is_file() {
            "installed"
        } else {
            "not installed"
        }
    );
}
fn list_models(c: &Config) {
    println!("  {:<42} {:>8}  FEATURES  PATH", "MODEL", "SIZE");
    for m in models::scan(c) {
        println!(
            "  {:<42} {:>6.1}G  {:<8}  {}",
            m.id,
            m.bytes as f64 / (1u64 << 30) as f64,
            if m.vision { "vision" } else { "text" },
            m.relative
        )
    }
}
fn list_builds(p: &Paths) -> Result<()> {
    let cfg = Config::load(p)?;
    println!("  ACTIVE  RUNTIME");
    if p.versions.is_dir() {
        for e in fs::read_dir(&p.versions)? {
            let e = e?;
            if e.path().join("llama-server").is_file() {
                let active = fs::canonicalize(&p.current).ok() == fs::canonicalize(e.path()).ok();
                let id = format!("managed:{}", e.file_name().to_string_lossy());
                println!(
                    "  {:<6}  {}",
                    if active || cfg.runtime == id {
                        "yes"
                    } else {
                        ""
                    },
                    id
                )
            }
        }
    }
    if let Some(home) = directories::BaseDirs::new().map(|base| base.home_dir().to_owned()) {
        let backends = home.join(".lmstudio/extensions/backends");
        if backends.is_dir() {
            for entry in fs::read_dir(backends)?.filter_map(Result::ok) {
                let path = entry.path();
                if path.join("llama-server").is_file()
                    && path.join("backend-manifest.json").is_file()
                    && entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with("llama.cpp-")
                {
                    let id = format!("lmstudio:{}", entry.file_name().to_string_lossy());
                    println!(
                        "  {:<6}  {}",
                        if cfg.runtime == id { "yes" } else { "" },
                        id
                    );
                }
            }
        }
    }
    Ok(())
}
fn retarget_profile(
    c: &Config,
    profiles: &mut Profiles,
    name: &str,
    model: &str,
) -> Result<(String, Option<String>, bool)> {
    let (_, path, id) = models::resolve(c, Some(model))?;
    let path = path.with_context(|| format!("model '{model}' is not a local model file"))?;
    let (old_owner, removed_mtp) = {
        let profile = profiles
            .profiles
            .get_mut(name)
            .with_context(|| format!("unknown profile '{name}'"))?;
        let old = profile
            .get("_model")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        profile.insert("_model".into(), serde_json::Value::String(id.clone()));
        // Internal MTP speculation only works when the target GGUF carries
        // NextN tensors. Drop it rather than leave the profile unloadable.
        let mut removed_mtp = false;
        if profile.get("spec-type").and_then(serde_json::Value::as_str) == Some("draft-mtp")
            && !profile.contains_key("spec-draft-model")
            && !models::has_mtp(&path)
        {
            profile.remove("spec-type");
            profile.remove("spec-draft-n-max");
            removed_mtp = true;
        }
        (old, removed_mtp)
    };
    if let Some(old) = &old_owner
        && profiles.models.get(old).and_then(serde_json::Value::as_str) == Some(name)
    {
        profiles.models.remove(old);
    }
    profiles
        .models
        .insert(id.clone(), serde_json::Value::String(name.to_owned()));
    Ok((id, old_owner, removed_mtp))
}

fn profile_command(c: &Config, p: &Paths, a: Option<ProfileAction>) -> Result<()> {
    let mut profiles = Profiles::load(p)?;
    let mut refresh = false;
    match a {
        None => {
            println!("  {:<30} OWNER", "PROFILE");
            for name in profiles.profiles.keys() {
                println!(
                    "  {:<30} {}",
                    name,
                    profiles.owner(name).unwrap_or("unassigned")
                )
            }
        }
        Some(ProfileAction::Clone { source, new_name }) => {
            profiles.clone_profile(&source, &new_name)?;
            profiles.save(p)?;
            refresh = true;
        }
        Some(ProfileAction::Rename { source, new_name }) => {
            profiles.rename(&source, &new_name)?;
            profiles.save(p)?;
            let mut benchmarks = benchmark::ProfileBenchmarks::load(p)?;
            benchmarks.rename(&source, &new_name);
            benchmarks.save(p)?;
            refresh = true;
        }
        Some(ProfileAction::Delete { name }) => {
            profiles.remove(&name)?;
            profiles.save(p)?;
            let mut benchmarks = benchmark::ProfileBenchmarks::load(p)?;
            benchmarks.remove(&name);
            benchmarks.save(p)?;
            refresh = true;
        }
        Some(ProfileAction::Benchmark { name, max_case }) => {
            let max_cases = match max_case.as_deref() {
                None | Some("long") | Some("all") => benchmark::TOTAL_CASES,
                Some("small") => 1,
                Some("medium") => 5,
                Some(other) => bail!("unknown case '{other}' - use small, medium, long, or all"),
            };
            let cancelled = Arc::new(AtomicBool::new(false));
            let handler_cancelled = cancelled.clone();
            ctrlc::set_handler(move || {
                handler_cancelled.store(true, Ordering::Relaxed);
                eprintln!("cancelling - completing the current case and restoring the server…");
            })?;
            let run = benchmark::run_partial(c, p, &profiles, &name, cancelled, max_cases)?;
            println!("{}", benchmark::summary(&run));
        }
        Some(ProfileAction::Benchmarks { name }) => {
            let benchmarks = benchmark::ProfileBenchmarks::load(p)?;
            let runs = benchmarks
                .profiles
                .get(&name)
                .with_context(|| format!("profile '{name}' has no benchmark results"))?;
            for (index, run) in runs.iter().rev().enumerate() {
                if index > 0 {
                    println!();
                }
                println!("{}", benchmark::summary(run));
            }
        }
        Some(ProfileAction::Create { name, model }) => {
            profiles::valid_name(&name)?;
            if profiles.profiles.contains_key(&name) {
                bail!("profile '{name}' already exists")
            }
            let (_, path, id) = models::resolve(c, Some(&model))?;
            let context = path
                .as_deref()
                .and_then(models::context_limit)
                .unwrap_or(c.ctx_size)
                .min(32768);
            let mut profile = serde_json::Map::new();
            profile.insert("_model".into(), serde_json::Value::String(id.clone()));
            profile.insert("ctx-size".into(), serde_json::Value::from(context));
            profile.insert("parallel".into(), serde_json::Value::from(1));
            profile.insert("cache-type-k".into(), serde_json::Value::String("q8_0".into()));
            profile.insert("cache-type-v".into(), serde_json::Value::String("q8_0".into()));
            profile.insert("flash-attn".into(), serde_json::Value::String("on".into()));
            profile.insert("n-gpu-layers".into(), serde_json::Value::String("all".into()));
            profile.insert(
                "_extra_args".into(),
                serde_json::Value::Array(vec![serde_json::Value::String("--kv-unified".into())]),
            );
            profiles.profiles.insert(name.clone(), profile);
            profiles.models.insert(id, serde_json::Value::String(name.clone()));
            profiles.save(p)?;
            refresh = true;
            println!("✓ created and bound profile {name}");
        }
        Some(ProfileAction::Load { name }) => {
            if !profiles.profiles.contains_key(&name) {
                bail!("unknown profile '{name}'")
            }
            let swap_running = process::pid(p).is_some() && process::swap_mode(p);
            if !swap_running {
                if process::pid(p).is_some() {
                    process::stop(p)?;
                }
                process::start(c, p, &profiles, None, &[])?;
                process::wait_swap_ready(c, std::time::Duration::from_secs(10))?;
            }
            process::swap_load(c, &profiles, &name)?;
            println!("✓ loaded - {name}");
        }
        Some(ProfileAction::Bind { name }) => {
            let owner = profiles
                .owner(&name)
                .with_context(|| format!("profile '{name}' has no owner"))?
                .to_owned();
            profiles
                .models
                .insert(owner.clone(), serde_json::Value::String(name.clone()));
            profiles.save(p)?;
            refresh = true;
            println!("✓ bound {owner} → {name}");
        }
        Some(ProfileAction::Set { name, key, value }) => {
            let profile = profiles
                .profiles
                .get_mut(&name)
                .with_context(|| format!("unknown profile '{name}'"))?;
            let parsed: serde_json::Value = serde_json::from_str(&value)
                .unwrap_or_else(|_| serde_json::Value::String(value.clone()));
            profile.insert(key.clone(), parsed);
            profiles.save(p)?;
            refresh = true;
            println!("✓ set {key} on profile {name}");
        }
        Some(ProfileAction::SetModel { name, model }) => {
            let (id, old_owner, removed_mtp) = retarget_profile(c, &mut profiles, &name, &model)?;
            profiles.save(p)?;
            refresh = true;
            match old_owner {
                Some(old) if old != id => println!("✓ profile {name} now uses model {id} (was {old})"),
                Some(_) => println!("✓ profile {name} already uses model {id}"),
                None => println!("✓ profile {name} now uses model {id}"),
            }
            if removed_mtp {
                println!("⚠ removed draft-mtp speculation - model '{id}' lacks MTP tensors");
            }
        }
    }
    if refresh {
        refresh_swap(c, p)?;
    }
    Ok(())
}
fn keys(c: &mut Config, p: &Paths, a: Option<&str>, v: Option<String>) -> Result<()> {
    match a {
        None | Some("list") => {
            for (i, k) in c.keys().iter().enumerate() {
                println!(
                    "  {}  {}…{}",
                    i + 1,
                    &k[..k.len().min(5)],
                    if k.len() > 9 { &k[k.len() - 4..] } else { "" }
                )
            }
        }
        Some("add") => {
            let k = v.context("key required")?;
            if !c.api_keys.contains(&k) {
                c.api_keys.push(k);
                c.save(p)?
            }
        }
        Some("remove") => {
            let k = v.context("key required")?;
            c.api_keys.retain(|x| x != &k);
            if c.api_key == k {
                c.api_key.clear()
            }
            c.save(p)?
        }
        Some("generate") => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let k = format!("sk-llamactl-{now:x}-{:x}", std::process::id());
            c.api_keys.push(k.clone());
            c.save(p)?;
            println!("{k}")
        }
        Some(x) => bail!("unknown keys action '{x}'"),
    }
    refresh_swap(c, p)?;
    Ok(())
}
fn scheduler(c: &mut Config, p: &Paths, a: Option<&str>, m: Option<String>) -> Result<()> {
    match a {
        None => {
            println!(
                "enabled: {}\nVRAM fraction: {:.0}%\nmax models: {}\npinned: {}",
                c.scheduler_enabled,
                c.scheduler_vram_fraction * 100.0,
                c.scheduler_max_models,
                c.scheduler_pinned_models.join(", ")
            )
        }
        Some("enable") => {
            c.scheduler_enabled = true;
            c.save(p)?
        }
        Some("disable") => {
            c.scheduler_enabled = false;
            c.save(p)?
        }
        Some("pin") => {
            let m = m.context("model required")?;
            if !c.scheduler_pinned_models.contains(&m) {
                c.scheduler_pinned_models.push(m)
            }
            c.save(p)?
        }
        Some("unpin") => {
            let m = m.context("model required")?;
            c.scheduler_pinned_models.retain(|x| x != &m);
            c.save(p)?
        }
        Some(x) => bail!("unknown scheduler action '{x}'"),
    }
    if a.is_some() {
        refresh_swap(c, p)?;
    }
    Ok(())
}
static FLAG_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"--([A-Za-z0-9][A-Za-z0-9-]*)").unwrap());

fn flags(p: &Paths, pat: Option<&str>) -> Result<()> {
    let text = process::server_help(p)?;
    let profiles = Profiles::load(p)?;
    let mut shown = 0;
    for line in text.lines() {
        let Some(flag) = FLAG_RE.captures(line).map(|capture| capture[1].to_owned()) else {
            continue;
        };
        if pat.is_some_and(|pattern| !flag.contains(pattern) && !line.contains(pattern)) {
            continue;
        }
        shown += 1;
        println!(
            "{} {}",
            if profiles.is_exposed(&flag) {
                "  "
            } else {
                "✗"
            },
            line.trim()
        );
    }
    if shown == 0 {
        bail!("no llama-server flags match '{}'", pat.unwrap_or("*"))
    }
    Ok(())
}
fn fit(
    c: &Config,
    p: &Paths,
    query: Option<&str>,
    contexts: bool,
    compact: bool,
    extra: &[String],
) -> Result<()> {
    let profiles = Profiles::load(p)?;
    let selected = if let Some(query) = query {
        let profile = profiles.profiles.contains_key(query).then_some(query);
        let owner = profile
            .and_then(|name| profiles.owner(name))
            .unwrap_or(query);
        let (_, path, id) = models::resolve(c, Some(owner))?;
        vec![(id, path.context("remote model has no local size")?, profile)]
    } else {
        models::scan(c)
            .into_iter()
            .map(|model| (model.id, model.path, None))
            .collect()
    };
    println!("  {:<42} WEIGHTS  EST. VRAM  CONTEXT", "MODEL");
    for (id, path, profile) in selected {
        let mut args = common_fit_args(c, &profiles, profile)?;
        args.extend(extra.to_owned());
        let weights = models::model_bytes(&path);
        let trained_context = models::context_limit(&path);
        let ctx = flag_u64(&args, "ctx-size")
            .unwrap_or(c.ctx_size)
            .min(trained_context.unwrap_or(u64::MAX));
        let estimate = models::estimate_vram(&path, &args) as f64;
        println!(
            "  {:<42} {:>6.1}G  {:>8.1}G  {:>7}",
            profile.unwrap_or(&id),
            weights as f64 / (1u64 << 30) as f64,
            estimate / (1u64 << 30) as f64,
            ctx
        );
        if contexts && !compact {
            for step in [4096, 8192, 16_384, 32_768, 65_536, 131_072] {
                if trained_context.is_some_and(|limit| step > limit) {
                    println!("    ctx {:>7}: exceeds trained context", step);
                    continue;
                }
                let mut step_args = args.clone();
                step_args.extend(["--ctx-size".into(), step.to_string()]);
                let total = models::estimate_vram(&path, &step_args) as f64;
                println!(
                    "    ctx {:>7}: {:>6.1} G",
                    step,
                    total / (1u64 << 30) as f64
                );
            }
        }
    }
    Ok(())
}

fn common_fit_args(c: &Config, profiles: &Profiles, profile: Option<&str>) -> Result<Vec<String>> {
    let mut args = process::common_args(c);
    if let Some(profile) = profile {
        args.extend(profiles.args(profile)?);
    }
    Ok(args)
}
fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let long = format!("--{name}");
    args.windows(2)
        .rev()
        .find(|pair| pair[0] == long)
        .map(|pair| pair[1].as_str())
}
fn flag_u64(args: &[String], name: &str) -> Option<u64> {
    flag_value(args, name).and_then(|value| value.parse().ok())
}
pub(crate) fn ensure_model_not_loaded(c: &Config, p: &Paths, model: &str) -> Result<()> {
    if process::pid(p).is_none() {
        return Ok(());
    }
    let url = format!("http://{}:{}/running", c.host, c.port);
    let response = reqwest::blocking::get(url)?;
    if !response.status().is_success() {
        bail!("server is in direct mode; stop it before deleting models")
    }
    let payload: serde_json::Value = response.json()?;
    if payload
        .get("running")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .any(|item| {
            item.get("model")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|name| name == model || name.starts_with(&format!("{model}@")))
        })
    {
        bail!("model '{model}' is loaded; unload it before deletion")
    }
    Ok(())
}

pub(crate) fn refresh_swap(c: &Config, p: &Paths) -> Result<()> {
    if process::pid(p).is_some() {
        let profiles = Profiles::load(p)?;
        process::write_swap_config(c, p, &profiles)?;
    }
    Ok(())
}

pub fn service_enabled() -> bool {
    directories::BaseDirs::new()
        .map(|base| {
            base.home_dir()
                .join(".config/systemd/user/default.target.wants/llamactl.service")
                .exists()
        })
        .unwrap_or(false)
}

pub fn set_start_on_boot(p: &Paths, enabled: bool) -> Result<()> {
    if enabled {
        install_service(p)?;
    }
    let action = if enabled { "enable" } else { "disable" };
    let status = Command::new("systemctl")
        .args(["--user", action, "llamactl.service"])
        .status()?;
    if !status.success() {
        bail!("systemctl --user {action} llamactl.service failed with {status}")
    }
    Ok(())
}

pub fn install_service(p: &Paths) -> Result<()> {
    let home = directories::BaseDirs::new()
        .context("home unavailable")?
        .home_dir()
        .to_owned();
    let dir = home.join(".config/systemd/user");
    fs::create_dir_all(&dir)?;
    let exe = std::env::current_exe()?;
    fs::write(
        dir.join("llamactl.service"),
        format!(
            "[Unit]\nDescription=llama.cpp OpenAI-compatible server (llamactl)\nAfter=network-online.target\n\n[Service]\nExecStart={} serve\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n",
            exe.display()
        ),
    )?;
    fs::write(
        dir.join("llamactl-update.service"),
        format!(
            "[Unit]\nDescription=Update llama.cpp and restart server if needed\n\n[Service]\nType=oneshot\nExecStart={} update --restart\n",
            exe.display()
        ),
    )?;
    fs::write(
        dir.join("llamactl-update.timer"),
        "[Unit]\nDescription=Daily llama.cpp update check\n\n[Timer]\nOnCalendar=daily\nRandomizedDelaySec=1h\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n",
    )?;
    let status = Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .status()?;
    if !status.success() {
        bail!("systemctl --user daemon-reload failed with {status}")
    }
    let _ = p;
    Ok(())
}

fn format_count(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn format_size(bytes: u64) -> String {
    let gib = 1u64 << 30;
    let mib = 1u64 << 20;
    if bytes >= gib {
        format!("{:.1} GiB", bytes as f64 / gib as f64)
    } else if bytes >= mib {
        format!("{:.1} MiB", bytes as f64 / mib as f64)
    } else {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    }
}

fn search_command(query: Option<&str>, templates: bool) -> Result<()> {
    let query = query.unwrap_or("");
    if templates {
        let hits = huggingface::search_templates(query)?;
        println!("  {:<42} {:>9}  PREVIEW", "REPOSITORY", "DOWNLOADS");
        for hit in &hits {
            let preview = hit.template.split_whitespace().collect::<Vec<_>>().join(" ");
            let preview = preview.chars().take(60).collect::<String>();
            println!(
                "  {:<42} {:>9}  {}",
                hit.id,
                format_count(hit.downloads),
                preview
            );
        }
        if hits.is_empty() {
            println!("  no public repositories with a chat template matched");
        }
    } else {
        let repositories = huggingface::search(query)?;
        println!(
            "  {:<42} {:>9}  {:<16}  UPDATED",
            "REPOSITORY", "DOWNLOADS", "LICENSE"
        );
        for repository in &repositories {
            println!(
                "  {:<42} {:>9}  {:<16}  {}",
                repository.id,
                format_count(repository.downloads),
                repository.license,
                repository.updated
            );
        }
        if repositories.is_empty() {
            println!("  no public, non-gated GGUF repositories matched");
        }
    }
    Ok(())
}

fn repo_command(id: &str) -> Result<()> {
    let artifacts = huggingface::artifacts(id)?;
    println!("  {:<12} {:>9}  {:<8}  NOTES", "QUANTIZATION", "SIZE", "FILES");
    for artifact in &artifacts {
        let files = if artifact.shard_count > 1 {
            format!("{} shards", artifact.shard_count)
        } else {
            "1 file".into()
        };
        let notes = if artifact.recommended {
            format!("{} - RECOMMENDED", artifact.description)
        } else {
            artifact.description.clone()
        };
        println!(
            "  {:<12} {:>9}  {:<8}  {}",
            artifact.label,
            format_size(artifact.size),
            files,
            notes
        );
    }
    if artifacts.is_empty() {
        println!("  no downloadable GGUF files found");
    }
    Ok(())
}

fn card_command(id: &str) -> Result<()> {
    let details = huggingface::details(id)?;
    println!("  {}", details.id);
    println!("  Author     {}", details.author);
    println!("  License    {}", details.license);
    println!("  Downloads  {}", details.downloads);
    println!("  Likes      {}", details.likes);
    println!("  Updated    {}", details.updated);
    println!("  Task       {}", details.task);
    println!("  Library    {}", details.library);
    println!("  Base model {}", details.base_model);
    println!("  Languages  {}", details.languages.join(", "));
    println!("  Tags       {}", details.tags.join(" - "));
    println!("  Page       {}", details.url);
    println!();
    println!("{}", details.readme);
    Ok(())
}

fn download_command(paths: &Paths, id: &str, quant: Option<&str>) -> Result<()> {
    let artifacts = huggingface::artifacts(id)?;
    let artifact = match quant {
        Some(label) => artifacts
            .iter()
            .find(|artifact| artifact.label.eq_ignore_ascii_case(label))
            .with_context(|| {
                let available = artifacts
                    .iter()
                    .map(|artifact| artifact.label.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("quantization '{label}' not found; available: {available}")
            })?,
        None => artifacts
            .iter()
            .find(|artifact| artifact.recommended)
            .or_else(|| artifacts.first())
            .context("no downloadable GGUF files found")?,
    };
    let (owner, name) = id
        .split_once('/')
        .context("repository must be owner/name")?;
    let root = paths.data_dir.join("models");
    let destination = root.join(owner).join(name);
    eprintln!(
        "- downloading {} from {} to {}",
        artifact.label,
        id,
        destination.display()
    );
    let handle =
        huggingface::spawn_download(id.to_owned(), artifact.files.clone(), destination.clone());
    for event in handle.events {
        match event {
            huggingface::DownloadEvent::FileStarted { path, total } => {
                eprintln!("- {path} ({})", format_size(total));
            }
            huggingface::DownloadEvent::FileDone { path, skipped } => {
                eprintln!("  {} {}", if skipped { "kept" } else { "done" }, path);
            }
            huggingface::DownloadEvent::Finished(result) => {
                let summary = result.map_err(anyhow::Error::msg)?;
                println!(
                    "✓ downloaded {} file(s) to {}",
                    summary.downloaded + summary.skipped,
                    summary.destination.display()
                );
                return Ok(());
            }
            _ => {}
        }
    }
    bail!("download worker stopped unexpectedly")
}

fn template_fetch_command(paths: &Paths, id: &str, save: bool) -> Result<()> {
    let Some(template) = huggingface::fetch_chat_template(id)? else {
        bail!("repository '{id}' has no chat template")
    };
    if save {
        let mut library = templates::Templates::load(paths)?;
        let name = id.rsplit('/').next().unwrap_or(id).to_owned();
        templates::valid_name(&name)?;
        if library.templates.contains_key(&name) {
            bail!("template '{name}' already exists")
        }
        library.templates.insert(name.clone(), template);
        library.save(paths)?;
        println!("✓ saved template {name}");
    } else {
        print!("{template}");
    }
    Ok(())
}

fn logs_command(cfg: &Config, paths: &Paths, lines: usize) -> Result<()> {
    let mut log = if process::pid(paths).is_some() {
        process::upstream_log(cfg, lines)
    } else {
        String::new()
    };
    if log.trim().is_empty() {
        log = fs::read_to_string(&paths.log).unwrap_or_default();
    }
    if log.trim().is_empty() {
        println!("- no server log available");
    } else {
        print!("{}", tail_lines(&log, lines));
    }
    Ok(())
}

fn templates_command(paths: &Paths, action: Option<TemplateAction>) -> Result<()> {
    let mut library = templates::Templates::load(paths)?;
    match action {
        None | Some(TemplateAction::List) => {
            println!("  {:<30} {:>9}  PREVIEW", "NAME", "SIZE");
            for (name, template) in &library.templates {
                let preview = template.split_whitespace().collect::<Vec<_>>().join(" ");
                let preview = preview.chars().take(50).collect::<String>();
                println!(
                    "  {:<30} {:>9}  {}",
                    name,
                    format_size(template.len() as u64),
                    preview
                );
            }
        }
        Some(TemplateAction::Show { name }) => {
            let template = library
                .templates
                .get(&name)
                .with_context(|| format!("unknown template '{name}'"))?;
            print!("{template}");
        }
        Some(TemplateAction::Add { name, file }) => {
            templates::valid_name(&name)?;
            if library.templates.contains_key(&name) {
                bail!("template '{name}' already exists")
            }
            let content = read_input(file.as_deref())?;
            library.templates.insert(name.clone(), content);
            library.save(paths)?;
            println!("✓ added template {name}");
        }
        Some(TemplateAction::Edit { name, file }) => {
            if !library.templates.contains_key(&name) {
                bail!("unknown template '{name}'")
            }
            let content = read_input(file.as_deref())?;
            library.templates.insert(name.clone(), content);
            library.save(paths)?;
            println!("✓ updated template {name}");
        }
        Some(TemplateAction::Rename { old, new }) => {
            templates::valid_name(&new)?;
            if library.templates.contains_key(&new) {
                bail!("template '{new}' already exists")
            }
            let value = library
                .templates
                .remove(&old)
                .with_context(|| format!("unknown template '{old}'"))?;
            library.templates.insert(new.clone(), value);
            library.save(paths)?;
            println!("✓ renamed template {old} to {new}");
        }
        Some(TemplateAction::Delete { name }) => {
            library
                .templates
                .remove(&name)
                .with_context(|| format!("unknown template '{name}'"))?;
            library.save(paths)?;
            println!("✓ deleted template {name}");
        }
    }
    Ok(())
}

fn read_input(file: Option<&std::path::Path>) -> Result<String> {
    match file {
        Some(path) => fs::read_to_string(path).with_context(|| format!("read {}", path.display())),
        None => {
            let mut content = String::new();
            std::io::stdin()
                .read_to_string(&mut content)
                .context("read template from stdin")?;
            Ok(content)
        }
    }
}

fn tail_lines(text: &str, max_lines: usize) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    if lines.len() <= max_lines {
        return text.to_owned();
    }
    lines[lines.len() - max_lines..].join("\n")
}
