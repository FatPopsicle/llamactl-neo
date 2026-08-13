mod config;
mod drm;
mod models;
mod process;
mod profiles;
mod ui;
mod update;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use config::{Config, Paths};
use profiles::Profiles;
use std::{fs, io::IsTerminal, process::Command, sync::LazyLock};

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
}
#[derive(Subcommand)]
enum ProfileAction {
    Clone { source: String, new_name: String },
    Rename { source: String, new_name: String },
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
                    println!("✓ restarted · pid {}", process::restart(&cfg, &paths, &p)?);
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
                    "✓ restarted · pid {}",
                    process::restart(&cfg, &paths, &profiles)?
                );
            }
        }
        Some(Commands::Start { model, extra }) => {
            let p = Profiles::load(&paths)?;
            println!(
                "✓ server started · pid {}",
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
                "• server is not running"
            }
        ),
        Some(Commands::Restart) => {
            let p = Profiles::load(&paths)?;
            println!(
                "✓ server restarted · pid {}",
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
            println!("✓ loaded · {model}")
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
fn profile_command(c: &Config, p: &Paths, a: Option<ProfileAction>) -> Result<()> {
    let mut d = Profiles::load(p)?;
    let changed = a.is_some();
    match a {
        None => {
            println!("  {:<30} OWNER", "PROFILE");
            for n in d.profiles.keys() {
                println!("  {:<30} {}", n, d.owner(n).unwrap_or("unassigned"))
            }
        }
        Some(ProfileAction::Clone { source, new_name }) => {
            d.clone_profile(&source, &new_name)?;
            d.save(p)?
        }
        Some(ProfileAction::Rename { source, new_name }) => {
            d.rename(&source, &new_name)?;
            d.save(p)?
        }
        Some(ProfileAction::Delete { name }) => {
            d.remove(&name)?;
            d.save(p)?
        }
    }
    if changed {
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
fn ensure_model_not_loaded(c: &Config, p: &Paths, model: &str) -> Result<()> {
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

fn refresh_swap(c: &Config, p: &Paths) -> Result<()> {
    if process::pid(p).is_some() {
        let profiles = Profiles::load(p)?;
        process::write_swap_config(c, p, &profiles)?;
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
