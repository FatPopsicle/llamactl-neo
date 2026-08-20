use crate::config::{Config, Paths};
use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf};
use tar::Archive;

const LLAMA_REPO: &str = "ggml-org/llama.cpp";
const SWAP_REPO: &str = "mostlygeek/llama-swap";
#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}
#[derive(Debug, Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}
#[derive(Serialize)]
struct BuildInfo<'a> {
    tag: &'a str,
    backend: &'a str,
    source: bool,
    built_at: u64,
}
fn latest(repo: &str) -> Result<Release> {
    reqwest::blocking::Client::new()
        .get(format!(
            "https://api.github.com/repos/{repo}/releases/latest"
        ))
        .header("User-Agent", "llamactl-rust")
        .send()?
        .error_for_status()?
        .json()
        .context("decode GitHub release")
}
pub fn check(paths: &Paths) -> Result<(String, bool, String, bool)> {
    let llama = latest(LLAMA_REPO)?;
    let swap = latest(SWAP_REPO)?;
    let swap_tag = fs::read_to_string(paths.swap_bin.parent().unwrap().join("version"))
        .ok()
        .map(|s| s.trim().to_owned());
    let llama_installed = fs::read_dir(&paths.versions)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .any(|name| name == llama.tag_name || name.starts_with(&format!("{}-", llama.tag_name)));
    Ok((
        llama.tag_name.clone(),
        !llama_installed,
        swap.tag_name.clone(),
        swap_tag.as_deref() != Some(&swap.tag_name),
    ))
}

// --- managed vllm ---------------------------------------------------------
// vllm is a pip package, so "managed" means one managed venv under data_dir,
// upgrade-in-place and version-tracked (not the binary versions/<tag> +
// current-symlink + prune model, which is specific to downloaded binaries).
// ponytail: install/upgrade execution (install_vllm) + auto-update wiring are
// the heavy/network follow-on; resolution + version read land here first.
pub fn vllm_venv(paths: &Paths) -> PathBuf {
    paths.data_dir.join("vllm")
}
pub fn vllm_binary(paths: &Paths) -> Option<PathBuf> {
    let bin = vllm_venv(paths).join("bin").join("vllm");
    bin.is_file().then_some(bin)
}
/// Installed vllm version, read from the managed venv's dist-info dir name
/// (instant; does not import vllm/torch).
pub fn vllm_installed_version(paths: &Paths) -> Option<String> {
    let lib = vllm_venv(paths).join("lib");
    let Ok(pys) = fs::read_dir(lib) else { return None };
    for py in pys.filter_map(Result::ok) {
        let site = py.path().join("site-packages");
        let Ok(entries) = fs::read_dir(&site) else { continue };
        for entry in entries.filter_map(Result::ok) {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(rest) = name.strip_prefix("vllm-")
                && rest.ends_with(".dist-info")
            {
                return Some(rest.trim_end_matches(".dist-info").to_owned());
            }
        }
    }
    None
}
pub fn vllm_latest() -> Result<String> {
    let value: serde_json::Value = reqwest::blocking::Client::new()
        .get("https://pypi.org/pypi/vllm/json")
        .header("User-Agent", "llamactl-rust")
        .send()?
        .error_for_status()?
        .json()
        .context("decode PyPI release")?;
    value
        .get("info")
        .and_then(|info| info.get("version"))
        .and_then(|version| version.as_str())
        .map(str::to_owned)
        .context("no version in PyPI response")
}
pub fn vllm_check(paths: &Paths) -> Result<(Option<String>, String, bool)> {
    let latest = vllm_latest()?;
    let installed = vllm_installed_version(paths);
    let available = installed.as_deref() != Some(latest.as_str());
    Ok((installed, latest, available))
}
/// Install/upgrade the managed vllm venv. Separate from `llamactl update`
/// (which manages llama.cpp/llama-swap). Untested-by-execution like
/// `install_llama` — heavy/network, fails loudly with pip's stderr.
pub fn install_vllm(paths: &Paths, version: Option<&str>) -> Result<()> {
    let venv = vllm_venv(paths);
    let pip = venv.join("bin").join("pip");
    if !pip.is_file() {
        create_venv_with_pip(&venv)?;
    }
    let mut args: Vec<String> = vec!["install".into(), "--upgrade".into()];
    // ponytail: --only-binary fails fast instead of multi-hour source builds;
    // drop if a needed dep ever ships source-only.
    args.push("--only-binary=:all:".into());
    match version {
        Some(v) => args.push(format!("vllm=={v}")),
        None => args.push("vllm".into()),
    }
    eprintln!("- installing/upgrading vllm into {}", venv.display());
    let status = std::process::Command::new(&pip)
        .args(&args)
        .status()
        .context("pip install vllm")?;
    if !status.success() {
        bail!("vllm install failed");
    }
    let installed = vllm_installed_version(paths).unwrap_or_else(|| "?".into());
    println!("✓ vllm {installed} at {}", venv.display());
    Ok(())
}

fn create_venv_with_pip(venv: &std::path::Path) -> Result<()> {
    let python = find_python3()?;
    if let Some(parent) = venv.parent() {
        fs::create_dir_all(parent)?;
    }
    // Prefer the standard pip-provisioned venv; fall back to get-pip when
    // ensurepip is absent (common on minimal systems).
    let status = std::process::Command::new(&python)
        .args(["-m", "venv"])
        .arg(venv)
        .status()
        .context("create venv")?;
    let pip = venv.join("bin").join("pip");
    if status.success() && pip.is_file() {
        return Ok(());
    }
    let _ = fs::remove_dir_all(venv);
    let status = std::process::Command::new(&python)
        .args(["-m", "venv", "--without-pip"])
        .arg(venv)
        .status()
        .context("create venv")?;
    if !status.success() {
        bail!("venv creation failed");
    }
    let bytes = reqwest::blocking::Client::new()
        .get("https://bootstrap.pypa.io/get-pip.py")
        .header("User-Agent", "llamactl-rust")
        .send()?
        .error_for_status()?
        .bytes()?;
    let getpip = tempfile::NamedTempFile::new()?;
    fs::write(getpip.path(), bytes)?;
    let status = std::process::Command::new(venv.join("bin").join("python"))
        .arg(getpip.path())
        .status()
        .context("bootstrap pip")?;
    if !status.success() {
        bail!("pip bootstrap failed");
    }
    Ok(())
}

fn find_python3() -> Result<PathBuf> {
    let probe = std::process::Command::new("python3")
        .arg("--version")
        .output()
        .context("locate python3")?;
    if !probe.status.success() {
        bail!("python3 not usable on PATH");
    }
    Ok(PathBuf::from("python3"))
}
pub fn install(cfg: &Config, paths: &Paths) -> Result<()> {
    install_llama(cfg, paths)?;
    install_swap(paths)
}

pub fn install_llama(cfg: &Config, paths: &Paths) -> Result<()> {
    let llama = latest(LLAMA_REPO)?;
    let marker = if cfg.backend == "cpu" {
        "bin-ubuntu-x64.tar.gz".to_owned()
    } else {
        format!("bin-ubuntu-{}", cfg.backend)
    };
    let asset = llama
        .assets
        .iter()
        .find(|a| a.name.contains(&marker) && a.name.ends_with("x64.tar.gz"))
        .with_context(|| format!("no ubuntu-x64 asset for backend '{}'", cfg.backend))?;
    fs::create_dir_all(&paths.versions)?;
    fs::create_dir_all(&paths.data_dir)?;
    let response = download(asset)?;
    let temp = tempfile::tempdir_in(&paths.data_dir)?;
    Archive::new(GzDecoder::new(response)).unpack(temp.path())?;
    let server = walkdir::WalkDir::new(temp.path())
        .into_iter()
        .filter_map(Result::ok)
        .find(|e| e.file_name() == "llama-server")
        .context("llama-server not found in release")?;
    let source = server.path().parent().unwrap();
    let dest = paths
        .versions
        .join(format!("{}-{}", llama.tag_name, cfg.backend));
    if dest.exists() {
        fs::remove_dir_all(&dest)?;
    }
    copy_dir(source, &dest)?;
    write_build_info(&dest, &llama.tag_name, &cfg.backend, false)?;
    switch_link(&paths.current, &dest)?;
    prune_versions(cfg, paths)?;
    Ok(())
}

pub fn install_swap(paths: &Paths) -> Result<()> {
    let swap = latest(SWAP_REPO)?;
    let asset = swap
        .assets
        .iter()
        .find(|asset| asset.name.contains("linux_amd64.tar.gz"))
        .context("no Linux llama-swap asset")?;
    let response = download(asset)?;
    let temp = tempfile::tempdir_in(&paths.data_dir)?;
    Archive::new(GzDecoder::new(response)).unpack(temp.path())?;
    let binary = walkdir::WalkDir::new(temp.path())
        .into_iter()
        .filter_map(Result::ok)
        .find(|entry| entry.file_name() == "llama-swap" && entry.file_type().is_file())
        .context("llama-swap missing from release")?;
    fs::create_dir_all(paths.swap_bin.parent().unwrap())?;
    let staging = PathBuf::from(format!("{}.download", paths.swap_bin.display()));
    fs::copy(binary.path(), &staging)?;
    fs::set_permissions(&staging, fs::Permissions::from_mode(0o755))?;
    fs::rename(&staging, &paths.swap_bin)?;
    fs::write(
        paths.swap_bin.parent().unwrap().join("version"),
        &swap.tag_name,
    )?;
    Ok(())
}
pub fn build_source(_cfg: &Config, paths: &Paths, backend: &str) -> Result<()> {
    let release = latest(LLAMA_REPO)?;
    fs::create_dir_all(&paths.data_dir)?;
    fs::create_dir_all(&paths.versions)?;
    let source_url = format!(
        "https://github.com/{LLAMA_REPO}/archive/refs/tags/{}.tar.gz",
        release.tag_name
    );
    eprintln!("- downloading llama.cpp {} source", release.tag_name);
    let response = reqwest::blocking::Client::new()
        .get(source_url)
        .header("User-Agent", "llamactl-rust")
        .send()?
        .error_for_status()?;
    let temp = tempfile::tempdir_in(&paths.data_dir)?;
    Archive::new(GzDecoder::new(response)).unpack(temp.path())?;
    let source = fs::read_dir(temp.path())?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.join("CMakeLists.txt").is_file())
        .context("CMakeLists.txt not found in source release")?;
    let build = temp.path().join("build");
    let mut flags = vec!["-DGGML_NATIVE=ON"];
    match backend {
        "cuda" => flags.extend([
            "-DGGML_CUDA=ON",
            "-DGGML_CUDA_NCCL=ON",
            "-DGGML_CUDA_FA_ALL_QUANTS=ON",
        ]),
        "vulkan" => flags.push("-DGGML_VULKAN=ON"),
        "rocm" => flags.push("-DGGML_HIP=ON"),
        "sycl-fp16" => flags.extend(["-DGGML_SYCL=ON", "-DGGML_SYCL_F16=ON"]),
        "sycl-fp32" => flags.extend(["-DGGML_SYCL=ON", "-DGGML_SYCL_F16=OFF"]),
        "openvino" => flags.push("-DGGML_OPENVINO=ON"),
        "cpu" => {}
        _ => bail!("invalid backend '{backend}'"),
    }
    let configured = std::process::Command::new("cmake")
        .args([
            "-S",
            source.to_str().unwrap(),
            "-B",
            build.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        .args(flags)
        .status()
        .context("run CMake configure (is cmake installed?)")?;
    if !configured.success() {
        bail!("llama.cpp CMake configuration failed")
    }
    let jobs = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(16);
    let compiled = std::process::Command::new("cmake")
        .args([
            "--build",
            build.to_str().unwrap(),
            "--config",
            "Release",
            "--target",
            "llama-server",
            "--parallel",
            &jobs.to_string(),
        ])
        .status()?;
    if !compiled.success() {
        bail!("llama.cpp build failed")
    }
    let bin = build.join("bin");
    let dest = paths
        .versions
        .join(format!("{}-{backend}", release.tag_name));
    if dest.exists() {
        fs::remove_dir_all(&dest)?;
    }
    copy_dir(&bin, &dest)?;
    write_build_info(&dest, &release.tag_name, backend, true)?;
    switch_link(&paths.current, &dest)?;
    prune_versions(_cfg, paths)?;
    Ok(())
}

fn write_build_info(path: &std::path::Path, tag: &str, backend: &str, source: bool) -> Result<()> {
    let built_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let info = BuildInfo {
        tag,
        backend,
        source,
        built_at,
    };
    fs::write(
        path.join(".llamactl-build.json"),
        format!("{}\n", serde_json::to_string_pretty(&info)?),
    )?;
    Ok(())
}

fn prune_versions(cfg: &Config, paths: &Paths) -> Result<()> {
    let active = fs::canonicalize(&paths.current).ok();
    let mut by_backend = std::collections::BTreeMap::<String, Vec<PathBuf>>::new();
    if !paths.versions.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(&paths.versions)?.filter_map(Result::ok) {
        let path = entry.path();
        if !path.join("llama-server").is_file() {
            continue;
        }
        let manifest = fs::read_to_string(path.join(".llamactl-build.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());
        let backend = manifest
            .as_ref()
            .and_then(|value| value.get("backend").and_then(|item| item.as_str()))
            .unwrap_or("unknown")
            .to_owned();
        by_backend.entry(backend).or_default().push(path);
    }
    for paths_for_backend in by_backend.values_mut() {
        paths_for_backend.sort_by_key(|path| {
            std::cmp::Reverse(path.metadata().and_then(|meta| meta.modified()).ok())
        });
        for old in paths_for_backend.iter().skip(cfg.keep_versions.max(1)) {
            let pinned = fs::read_to_string(old.join(".llamactl-build.json"))
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| value.get("pinned").and_then(|item| item.as_bool()))
                .unwrap_or(false);
            if !pinned && active.as_ref() != fs::canonicalize(old).ok().as_ref() {
                let _ = fs::remove_dir_all(old);
            }
        }
    }
    Ok(())
}

fn download(asset: &Asset) -> Result<reqwest::blocking::Response> {
    eprintln!("- downloading {}", asset.name);
    Ok(reqwest::blocking::Client::new()
        .get(&asset.browser_download_url)
        .header("User-Agent", "llamactl-rust")
        .send()?
        .error_for_status()?)
}
fn copy_dir(from: &std::path::Path, to: &std::path::Path) -> Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}
fn switch_link(link: &std::path::Path, target: &std::path::Path) -> Result<()> {
    let temp = PathBuf::from(format!("{}.new", link.display()));
    let _ = fs::remove_file(&temp);
    std::os::unix::fs::symlink(target, &temp)?;
    if link.exists() || link.symlink_metadata().is_ok() {
        fs::remove_file(link)?;
    }
    fs::rename(temp, link)?;
    Ok(())
}

#[cfg(test)]
mod vllm_tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn paths(data_dir: &std::path::Path) -> Paths {
        Paths {
            data_dir: data_dir.to_path_buf(),
            state_dir: PathBuf::new(),
            config: PathBuf::new(),
            profiles: PathBuf::new(),
            versions: PathBuf::new(),
            current: PathBuf::new(),
            pid: PathBuf::new(),
            launch: PathBuf::new(),
            log: PathBuf::new(),
            swap_bin: PathBuf::new(),
            swap_config: PathBuf::new(),
            metadata_cache: PathBuf::new(),
            profile_benchmarks: PathBuf::new(),
            templates: PathBuf::new(),
        }
    }

    #[test]
    fn reads_installed_version_and_binary_from_managed_venv() {
        let dir = tempfile::tempdir().unwrap();
        let site = dir.path().join("vllm/lib/python3.13/site-packages");
        fs::create_dir_all(&site).unwrap();
        fs::create_dir_all(site.join("vllm-0.27.1.dist-info")).unwrap();
        let p = paths(dir.path());
        assert_eq!(vllm_installed_version(&p).as_deref(), Some("0.27.1"));
        assert!(vllm_binary(&p).is_none());
        fs::create_dir_all(dir.path().join("vllm/bin")).unwrap();
        fs::write(dir.path().join("vllm/bin/vllm"), b"#!/bin/sh\n").unwrap();
        assert!(vllm_binary(&p).is_some());
    }

    #[test]
    fn missing_venv_means_not_installed() {
        let dir = tempfile::tempdir().unwrap();
        let p = paths(dir.path());
        assert_eq!(vllm_installed_version(&p), None);
        assert!(vllm_binary(&p).is_none());
    }
}
