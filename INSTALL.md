# Installing llamactl NEO

llamactl NEO is currently supported on Linux. Its managed-runtime downloader targets Linux x86-64 releases of `llama.cpp` and `llama-swap`.

## 1. Requirements

For llamactl itself:

- Linux
- Rust 1.85 or newer when building from source
- A C/C++ toolchain and linker
- Internet access to GitHub when installing or updating managed runtimes

Install a basic build environment using your distribution's package manager:

### Debian or Ubuntu

```bash
sudo apt update
sudo apt install build-essential curl
```

### Fedora

```bash
sudo dnf install gcc gcc-c++ make curl
```

### Arch Linux

```bash
sudo pacman -S --needed base-devel curl
```

Install Rust through [rustup](https://rustup.rs/) if `cargo` is not already available, then verify it:

```bash
rustc --version
cargo --version
```

Your GPU driver must already work on the host. For example, confirm NVIDIA cards with `nvidia-smi` before selecting the CUDA backend.

## 2. Install llamactl

### From source

From the extracted or checked-out llamactl NEO source directory:

```bash
cargo build --locked --release
install -Dm755 target/release/llamactl ~/.local/bin/llamactl
```

Ensure `~/.local/bin` is on `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

To make that permanent, add the export to `~/.profile`, `~/.bashrc`, or the equivalent file for your shell.

Verify the installation:

```bash
llamactl --version
llamactl --help
```

### From a supplied binary

If you received a prebuilt `llamactl` executable, install it directly:

```bash
chmod +x ./llamactl
install -Dm755 ./llamactl ~/.local/bin/llamactl
llamactl --version
```

Only install binaries obtained from a source you trust.

## 3. Select a backend

The default backend is Vulkan. Set the backend before installing the managed runtime:

```bash
llamactl config backend vulkan
```

Available backend names are:

| Backend | Command | Host requirement |
| --- | --- | --- |
| Vulkan | `llamactl config backend vulkan` | Working Vulkan driver and loader |
| NVIDIA CUDA | `llamactl config backend cuda` | Compatible NVIDIA driver |
| CPU | `llamactl config backend cpu` | No GPU runtime required |
| AMD ROCm | `llamactl config backend rocm` | Compatible ROCm installation |
| Intel SYCL FP16 | `llamactl config backend sycl-fp16` | Compatible oneAPI/SYCL runtime |
| Intel SYCL FP32 | `llamactl config backend sycl-fp32` | Compatible oneAPI/SYCL runtime |
| OpenVINO | `llamactl config backend openvino` | Compatible OpenVINO runtime |

Not every upstream `llama.cpp` release publishes an archive for every backend. If `llamactl update` reports that no matching release asset exists, use the source-build instructions below.

## 4. Install llama.cpp and llama-swap

Install the latest managed runtime and llama-swap release:

```bash
llamactl update
```

The first launch of `llamactl`, `llamactl start`, or `llamactl serve` also performs this installation automatically if no runtime is present.

Check the result:

```bash
llamactl status
llamactl builds
```

Managed files are stored under:

```text
~/.local/share/llamactl/current
~/.local/share/llamactl/versions
~/.local/share/llamactl/llama-swap
```

The actual base directory follows `XDG_DATA_HOME` when that variable is set.

## 5. Configure model directories

llamactl automatically checks common LM Studio model locations and its own data directory. To set model directories explicitly, pass a JSON array:

```bash
llamactl config models_dirs '["/home/USER/models", "/srv/gguf"]'
```

Use absolute paths and replace `USER` with your account name. Then verify discovery:

```bash
llamactl models
```

Split GGUF models are supported; point llamactl at the containing directory rather than an individual shard.

You can inspect estimated memory requirements before loading a model:

```bash
llamactl fit MODEL
llamactl fit MODEL --contexts
```

## 6. Start the server

Start the normal llama-swap catalog in the background:

```bash
llamactl start
llamactl status
```

Open the terminal UI with:

```bash
llamactl
```

By default, the OpenAI-compatible endpoint and llama-swap WebUI are available at:

```text
API:    http://127.0.0.1:1234/v1
WebUI:  http://127.0.0.1:1234/
        http://127.0.0.1:1234/ui
```

Test model advertisement:

```bash
curl http://127.0.0.1:1234/v1/models
```

Useful lifecycle commands:

```bash
llamactl load MODEL
llamactl unload MODEL
llamactl reload
llamactl restart
llamactl stop
```

To run one model directly without the llama-swap catalog:

```bash
llamactl start MODEL       # background
llamactl serve MODEL       # foreground
```

## 7. API authentication and LAN access

Generate an API key:

```bash
llamactl keys generate
```

List configured keys in redacted form:

```bash
llamactl keys list
```

When authentication is enabled, use the key as a bearer token:

```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  http://127.0.0.1:1234/v1/models
```

The server listens only on localhost by default. To accept LAN connections:

```bash
llamactl config host 0.0.0.0
llamactl restart
```

Configure an API key and host firewall before exposing the service. Do not expose the endpoint directly to the public Internet without an authenticated, TLS-enabled reverse proxy.

## 8. Optional systemd user service

Install user service units after placing the final binary at `~/.local/bin/llamactl`:

```bash
llamactl install-service
systemctl --user enable --now llamactl.service llamactl-update.timer
```

The service runs the llama-swap catalog. The timer checks for runtime updates daily and restarts the server when necessary. You can also toggle **Start on boot** from llamactl's Settings page; this installs and enables the server service (but not the optional update timer).

Inspect it with:

```bash
systemctl --user status llamactl.service
systemctl --user status llamactl-update.timer
journalctl --user -u llamactl.service -f
```

For a user service that must start at boot without an interactive login, an administrator can enable lingering for the account:

```bash
sudo loginctl enable-linger "$USER"
```

The generated unit stores the absolute path of the executable used to run `install-service`. Re-run `llamactl install-service` if the binary is moved.

## 9. Build llama.cpp from source

Use this when an upstream prebuilt archive is unavailable or when you want a native optimized build.

Install CMake and a compiler first. On Debian or Ubuntu:

```bash
sudo apt install build-essential cmake
```

Then build the selected backend:

```bash
llamactl build vulkan
# or: llamactl build cuda
# or: llamactl build cpu
```

Backend-specific development packages must already be installed:

- **Vulkan:** Vulkan headers/loader and a shader compiler such as `glslc`
- **CUDA:** CUDA toolkit and NCCL development libraries
- **ROCm:** ROCm/HIP development environment
- **SYCL:** Intel oneAPI/SYCL development environment
- **OpenVINO:** OpenVINO development environment

For example, a typical Ubuntu Vulkan setup also needs:

```bash
sudo apt install libvulkan-dev glslc
```

Build and restart an already running service in one step:

```bash
llamactl build vulkan --restart
```

Source compilation uses up to 16 parallel jobs and may require substantial disk space and RAM.

## 10. Files and migration

Default locations are:

| Purpose | Path |
| --- | --- |
| Main configuration | `~/.config/llamactl/config.json` |
| Profiles | `~/.config/llamactl/profiles.json` |
| Managed runtimes and models | `~/.local/share/llamactl/` |
| PID, logs, generated swap config, cache | `~/.local/state/llamactl/` |

The paths follow `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, and `XDG_STATE_HOME` when set.

The Rust implementation uses the existing llamactl JSON formats, so an older installation can normally be upgraded in place. Back up the configuration first if it contains important profiles:

```bash
cp -a ~/.config/llamactl ~/.config/llamactl.backup
```

## 11. Updating

Update managed `llama.cpp` and llama-swap releases:

```bash
llamactl update --check
llamactl update --restart
```

To update llamactl itself from a newer source tree:

```bash
cargo build --locked --release
install -Dm755 target/release/llamactl ~/.local/bin/llamactl
```

If a systemd service is running, restart it afterward:

```bash
systemctl --user restart llamactl.service
```

## 12. Troubleshooting

### `llamactl: command not found`

Confirm the binary exists and update `PATH`:

```bash
ls -l ~/.local/bin/llamactl
export PATH="$HOME/.local/bin:$PATH"
```

### No models are listed

Check the configured directories and permissions:

```bash
llamactl config
llamactl models
find /path/to/models -type f -iname '*.gguf' | head
```

A model must be at least 50 MB. Files beginning with `mmproj` are treated as vision projectors rather than standalone models.

### No release asset exists for the backend

Install the backend's development SDK and use:

```bash
llamactl build BACKEND
```

Alternatively select a backend with a published archive, such as `cpu`, `vulkan`, or `cuda`, and run `llamactl update` again.

### The server exits during startup

Inspect upstream llama.cpp output in the UI's Logs page or the state log:

```bash
tail -n 200 ~/.local/state/llamactl/server.log
```

Also verify the selected backend and driver:

```bash
llamactl status
llamactl builds
```

### Port 1234 is already in use

Choose another API port:

```bash
llamactl config port 1236
llamactl config telemetry_port 1237
llamactl restart
```

The API and telemetry ports must differ.

### Configuration changes do not affect a loaded model

Unload and reload the model, or restart the service:

```bash
llamactl unload MODEL
llamactl load MODEL
# or
llamactl restart
```

### Completely uninstall

Disable optional services first:

```bash
systemctl --user disable --now llamactl.service llamactl-update.timer
rm -f ~/.config/systemd/user/llamactl.service
rm -f ~/.config/systemd/user/llamactl-update.service
rm -f ~/.config/systemd/user/llamactl-update.timer
systemctl --user daemon-reload
```

Remove the binary:

```bash
rm -f ~/.local/bin/llamactl
```

Optionally remove configuration, downloaded runtimes, and state. This is destructive:

```bash
rm -rf ~/.config/llamactl ~/.local/share/llamactl ~/.local/state/llamactl
```
