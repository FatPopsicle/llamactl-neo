# llamactl NEO

A native Rust manager for local `llama.cpp` servers, with a full-screen terminal interface built on [Ratatui](https://ratatui.rs/).

The Rust implementation preserves llamactl's XDG paths and JSON configuration/profile formats, so it can use an existing installation without conversion.

See [INSTALL.md](INSTALL.md) for the complete installation, backend, systemd, update, and troubleshooting guide.

## Quick install

```bash
cargo build --release
install -Dm755 target/release/llamactl ~/.local/bin/llamactl
```

The release profile enables thin LTO, single-unit code generation, symbol stripping, and abort-on-panic for a smaller, faster binary.

Rust 1.85 or newer is recommended. Linux is currently the supported platform. After installing the binary, select a backend and install the managed runtimes:

```bash
llamactl config backend vulkan  # or cuda / cpu
llamactl update
llamactl models
llamactl start
```

## Use

```bash
llamactl                 # Ratatui control center
llamactl models          # discover local GGUF models
llamactl update          # install llama.cpp and llama-swap releases
llamactl start           # background llama-swap catalog
llamactl start MODEL     # background single-model server
llamactl serve [MODEL]   # foreground server
llamactl status
llamactl --help
```

The UI has six workspaces: Dashboard, Models, Profiles, Settings, Logs, and Maintenance. Use arrow keys or `1`–`6` to navigate, Enter to act, `r` to refresh, and `q` to quit.

## Configuration

Files remain compatible with the original app:

- `$XDG_CONFIG_HOME/llamactl/config.json`
- `$XDG_CONFIG_HOME/llamactl/profiles.json`
- `$XDG_DATA_HOME/llamactl/` for managed runtimes
- `$XDG_STATE_HOME/llamactl/` for PID, launch metadata, generated swap config, and logs

By default, llama-swap advertises base model IDs only. A base model uses its assigned profile, if present. Profile aliases can be exposed separately from the Settings page or with `advertise_profiles`.

Common examples:

```bash
llamactl config backend vulkan
llamactl config host 0.0.0.0
llamactl config models_dirs '["/srv/models"]'
llamactl config advertise_base_models true
llamactl config advertise_profiles false
llamactl keys generate
llamactl scheduler pin MODEL
llamactl profiles clone quality quality-long
```

## Source layout

- `src/config.rs` — XDG paths, typed config, atomic persistence
- `src/models.rs` — GGUF discovery, split shards, IDs, deletion
- `src/profiles.rs` — compatible profile storage and argument generation
- `src/process.rs` — commands, background lifecycle, swap configuration
- `src/update.rs` — GitHub release installation and source builds
- `src/ui.rs` — Ratatui application
- `src/main.rs` — Clap CLI and command dispatch

## Performance notes

- GGUF metadata and tokenizer compatibility hashes are persisted in the state directory and invalidated by file size and modification time.
- External speculative draft models are checked for compatible BOS/EOS behavior and vocabulary prefixes before launch.
- Profile advanced settings expose Jinja, inline/file chat-template overrides, and template kwargs.
- The dashboard shows llama-swap request/input/output/cache totals and uses its performance feed for GPU telemetry when available.
- Routine estimates read only the GGUF key/value header; tensor names are scanned only for MTP capability checks.
- Runtime archives are streamed through gzip/tar rather than buffered fully in RAM.
- The scheduler generates feasible combinations without an exponential-width bitmask.
