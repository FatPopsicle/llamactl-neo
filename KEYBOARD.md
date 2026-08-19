# Keyboard Controls

llamactl NEO uses the same navigation and confirmation rules throughout the TUI. The footer shows the controls for the current workspace or modal; press `?` for the in-app reference.

## Common controls

| Key | Action |
|---|---|
| `Up` / `Down`, `k` / `j` | Select the previous or next item |
| `Home` / `End` | Select the first or last item |
| `Left` / `Right`, `h` / `l` | Switch workspace |
| `Tab` / `Shift+Tab` | Switch workspace forward or backward |
| `1`–`8` | Jump to Dashboard, Models, Profiles, Templates, Search, Settings, Logs, or Maintenance |
| `Enter` | Run the selected workspace's primary action |
| `r` | Refresh application state |
| `?` | Open or close the in-app control reference |
| `q` | Quit when no modal or editor is open |
| `Ctrl+C` | Quit; during a benchmark, cancel safely and restore the server |

Navigation keys are contextual inside editors and modals. For example, arrow keys move through fields in the Profile Editor instead of changing workspaces.

## Workspaces

### Dashboard

| Key | Action |
|---|---|
| `Enter` | Start or stop the server |

### Models

| Key | Action |
|---|---|
| `Enter` | Load or start the selected model |
| `c` | Create a profile for the selected model |
| `u` | Unload the selected model from the scheduler |
| `d` | Delete the selected model after confirmation |

### Profiles

| Key | Action |
|---|---|
| `Enter` | Load or start the selected profile |
| `m` | Start a profile benchmark |
| `v` | View retained benchmark results |
| `e` | Open the Profile Editor |
| `R` | Rename the profile |
| `c` | Clone the profile |
| `d` | Delete the profile after confirmation |
| `b` | Bind the profile to its owner model |
| `p` | Pin or unpin the profile in the scheduler |
| `u` | Unload the profile from the scheduler |
| `+` / `=` | Increase context by the configured Context step |
| `-` | Decrease context by the configured Context step |
| `[` / `]` | Decrease or increase parallel slots |
| `t` | Cycle split mode |
| `f` | Toggle flash attention |
| `k` | Cycle KV-cache type |

`r` always means refresh. Profile rename therefore uses uppercase `R`.

### Settings

| Key | Action |
|---|---|
| `Enter`, `+`, `=` | Enable, advance, or increase the selected setting |
| `-` | Disable, go backward, or decrease the selected setting |

### Logs

| Key | Action |
|---|---|
| `r` | Refresh logs and application state |

### Maintenance

| Key | Action |
|---|---|
| `Enter` | Run the selected maintenance action |

### Downloads

| Key | Action |
|---|---|
| `/` / `s` | Edit the Hugging Face search query |
| `Enter` | Open search when no results are listed; otherwise open the selected repository/model card or review the selected quantization |
| `i` | Open or reopen the selected repository’s model card |
| `d` | Cycle through configured model-directory destinations |
| `b` / `Esc` | Return from quantizations to repository results |
| `r` | Repeat the search or reload repository files |

Search results include only public, non-gated GGUF repositories. Opening a result first shows a scrollable model card with repository metadata, its Hugging Face URL, and README; use arrows or Page Up/Page Down to scroll and Enter or Esc to continue to quantizations. The confirmation dialog shows size, file count, license, destination, and verification mode. During a transfer, unrelated controls are locked; `Esc`, `q`, `c`, or `Ctrl+C` cancels while retaining resumable partial files.

## Profile Editor

| Key | Action |
|---|---|
| `Up` / `Down`, `k` / `j` | Select a field |
| `Left` / `Right`, `h` / `l` | Select the previous or next value |
| `t` | Enter an exact value for the selected field |
| `e` | Edit extra llama-server flags when Extra flags is selected |
| `Enter`, `s` | Save the profile |
| `Esc`, `q` | Cancel editing |

`Context step` appears directly below `Context size`. Use left/right to select a standard increment from 1,024 to 65,536 tokens, or press `t` to enter an exact increment in that range. It controls context adjustments in Settings, Profiles, and the Profile Editor. On `Draft tokens`, left/right or `h`/`l` decreases or increases the amount by one, stopping at zero.

## Pickers and result views

| Context | Controls |
|---|---|
| Runtime picker | `Up`/`Down` or `k`/`j` select, `Home`/`End` jump, `Enter` confirms, `Esc`/`q` cancels |
| Benchmark results | `Enter`, `Esc`, or `q` closes |
| Keyboard reference | `Enter`, `Esc`, `q`, or `?` closes |

## Confirmations and running tasks

Confirmations consistently use:

- `Enter` or `y` to confirm.
- `Esc`, `n`, or `q` to cancel.

While a benchmark is running, all unrelated controls are locked. `Esc`, `q`, `c`, or `Ctrl+C` requests cancellation and server restoration. Completed benchmark cases are retained as a partial run.

## Text editing

Rename and exact-value inputs use standard terminal editing controls:

| Key | Action |
|---|---|
| `Left` / `Right` | Move one character |
| `Ctrl+Left` / `Ctrl+Right` | Move one word |
| `Home`, `Ctrl+A` | Move to the beginning |
| `End`, `Ctrl+E` | Move to the end |
| `Backspace`, `Ctrl+H` | Delete before the cursor |
| `Delete` | Delete at the cursor |
| `Ctrl+W` | Delete the previous word |
| `Ctrl+U` | Delete to the beginning |
| `Ctrl+K` | Delete to the end |
| `Enter` | Confirm input |
| `Esc` | Cancel input |

Printable characters, including `q`, are inserted normally while entering text.
