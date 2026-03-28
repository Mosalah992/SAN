# Sancta Launcher (Go)

Go rewrite of `backend/sancta_launcher.py` with:

- **GUI** ([Fyne](https://fyne.io/)): **Services** · **Live log** · **CLI** (embedded REPL, one command per line — same verbs as terminal mode).
- **Terminal**: `sancta-launcher cli` for interactive stdin, or `start` / `status` / `run <svc>`.
- **Merged Sangpt controls**: launcher services for the Sangpt CLI and a bounded Sangpt training run.
- **Live Sangpt engine telemetry** from `sancta_gpt.status()` in `status`, the CLI tab, and the GUI control view.
- **Recommended primary launcher** for the merged Sancta + Sangpt repo.

## Prerequisites

1. **Go 1.21+** — [https://go.dev/dl/](https://go.dev/dl/)  
   The **Cursor/VS Code Go extension does not install the Go SDK**; you need `go` on your PATH (or use the default install under `C:\Program Files\Go\bin`).

   **Winget:** `winget install GoLang.Go`

2. **C compiler (CGO)** — required by Fyne on Windows (`gcc` on PATH).

   - **Winget (LLVM-MinGW UCRT):** `winget install --id MartinStorsjo.LLVM-MinGW.UCRT -e --source winget`  
     If winget says **Waiting for another install/uninstall**, finish or close the other installer, then retry.
   - **MSYS2:** [https://www.msys2.org/](https://www.msys2.org/) → UCRT64 or MINGW64 → `pacman -S mingw-w64-ucrt-x86_64-gcc` (or `mingw-w64-x86_64-gcc`), then add `C:\msys64\ucrt64\bin` or `C:\msys64\mingw64\bin` to **User** PATH.
   - **Other:** TDM-GCC or any MinGW-w64 build as long as `gcc --version` works in PowerShell.

After changing PATH, **open a new terminal** (PATH is read at session start).

### IDE / gopls (Fyne + `go-gl`)

- If the Go language server reports **`build constraints exclude all Go files … [darwin]`** while you are on **Windows**, something (often `GOOS=darwin` for cross-compile) is forcing the wrong target. This repo’s **`.vscode/settings.json`** sets `gopls.build.env.GOOS` to `windows` for the workspace. On **macOS/Linux**, change that value to `darwin` or `linux` in your local settings, or remove the override.
- **Package layout:** `export.go` defines `Run` for all builds; `gui_fyne.go` (`cgo`) and `gui_stub.go` (`!cgo`) register the implementation in `init`. With **CGO disabled** (`CGO_ENABLED=0`), only the stub is linked: `go run .` with no args exits with a hint to install a C compiler or use `sancta-launcher cli`.
- If gopls says **“No packages found”** for `gui_stub.go` while **CGO is enabled**, that file is correctly excluded from the active build. Either ignore the hint, or temporarily set `gopls.build.env.CGO_ENABLED` to `"0"` in `.vscode/settings.json` while editing the stub.

### Windows helper scripts

From `tools/sancta-launcher`:

```powershell
.\setup-windows-build.ps1              # check go + gcc; prepends common paths for this session
.\setup-windows-build.ps1 -Install     # optional: winget install missing Go / LLVM-MinGW
.\build-windows.ps1                    # tidy + build sancta-launcher.exe (no extra CMD window)
.\build-windows.ps1 -NoWindowsGUI       # optional: console subsystem for stdio debugging
```

## Build

**Windows (after prerequisites):** use `.\build-windows.ps1` above (recommended). It passes **`-ldflags "-s -w -H windowsgui"`** so the `.exe` is a **Windows GUI app** and **Explorer won’t open a black CMD window** behind Fyne.

Manual equivalent:

```bash
cd tools/sancta-launcher
go mod tidy
go build -ldflags "-s -w -H windowsgui" -o sancta-launcher.exe .
```

A plain `go build` produces a **console** subsystem binary — Windows attaches a **CMD** to it; that’s expected until you add **`-H windowsgui`**. For a console build on purpose: `.\build-windows.ps1 -NoWindowsGUI` or omit `-H windowsgui`.

**Linux/macOS:**

```bash
cd tools/sancta-launcher
go mod tidy
go build -o sancta-launcher .
```

Run from **repository root** (or any directory under the repo) so `backend/sancta.py` resolves.

## Usage

| Command | Action |
|--------|--------|
| `sancta-launcher` | Open GUI |
| `sancta-launcher gui` | Same |
| `sancta-launcher cli` | Terminal REPL (`sancta> …`) |
| `sancta-launcher start` | Start Ollama → SIEM → Sancta, open browser, log until Ctrl+C |
| `sancta-launcher status` | Snapshot |
| `sancta-launcher run ollama` | Start one service; Ctrl+C to exit |
| `sancta-launcher run sangpt` | Start the merged Sangpt CLI |
| `sancta-launcher run sangpt-train` | Run a bounded Sangpt training job |

**CLI tab / REPL verbs:** `help`, `start [svc]`, `stop [svc]`, `status`, `curiosity`, `phenomenology`, `dashboard`, `exit`.
Valid services include `sangpt` and `sangpt-train`.

The launcher also probes `sancta_gpt.status()` and surfaces Sangpt engine state such as backend, corpus size, step count, checkpoint presence, and training mode.

## Positioning

- Writes **`ROOT/.agent.pid`** when starting Sancta (SIEM Control tab stays aligned).
- **STOP ALL** stops Ollama (child + listener on `:11434` best-effort), same idea as the Python version.
- Log tail: `logs/security.jsonl`, `red_team.jsonl`, `behavioral.jsonl`, `agent_activity.log`.
- **Frozen exe**: resolve paths from executable location (best-effort); dev mode resolves by walking up to `backend/sancta.py`.
- `status` now performs an immediate probe so Sangpt engine details are available without waiting for the background ticker.

The Python launcher is still available as a fallback and reference implementation, but normal service control in the merged repo should go through the Go launcher first.

## Cursor / VS Code (gopls)

1. Install the **Go** extension (Go Team at Google).
2. Command Palette → **Go: Install/Update Tools** → select **gopls** (and others if prompted).
3. If tool install fails, run `go version` in the **same** integrated terminal; if that fails, fix **User PATH** for Go and reload the window.
