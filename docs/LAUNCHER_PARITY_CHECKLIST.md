# Launcher Parity Checklist

## Purpose

This document compares:

- the Python launcher at [backend/sancta_launcher.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta_launcher.py)
- the Go launcher centered on [tools/sancta-launcher/internal/core/manager.go](E:\CODE PROKECTS\merge plan\sancta-merged\tools\sancta-launcher\internal\core\manager.go)

The goal is to track feature parity, known drift, and remaining gaps after the Sancta + Sangpt merge.

---

## Summary

Current state:

- Core process management parity is mostly there.
- Both launchers now understand `sangpt` and `sangpt-train`.
- Both launchers now surface Sangpt engine telemetry from `sancta_gpt.status()`.
- Both support:
  - Ollama
  - SIEM
  - Sancta
  - curiosity
  - phenomenology
  - Sangpt CLI
  - Sangpt training
- The biggest remaining differences are in:
  - Python-only UI richness
  - interactive-process behavior details
  - log filtering/presentation polish
  - CLI ergonomics and feature completeness

---

## Parity Matrix

## Process orchestration

### Ollama start/stop

- Python: yes
  - [backend/sancta_launcher.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta_launcher.py)
- Go: yes
  - [manager.go](E:\CODE PROKECTS\merge plan\sancta-merged\tools\sancta-launcher\internal\core\manager.go)

Status: matched

### SIEM start/stop

- Python: yes
- Go: yes

Status: matched

### Sancta start/stop with restart behavior

- Python: yes
- Go: yes

Status: matched

### Curiosity run

- Python: yes
- Go: yes

Status: matched

### Phenomenology run

- Python: yes
- Go: yes

Status: matched

### Sangpt CLI launch

- Python: yes
  - launches in separate interactive console
- Go: yes
  - launches in separate interactive console on Windows

Status: matched after merge work

### Sangpt training run

- Python: yes
- Go: yes

Status: matched

### Start-all orchestration

- Python: yes
  - Ollama → SIEM → Sancta
- Go: yes
  - same broad sequence

Status: matched

### Stop-all orchestration

- Python: yes
- Go: yes

Status: matched

---

## Status and health visibility

### Net health checks

- Python: yes
- Go: yes

Status: matched

### PID visibility

- Python: yes
- Go: yes

Status: matched

### `.agent.pid` sync for Sancta

- Python: yes
- Go: yes

Status: matched

### Service status snapshot output

- Python: yes
- Go: yes

Status: matched

### Sangpt engine telemetry

- Python: yes
- Go: yes

Status: matched

---

## Log handling

### Tail project log files

- Python: yes
  - `security.jsonl`
  - `red_team.jsonl`
  - `behavioral.jsonl`
  - `agent_activity.log`
- Go: yes
  - same set

Status: matched

### Stream subprocess stdout/stderr

- Python: yes
- Go: yes

Status: matched

### ANSI/control-sequence cleanup

- Python: yes
  - added sanitization during audit
- Go: yes
  - added shared sanitization helper during audit

Status: matched

### Per-source coloring/tagging

- Python: yes
  - richer per-tag text coloring
- Go: partial
  - source separation exists, but the GUI relies more on plain text sections than fine-grained tag styling

Status: Python ahead

### Filterable live-log sources

- Python: yes
  - dropdown filter
- Go: yes
  - source filter in Fyne GUI

Status: matched

### Protection against UI freezes from log flood

- Python: partial
  - batched drain helps
- Go: yes
  - batched log flush path explicitly added in GUI

Status: Go slightly ahead in current implementation clarity

---

## GUI behavior

### Native GUI

- Python: yes
  - Tkinter control center
- Go: yes
  - Fyne GUI

Status: matched

### Embedded CLI tab

- Python: no true embedded REPL tab
  - Python has terminal CLI mode, not a GUI-embedded REPL
- Go: yes
  - embedded CLI tab exists

Status: Go ahead

### Terminal-style presentation

- Python: partial
  - cyber/terminal look, but older Tkinter layout
- Go: partial
  - dark theme and terminal-oriented labels added, but still visually simpler than a full terminal dashboard

Status: both partial

### Rich session panel

- Python: yes
  - uptime, restarts, model field, denser control panel
- Go: partial
  - simpler status presentation

Status: Python ahead

### Per-service control density

- Python: yes
  - denser left-side operations panel
- Go: partial
  - functional but simpler

Status: Python ahead

---

## CLI behavior

### Interactive REPL

- Python: yes
- Go: yes

Status: matched

### Help output

- Python: yes
- Go: yes

Status: matched

### Service verbs

- Python: yes
- Go: yes

Status: matched

### Sangpt-aware service names

- Python: yes
- Go: yes

Status: matched

---

## Path resolution and packaging

### Dev-mode path resolution

- Python: yes
- Go: yes

Status: matched

### Frozen/binary path handling

- Python: yes
- Go: yes

Status: matched in intent

### Windows GUI executable flow

- Python: yes
  - PyInstaller path still supported
- Go: yes
  - `sancta-launcher.exe` build flow supported

Status: matched

---

## Current Gaps

## Python launcher still ahead in

1. Richer left-panel control-center presentation.
2. More explicit session stats and operational framing.
3. More granular text-tag styling in the live log.
4. Slightly more mature operator-facing control-center feel.

## Go launcher still ahead in

1. Embedded GUI CLI tab.
2. Cleaner separation between manager, GUI, CLI, and path logic.
3. Better long-term maintainability as a compiled launcher.

## Shared gaps

1. Neither launcher yet has a truly polished “terminal dashboard” aesthetic.
2. Neither launcher exposes deep Sangpt corpus/checkpoint status in the UI.
3. Neither launcher yet surfaces:
   - last Sangpt checkpoint
   - corpus size
   - last training loss
   - training mode
   - sync freshness

---

## Recommended Next Work

## High-value parity work

1. Add Sangpt status telemetry to both launchers:
   - backend
   - corpus size
   - last loss
   - checkpoint presence
   - training mode

2. Make service labels explicit:
   - `Open Sangpt Console`
   - `Run Sangpt Training`

3. Add a small launcher-side status query for `sancta_gpt.status()` and show it in both UIs.

## Visual improvement work

1. Make the Go GUI look more like a terminal dashboard:
   - stronger monospace layout
   - tighter grid alignment
   - clearer section headers
   - more deliberate contrast
   - compact service rows

2. Clean up Python launcher encoding artifacts in visible labels.

3. Normalize terminology between both launchers so the same services are named the same way.

## Reliability work

1. Add automated smoke checks for launcher commands:
   - `status`
   - `run sangpt --help`
   - `run sangpt-train --help`

2. Add tests or scripted checks around log sanitization.

3. Add explicit detection of interactive-only services so they are always launched correctly.

---

## Quick Verdict

If the question is:

“Is the Go launcher functionally integrated with the Python launcher’s responsibilities?”

The answer is:

- yes for core service management
- yes for the new Sangpt service entries
- mostly yes for log handling
- not fully yet for UI richness and operator polish

If the question is:

“Can the Go launcher replace the Python launcher for normal service control?”

The answer is:

- mostly yes
- but the Python launcher still has a slightly richer control-center feel
- and both launchers still have room to improve around deep Sangpt visibility
