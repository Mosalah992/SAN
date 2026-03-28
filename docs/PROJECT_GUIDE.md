# Project Guide

## Overview

This repository is a merged project where:

- **Sancta** remains the main application shell
- **Sangpt** is the active local GPT learning/runtime subsystem
- `sancta_gpt` remains the compatibility surface used by the rest of Sancta

Use this guide as the practical onboarding document for day-to-day work.

---

## System Shape

Think of the repo as three connected layers.

### Sancta application layer

This is the outer product surface:

- agent orchestration
- SIEM server
- dashboard
- security pipeline
- trust/routing logic
- knowledge workflows

Main files:

- [backend/sancta.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta.py)
- [backend/agent_loop.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\agent_loop.py)
- [backend/siem_server.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\siem_server.py)

### Sangpt runtime layer

This is the local learning/training subsystem:

- pure-Python GPT runtime
- dataset pipeline
- checkpointed training
- memory manager
- CLI workflow

Main folder:

- [backend/sangpt](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sangpt)

### Adapter and sync layer

This is what makes the merge stable:

- [backend/sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta_gpt.py)
- [backend/sangpt/project_integration.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sangpt\project_integration.py)
- [sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\sancta_gpt.py)

This layer preserves Sancta’s old imports while routing behavior to Sangpt.

---

## Repo Landmarks

### Core integration files

- [backend/sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta_gpt.py)
  Main compatibility adapter.

- [backend/sangpt/sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sangpt\sancta_gpt.py)
  Sangpt engine implementation.

- [backend/sangpt/project_integration.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sangpt\project_integration.py)
  Syncs Sancta knowledge and logs into Sangpt corpora.

- [backend/run_sangpt_cli.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\run_sangpt_cli.py)
  Safe launcher for the Sangpt CLI from the merged repo.

- [sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\sancta_gpt.py)
  Repo-root import shim.

### Planning and merge references

- [mergeplan.md](E:\CODE PROKECTS\merge plan\sancta-merged\docs\mergeplan.md)
- [docs/SANGPT_INTEGRATION_AUDIT.md](E:\CODE PROKECTS\merge plan\sancta-merged\docs\SANGPT_INTEGRATION_AUDIT.md)

---

## First-Time Setup

## 1. Python environment

Use a modern Python version already compatible with Sancta’s requirements.

Recommended:

- Python 3.10+

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional for tests:

```powershell
pip install -r requirements-dev.txt
```

## 2. Environment config

Copy `.env.example` to `.env` if needed, then fill in the values required for the parts of the system you plan to run.

Typical Sancta-related values include:

- Moltbook settings
- local LLM/Ollama settings
- heartbeat/runtime config

## 3. Sanity-check the merge

Run these before deeper work:

```powershell
python -c "import sancta_gpt; print(sancta_gpt.status())"
python -m pytest backend/tests/test_sancta_gpt_smoke.py -q
python backend/run_sangpt_cli.py --help
```

Expected:

- root import works
- backend reports as `sangpt`
- GPT smoke tests pass
- CLI help prints normally

---

## Daily Workflow

## Primary launcher

For this merged repo, the recommended primary operator entrypoint is the Go launcher:

```powershell
cd tools/sancta-launcher
.\sancta-launcher.exe
```

Or in development:

```powershell
cd tools/sancta-launcher
go run .
```

Use the Python launcher mainly as a fallback or reference implementation.

If you want the fastest health snapshot from the merged control surface, use:

```powershell
cd tools/sancta-launcher
go run . status
```

That command now forces a fresh Sangpt probe immediately, so it reports backend, corpus, checkpoint, and training mode without waiting for the background refresh loop.

## If you are working on Sancta behavior

Use this path when the task is about:

- SIEM endpoints
- dashboard behavior
- agent loop
- security pipeline
- trust routing
- knowledge ingestion

Typical commands:

```powershell
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
python -m pytest backend/tests -q
```

## If you are working on Sangpt behavior

Use this path when the task is about:

- training
- checkpoints
- dataset normalization
- memory behavior
- retrieval/generation behavior

Typical commands:

```powershell
python backend/run_sangpt_cli.py
python backend/run_sancta_gpt_training.py 1
```

## If you are working on the integration itself

Use this path when:

- Sancta imports break
- old code expects legacy GPT internals
- training data is not syncing
- paths/manifests/checkpoints behave oddly

Focus files:

- [backend/sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta_gpt.py)
- [backend/sangpt/project_integration.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sangpt\project_integration.py)
- [backend/sangpt/dataset_pipeline.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sangpt\dataset_pipeline.py)

---

## How Learning Works

The merged learning flow is:

1. Sancta produces knowledge, security logs, and operator memory.
2. The sync layer writes selected data into Sangpt dataset files.
3. Sangpt rebuilds its manifest and corpora.
4. The adapter-backed `sancta_gpt` engine trains or retrieves from that merged corpus.

### Current synced sources

Knowledge:

- `knowledge/*.txt`
- `knowledge/*.md`

Security telemetry:

- `logs/security.jsonl`
- `logs/trust_decisions.jsonl`
- `logs/cognitive_outcomes.jsonl`

Operator memory:

- `logs/operator_memory.jsonl`

### Current generated corpus files

- `backend/sangpt/DATA/knowledge/sancta_project_knowledge.txt`
- `backend/sangpt/DATA/security/sancta_live_security.txt`
- `backend/sangpt/DATA/conversational/sancta_operator_memory.txt`

---

## Common Commands

## Start the SIEM server

```powershell
python -m uvicorn backend.siem_server:app --host 127.0.0.1 --port 8787
```

## Run GPT smoke tests

```powershell
python -m pytest backend/tests/test_sancta_gpt_smoke.py -q
```

## Check import and backend status

```powershell
python -c "import sancta_gpt; print(sancta_gpt.status())"
```

## Launch Sangpt CLI

```powershell
python backend/run_sangpt_cli.py
```

Launcher shortcuts:

```powershell
cd tools/sancta-launcher; go run . run sangpt
python backend/sancta_launcher.py run sangpt
```

Note:

- launcher-managed `sangpt` opens in its own interactive console window by design
- it is not meant to stream inside the launcher log pane, because Sangpt requires real terminal input
- the Go launcher still shows live Sangpt engine status from `sancta_gpt.status()` even though the console itself opens separately

## Show Sangpt CLI help

```powershell
python backend/run_sangpt_cli.py --help
```

## Run the legacy GPT training script

```powershell
python backend/run_sancta_gpt_training.py 1
```

Note:

- even very small runs can be slow because the model is pure Python

Launcher shortcuts:

```powershell
cd tools/sancta-launcher; go run . run sangpt-train
python backend/sancta_launcher.py run sangpt-train
```

---

## Validation Checklist

Use this after meaningful changes to the GPT integration.

### Minimum validation

Run:

```powershell
python -c "import sancta_gpt; print(sancta_gpt.status())"
python -m pytest backend/tests/test_sancta_gpt_smoke.py -q
python backend/run_sangpt_cli.py --help
```

Confirm:

- import succeeds from repo root
- backend is `sangpt`
- smoke tests pass
- CLI help exits cleanly

### Integration validation

Run:

```powershell
python -c "import sancta_gpt; e=sancta_gpt.get_engine(); print(e.generate_reply('What did the monitor detect?', use_retrieval=True))"
```

Confirm:

- engine initializes
- corpus exists
- retrieval/generation returns output

### Heavier validation

Run:

```powershell
python backend/run_sancta_gpt_training.py 1
```

Confirm:

- corpus builds
- checkpoint load/save works
- console output stays readable

---

## Current Known-Good Validation

Latest checked results in this merged repo:

- `python -m pytest backend/tests/test_sancta_gpt_smoke.py -q`
  Result: `2 passed, 2 skipped`

- `python backend/run_sangpt_cli.py --help`
  Result: usage output printed cleanly

- `python -c "import sancta_gpt; s=sancta_gpt.status(); ..."`
  Result:
  - `backend=sangpt`
  - `initialized=True`
  - `corpus=7482`

---

## Troubleshooting

## Problem: `import sancta_gpt` fails

Check:

- you are running from the merged repo root
- [sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\sancta_gpt.py) still exists
- [backend/sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta_gpt.py) still imports correctly

## Problem: corpus paths or manifest paths are wrong

Likely cause:

- Sangpt manifest contains stale relative paths

Check:

- [backend/sangpt/dataset_pipeline.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sangpt\dataset_pipeline.py)
- [backend/sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta_gpt.py)

Fix direction:

- force manifest regeneration inside the merged repo

## Problem: old Sancta code expects private GPT fields

Likely cause:

- legacy code uses `_step`, `_docs`, `_checkpoint_path`, `_init_model()`, or similar internals

Check:

- [backend/sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta_gpt.py)

Fix direction:

- preserve compatibility in the adapter rather than rewriting the whole repo at once

## Problem: Sangpt CLI crashes in automation

Likely cause:

- CLI is interactive and is being launched from a non-interactive shell

Use:

```powershell
python backend/run_sangpt_cli.py --help
```

Or run the full CLI from a normal interactive terminal.

## Problem: training is very slow

Likely cause:

- Sangpt is pure Python

What to do:

- keep test runs small
- checkpoint often
- use smoke tests for integration confidence
- avoid treating long training runs as fast feedback loops

## Problem: generated text looks noisy after tiny training runs

Likely cause:

- training step count is too small

Interpretation:

- this is not necessarily a merge bug
- it often just means the model has not trained enough yet

---

## Editing Guidance

Edit these carefully:

- [backend/sancta_gpt.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sancta_gpt.py)
- [backend/sangpt/project_integration.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sangpt\project_integration.py)
- [backend/sangpt/dataset_pipeline.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\sangpt\dataset_pipeline.py)
- [backend/run_sangpt_cli.py](E:\CODE PROKECTS\merge plan\sancta-merged\backend\run_sangpt_cli.py)

Safe rule:

- change Sancta files for application behavior
- change Sangpt files for learning/training behavior
- change adapter/sync files when the two systems disagree

---

## Collaboration Notes

If another person joins the repo, point them to these in order:

1. [docs/PROJECT_GUIDE.md](E:\CODE PROKECTS\merge plan\sancta-merged\docs\PROJECT_GUIDE.md)
2. [mergeplan.md](E:\CODE PROKECTS\merge plan\sancta-merged\docs\mergeplan.md)
3. [docs/SANGPT_INTEGRATION_AUDIT.md](E:\CODE PROKECTS\merge plan\sancta-merged\docs\SANGPT_INTEGRATION_AUDIT.md)

That sequence gives them:

- how to use the repo
- how the merge was designed
- what risks and caveats already exist

---

## Next Improvements

The most useful follow-up work is:

1. Add Sangpt status and checkpoint visibility to the SIEM dashboard.
2. Add integration-specific tests beyond smoke coverage.
3. Improve filtering of live telemetry before it enters training corpora.
4. Decide whether the long-term user-facing naming stays `SanctaGPT` or shifts more explicitly to `Sangpt`.
