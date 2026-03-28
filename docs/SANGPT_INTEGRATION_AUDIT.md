# Sangpt Integration Audit

## Key findings

- `sancta-main` was already in a dirty git state, so merging in place risked overwriting unrelated work. The merged build was staged separately in `merge plan/sancta-merged`.
- Sancta and Sangpt both define `sancta_gpt.py`. That name collision was the main integration hazard; the merged project now keeps Sangpt under `backend/sangpt/` and exposes it through a compatibility adapter at `backend/sancta_gpt.py`.
- Sangpt claimed "standard library only" in `REQUIREMENTS.txt`, but `sancta_bridge.py` imports LangChain types. That file remains optional and is not on the critical runtime path.
- Sangpt stored runtime artifacts inside the project tree (`.venv`, `__pycache__`, `logs`, `checkpoints`, SQLite DB). Those were not copied wholesale into the merged repo to avoid source-tree bloat and stale state.
- Both projects relied on relative paths like `./DATA` and `./logs`. The merged adapter anchors paths from the repo/backend location so training and checkpoints resolve consistently.

## Remaining risks

- Sangpt's CLI and training loop are pure-Python and can still be slow on large corpora.
- Existing tests written for Sancta's old internal GPT implementation may need minor updates if they assert old private field semantics rather than behavior.
- The merged project now syncs live Sancta logs into Sangpt corpora, but the quality of ongoing learning still depends on how clean those logs remain over time.
