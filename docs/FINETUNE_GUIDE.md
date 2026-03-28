# Sancta GPT Fine-Tuning Guide

Fine-tune Llama 3.2 1B (or another chat base) with Sancta's security corpus using **QLoRA**.

Two paths:

| Path | Platform | Tooling |
|------|----------|---------|
| **Native Windows** | Windows 10/11 + NVIDIA GPU | `train_sancta_llm.py` — `transformers` + `peft` + `trl` (no Axolotl) |
| **Axolotl** | Linux or WSL2 | `axolotl train axolotl_sancta.yml` |

Target hardware: NVIDIA 1660 Ti 6 GB VRAM (also works on any 6GB+ GPU).

---

## Native Windows: `train_sancta_llm.py` (recommended on Windows)

No WSL required. Uses 4-bit QLoRA, fp16 (Turing), gradient checkpointing, `micro_batch=1`, `grad_accum=8`.

### 0. Prerequisites

```powershell
# CUDA PyTorch (example: cu124)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-training.txt
```

- **Llama 3.2 (`meta-llama/*`) is gated**: open the model page on Hugging Face, click **Agree** on the license, then either:
  - add **`HF_TOKEN=hf_...`** to `.env` (create at [HF tokens](https://huggingface.co/settings/tokens)), or  
  - run **`huggingface-cli login`** once.  
  `train_sancta_llm.py` calls `huggingface_hub.login()` from `HF_TOKEN` after loading `.env`, so the token is actually used for downloads.
- **Base checkpoint**: set `SANCTA_BASE_MODEL` in `.env` (default `meta-llama/Llama-3.2-1B-Instruct`). `train_sancta_llm.py` loads `.env` via `python-dotenv`.
- Verify GPU: `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"`
- **One-shot (Windows)**: `.\train_sancta.ps1` — rebuilds corpus, trains, merges, exports GGUF (needs `LLAMA_CPP_ROOT` + llama.cpp for GGUF).

### 1. Build corpus (same as below)

```powershell
python backend/build_training_corpus.py
```

Output: `training_corpus.jsonl` at repo root.

### 2. Train

```powershell
python train_sancta_llm.py
python train_sancta_llm.py --epochs 5 --seq-len 384
# Smaller download for tests:
python train_sancta_llm.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Artifacts: `sancta-lora-out/` (LoRA adapter + tokenizer).

**TRL 0.29+**: training uses `SFTConfig` with `max_length` (mapped from `--seq-len`).

**“Tail” / follow logs (PowerShell):** You cannot attach `tail -f` to another process’s live stdout. Either watch the same terminal, or tee to a file and tail the file:

```powershell
# Terminal A — unbuffered + copy to log (still prints here)
python -u train_sancta_llm.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --fp16-lora 2>&1 | Tee-Object -FilePath sancta_train.log

# Terminal B — follow the log
Get-Content .\sancta_train.log -Wait -Tail 40
```

Avoid a stray `>>` on its own line after the command — in PowerShell that is the **secondary prompt** (often means an unfinished string or `(`), not “append”.

The training script prints a **heartbeat every 15s** while the model is loading so long silent phases are easier to interpret.

### 3. Merge LoRA (full HF weights for GGUF)

```powershell
python train_sancta_llm.py --merge-only
```

Output: `sancta-lora-out/merged/`

### 4. Test adapter (optional, 4-bit + LoRA)

```powershell
python train_sancta_llm.py --test-only
```

### 5. Export GGUF + Ollama

Clone [llama.cpp](https://github.com/ggerganov/llama.cpp) (or set `LLAMA_CPP_ROOT`).

```powershell
set LLAMA_CPP_ROOT=C:\path\to\llama.cpp
python train_sancta_llm.py --export-only
# custom name:
python train_sancta_llm.py --export-only --gguf-out .\my-sancta.gguf
```

Then follow **Step 5** below (`Modelfile` + `ollama create`).

### One-shot after training

```powershell
python train_sancta_llm.py --epochs 3 --merge --export-gguf
```

---

## Prerequisites (Axolotl / WSL path)

**Important**: Axolotl + flash-attn is easiest on Linux. If you use Axolotl on Windows, use WSL2:
```bash
wsl --install                  # if not already set up
wsl                            # enter WSL
```

Install inside WSL (or native Linux):
```bash
# CUDA toolkit (if not installed)
# Check: nvidia-smi

# Core packages
pip3 install -U packaging setuptools wheel ninja

# Axolotl WITHOUT flash-attn (1660Ti doesn't support it)
pip3 install --no-build-isolation axolotl

# Quantization support
pip3 install bitsandbytes>=0.43

# For GGUF export later
pip3 install llama-cpp-python
```

---

## Step 1: Build Training Corpus

```bash
cd /mnt/e/CODE\ PROKECTS/sancta-main/sancta-main   # adjust to your path
python backend/build_training_corpus.py
```

This scans:
- `SOUL_SYSTEM_PROMPT.md` (persona — baked into every training pair)
- `SECURITY_SEEDS` from `sancta_gpt.py` (115 security analysis examples)
- `CONVERSATIONAL_ENGLISH_SEEDS` (40 dialogue examples)
- `knowledge/*.txt` files (OWASP, zero trust, ATLAS, etc.)
- `logs/security.jsonl` (real blocked injection events)
- `logs/red_team.jsonl` (red team scan results)

Output: `training_corpus.jsonl` (~650 pairs, ~1.7MB)

To add more training data, either:
- Drop `.txt` files into `knowledge/` and re-run the script
- Or manually add ChatML JSONL pairs to `training_corpus.jsonl`

---

## Step 2: Fine-Tune with Axolotl

```bash
# Single GPU (your 1660Ti)
axolotl train axolotl_sancta.yml
```

Expected:
- ~649 pairs x 3 epochs = ~1947 training steps
- ~15-30 min on 1660Ti (fp16, QLoRA, micro_batch=1)
- VRAM usage: ~4-5GB (leaves headroom)
- Output: `./sancta-lora-out/` with LoRA adapter weights

If you get OOM errors:
1. Reduce `sequence_len` to 384 in `axolotl_sancta.yml`
2. Reduce `gradient_accumulation_steps` to 4
3. Make sure no other GPU-using process is running

---

## Step 3: Merge LoRA into Base Model

```bash
# Merge adapter weights back into the base model
axolotl merge_lora axolotl_sancta.yml --lora-model-dir ./sancta-lora-out
```

This creates a full merged model in `./sancta-lora-out/merged/`.

---

## Step 4: Convert to GGUF for Ollama

```bash
# Clone llama.cpp converter (if not already)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert merged model to GGUF (Q4_K_M quantization — good quality, small size)
python convert_hf_to_gguf.py ../sancta-lora-out/merged/ \
    --outfile sancta-analyst.gguf \
    --outtype q4_k_m
```

---

## Step 5: Import into Ollama

Create a Modelfile:
```bash
cat > Modelfile << 'EOF'
FROM ./sancta-analyst.gguf

SYSTEM "You are Sancta, an autonomous AI security analyst operating on Moltbook. You monitor the threat landscape, track adversarial behavior through a 5-layer security pipeline, and publish findings with technical precision. Evidence first. Specifics over vague warnings. Threat models over ghost stories."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
EOF

# Create the Ollama model
ollama create sancta-analyst -f Modelfile

# Test it
ollama run sancta-analyst "What did the latest security scan find?"
```

---

## Step 6: Wire into Sancta

In your `.env` file:
```
USE_LOCAL_LLM=true
LOCAL_MODEL=sancta-analyst
```

That's it. The existing fallback chain in `siem_server.py` routes through Ollama
first. The Chat tab, post generator, and reply handler all pick up the new model
automatically. No code changes needed.

---

## Retraining Cycle

As Sancta accumulates more security events and knowledge:

**Native Windows (`train_sancta_llm.py`):**
```powershell
python backend/build_training_corpus.py
python train_sancta_llm.py --epochs 3 --merge --export-gguf
ollama create sancta-analyst -f Modelfile
```

**Axolotl (Linux / WSL):**
```bash
python backend/build_training_corpus.py
axolotl train axolotl_sancta.yml
axolotl merge_lora axolotl_sancta.yml --lora-model-dir ./sancta-lora-out
cd llama.cpp && python convert_hf_to_gguf.py ../sancta-lora-out/merged/ \
    --outfile sancta-analyst.gguf --outtype q4_k_m
ollama create sancta-analyst -f Modelfile
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM during training | Reduce `sequence_len` to 384 or `gradient_accumulation_steps` to 4 |
| flash-attn error | Config already disables it (`flash_attention: false`). Make sure `xformers` is not installed |
| bitsandbytes error on Windows | Try native `train_sancta_llm.py` + CUDA PyTorch; if it still fails, use WSL2 |
| Model outputs generic text | Add more identity/security pairs to corpus and retrain |
| Ollama doesn't find model | Run `ollama list` to verify. Re-run `ollama create` if needed |
| bf16 error | Config uses `fp16: true` for 1660Ti compatibility. Don't change to bf16 |
| PowerShell `>>` errors | `>>` is not a line continuation in PowerShell. Run `cd ...` on one line, then `python train_sancta_llm.py ...` on the next (or use `` ` `` at end of line for continuation). |
| No new output for hours | Often **output buffering** or a long **4-bit load**. Use `python -u train_sancta_llm.py ...` or the script now line-buffers. Watch **GPU %** in Task Manager; run `--max-steps 20` once to verify the loop. |
| Stuck after "Loading model" (hours / overnight) | **Ctrl+C** and re-run with **`--fp16-lora`** (skips bitsandbytes 4-bit; fits TinyLlama 1.1B on 6GB with `--seq-len 384` if OOM). Common on **Windows + Python 3.13** where 4-bit init can hang. Otherwise first 4-bit load can take **10–30+ minutes** on slow disk. |
