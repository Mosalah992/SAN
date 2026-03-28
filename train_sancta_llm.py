"""
train_sancta_llm.py — Fine-tune Llama 3.2 1B as Sancta security analyst.

Native Windows compatible — no WSL, no Axolotl needed.
Uses: transformers + peft (QLoRA) + trl (SFTTrainer)

Hardware target: NVIDIA 1660Ti 6GB VRAM
  - 4-bit QLoRA quantization
  - fp16 training (Turing arch, no bf16)
  - gradient checkpointing
  - micro_batch=1, grad_accum=8

Usage:
    python train_sancta_llm.py                          # train with defaults
    python train_sancta_llm.py --epochs 5 --merge --export-gguf   # train, merge, GGUF
    python train_sancta_llm.py --merge-only             # merge LoRA to full HF weights
    python train_sancta_llm.py --export-only            # GGUF only (merge first if needed)
    python train_sancta_llm.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0  # smaller base
    python train_sancta_llm.py --base-model TinyLlama/... --fp16-lora --seq-len 384  # if 4-bit load hangs (Win/py3.13)

Requires HF access token in env for gated models:  set HF_TOKEN=...  or  huggingface-cli login
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "training_corpus.jsonl"
OUTPUT_DIR = ROOT / "sancta-lora-out"
MERGED_DIR = OUTPUT_DIR / "merged"
DEFAULT_GGUF_NAME = "sancta-analyst-q4_k_m.gguf"
# Hugging Face base for QLoRA (matches Ollama tag `llama3.2` family; Ollama name after fine-tune: sancta-analyst)
_DEFAULT_HF_BASE = "meta-llama/Llama-3.2-1B-Instruct"


def _load_dotenv() -> None:
    """Load repo .env so SANCTA_BASE_MODEL / HF_TOKEN apply without exporting in shell."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def setup_huggingface_auth() -> None:
    """
    Load .env, then log in to the Hub if HF_TOKEN / HUGGING_FACE_HUB_TOKEN is set.
    Without this, gated models (meta-llama/*) return 401 even when the token is only in .env.
    """
    _load_dotenv()
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not tok:
        return
    try:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
        print("Hugging Face: authenticated via HF_TOKEN (from environment / .env).")
    except Exception as exc:
        print(f"WARNING: huggingface_hub.login failed: {exc}")


def _hf_download_token():
    """
    Token for from_pretrained / Hub downloads.
    - Use HF_TOKEN from env when set.
    - Else use token from `huggingface-cli login` if present.
    - Else False = anonymous (works for public models like TinyLlama).
    Note: token=True without a stored token raises LocalTokenNotFoundError on recent huggingface_hub.
    """
    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if t:
        return t
    try:
        from huggingface_hub import get_token
        cached = get_token()
        if cached:
            return cached
    except Exception:
        pass
    return False


def assert_hf_access_for_gated_model(model_id: str) -> None:
    """Exit with instructions if meta-llama/* is used without any Hub token."""
    if not model_id.lower().startswith("meta-llama/"):
        return
    try:
        from huggingface_hub import get_token
    except ImportError:
        return
    if get_token():
        return
    print(
        "\nERROR: Gated Hugging Face model — not logged in.\n"
        f"  Model: {model_id}\n\n"
        "Fix (pick one):\n"
        "  • Request access + accept the license:\n"
        "      https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct\n"
        "  • Add a read token to .env (then re-run this script):\n"
        "      HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
        "    Create tokens: https://huggingface.co/settings/tokens\n"
        "  • Or log in once in the terminal:\n"
        "      huggingface-cli login\n"
        "  • Or use an open base model (no HF gate):\n"
        "      python train_sancta_llm.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0\n",
        file=sys.stderr,
    )
    sys.exit(1)


TRAIN_MANIFEST_NAME = "train_manifest.json"


def _read_train_manifest() -> dict:
    p = OUTPUT_DIR / TRAIN_MANIFEST_NAME
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_train_manifest(base_model: str, fp16_lora: bool) -> None:
    try:
        p = OUTPUT_DIR / TRAIN_MANIFEST_NAME
        p.write_text(
            json.dumps({"base_model": base_model, "fp16_lora": fp16_lora}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _lora_adapter_ready() -> bool:
    d = OUTPUT_DIR
    if not d.is_dir():
        return False
    if (d / "adapter_config.json").exists():
        return True
    return any(d.glob("adapter_model.*"))


def check_prerequisites(need_corpus: bool = True) -> int:
    """Verify CUDA PyTorch and required packages. Optionally require JSONL corpus."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: PyTorch CUDA not available. Install with:")
            print("  pip3 install torch --index-url https://download.pytorch.org/whl/cu124")
            sys.exit(1)
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({vram:.1f} GB)")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    except ImportError:
        print("ERROR: PyTorch not installed")
        sys.exit(1)

    missing = []
    for pkg in ["transformers", "peft", "trl", "datasets", "bitsandbytes", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"  pip3 install {' '.join(missing)}")
        sys.exit(1)

    if need_corpus:
        if not CORPUS_PATH.exists():
            print(f"ERROR: Training corpus not found at {CORPUS_PATH}")
            print("  Run: python backend/build_training_corpus.py")
            sys.exit(1)

        with open(CORPUS_PATH, encoding="utf-8") as _cf:
            n_pairs = sum(1 for _ in _cf)
        print(f"Corpus: {n_pairs} training pairs ({CORPUS_PATH.stat().st_size / 1024:.0f} KB)")
        return n_pairs
    return 0


def load_dataset_from_jsonl():
    """Load ChatML JSONL corpus into HuggingFace Dataset."""
    from datasets import Dataset

    records = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                # Convert conversations list to a single formatted text
                convos = obj.get("conversations", [])
                if convos:
                    records.append({"conversations": convos})
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"Loaded {len(records)} conversation pairs", flush=True)
    return Dataset.from_list(records)


def format_chatml(example):
    """Format a conversation into ChatML template for training."""
    text_parts = []
    for msg in example["conversations"]:
        role = msg["role"]
        content = msg["content"]
        text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    text_parts.append("<|im_start|>assistant\n")  # prompt for generation
    # Remove the last assistant prompt since the actual response is already included
    full_text = "\n".join(text_parts[:-1])
    return {"text": full_text}


def _start_load_heartbeat(log, interval_sec: float = 15.0):
    """Print periodic lines while from_pretrained runs (no native progress from HF)."""
    import threading

    done = threading.Event()

    def _beat():
        n = 0
        while True:
            if done.wait(timeout=interval_sec):
                break
            n += 1
            elapsed = n * interval_sec
            log(
                f"  ... still loading ({elapsed:.0f}s) — downloading shards to cache or moving "
                f"weights to GPU; first run is slow. Watch disk/network in Task Manager.",
            )

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    return done


def train(args):
    """Run QLoRA fine-tuning (4-bit) or fp16 LoRA (--fp16-lora, no bitsandbytes)."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    def log(msg: str) -> None:
        print(msg, flush=True)

    fp16_lora = bool(getattr(args, "fp16_lora", False))

    log(f"\n{'='*60}")
    log(f"  Sancta LLM Fine-Tuning")
    log(f"  Model: {args.base_model}")
    log(f"  Mode:  {'fp16 LoRA (no 4-bit)' if fp16_lora else 'QLoRA 4-bit'}")
    log(f"  Epochs: {args.epochs}" + (f"  (max_steps cap: {args.max_steps})" if args.max_steps > 0 else ""))
    log(f"  Output: {OUTPUT_DIR}")
    log(f"{'='*60}\n")

    if sys.version_info >= (3, 13) and not fp16_lora:
        log(
            "NOTE: Python 3.13 + Windows + bitsandbytes 4-bit often hangs at 'Loading model'. "
            "If stuck >30 min, Ctrl+C and re-run with:  --fp16-lora  (and consider --seq-len 384 if OOM).",
        )

    # ── Load tokenizer ──────────────────────────────────────────────────
    tok = _hf_download_token()
    log("Loading tokenizer...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        token=tok,
    )
    log(f"  tokenizer OK ({time.perf_counter() - t0:.1f}s)")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Single-GPU map avoids slow / stuck "auto" CPU offload heuristics on some Windows setups
    device_map = {"": 0} if torch.cuda.is_available() else "auto"

    t1 = time.perf_counter()
    hb = _start_load_heartbeat(log, 15.0)
    try:
        if fp16_lora:
            log(
                "Loading model in fp16 on GPU (skips bitsandbytes — fixes overnight hangs on some Win/py3.13 setups)...",
            )
            log(
                "  (Silent stretches are normal; heartbeat every 15s. To log to a file for a 2nd window, see docs/FINETUNE_GUIDE.md.)",
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                device_map=device_map,
                trust_remote_code=True,
                dtype=torch.float16,
                token=tok,
                low_cpu_mem_usage=True,
            )
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            log(
                "Loading model in 4-bit (first time: several minutes from disk + quantize; "
                "GPU fans should spin — check Task Manager / nvidia-smi)...",
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                dtype=torch.float16,
                token=tok,
                low_cpu_mem_usage=True,
            )
            model = prepare_model_for_kbit_training(model)
    finally:
        hb.set()

    log(f"  weights loaded ({time.perf_counter() - t1:.1f}s)")

    # ── LoRA config ─────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    log(f"Parameters: {total:,} total, {trainable:,} trainable "
        f"({100 * trainable / total:.1f}%)")

    # ── Dataset ─────────────────────────────────────────────────────────
    log("Tokenizing corpus (Windows: single-process map avoids multiprocessing hangs)...")
    t2 = time.perf_counter()
    dataset = load_dataset_from_jsonl()
    dataset = dataset.map(
        format_chatml,
        remove_columns=["conversations"],
        num_proc=1,
    )
    log(f"  dataset ready: {len(dataset)} rows ({time.perf_counter() - t2:.1f}s)")

    ms = args.max_steps if getattr(args, "max_steps", 0) and args.max_steps > 0 else -1

    # ── SFT Config (tuned for 6GB VRAM) ─────────────────────────────────
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        max_steps=ms,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,                          # Turing (1660Ti) — no bf16
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,        # save RAM on Windows
        max_length=args.seq_len,            # TRL 0.29+ (replaces max_seq_length)
        dataset_text_field="text",
        packing=False,                      # disable packing to save VRAM
    )

    # ── Trainer ─────────────────────────────────────────────────────────
    log("Building SFTTrainer — first forward can take 1–3+ min on 6GB; loss lines follow.\n")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    log("\nTraining loop finished.")

    # ── Save LoRA adapter ───────────────────────────────────────────────
    log(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    log("Training complete!")
    _write_train_manifest(args.base_model, fp16_lora)

    return model, tokenizer


def merge_and_export(args):
    """Merge LoRA adapter into base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\nMerging LoRA adapter into base model...")

    # Load base model (full precision for merge)
    tok = _hf_download_token()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=tok,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="cpu",  # merge on CPU to avoid VRAM limits
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=tok,
    )

    # Load and merge LoRA
    model = PeftModel.from_pretrained(model, str(OUTPUT_DIR))
    model = model.merge_and_unload()

    # Save merged model
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"Merged model saved to {MERGED_DIR}")

    return MERGED_DIR


def export_gguf(args) -> int:
    """Convert merged HF weights to GGUF via llama.cpp (for Ollama)."""
    import subprocess

    llama_root = Path(os.environ.get("LLAMA_CPP_ROOT", str(ROOT / "llama.cpp")))
    converter = llama_root / "convert_hf_to_gguf.py"
    if not converter.is_file():
        print(f"ERROR: llama.cpp converter not found at:\n  {converter}")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  Or set LLAMA_CPP_ROOT to your llama.cpp checkout.")
        return 1

    if not (MERGED_DIR / "config.json").is_file():
        print(f"ERROR: Merged model missing. Expected: {MERGED_DIR / 'config.json'}")
        print("  Run: python train_sancta_llm.py --merge-only")
        return 1

    out_name = args.gguf_out or DEFAULT_GGUF_NAME
    out_gguf = Path(out_name) if os.path.isabs(out_name) else ROOT / out_name

    cmd = [
        sys.executable,
        str(converter),
        str(MERGED_DIR),
        "--outfile",
        str(out_gguf),
        "--outtype",
        "q4_k_m",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nGGUF saved: {out_gguf}")
    print("  Ollama: create a Modelfile with FROM pointing to this file, then:")
    print('  ollama create sancta-analyst -f Modelfile')
    return 0


def test_generation(args):
    """Quick test: generate a few samples from the trained model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("\n--- Test Generation ---", flush=True)

    from peft import PeftModel

    if not _lora_adapter_ready():
        print(f"ERROR: LoRA adapter not found under {OUTPUT_DIR}")
        print("  Train first, or copy adapter files into that directory.")
        return

    manifest = _read_train_manifest()
    fp16_lora = bool(getattr(args, "fp16_lora", False) or manifest.get("fp16_lora"))

    tok = _hf_download_token()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=tok,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _dm = {"": 0} if torch.cuda.is_available() else "auto"
    if fp16_lora:
        print("(Loading base in fp16 to match --fp16-lora / train_manifest.json)", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map=_dm,
            dtype=torch.float16,
            trust_remote_code=True,
            token=tok,
            low_cpu_mem_usage=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map=_dm,
            dtype=torch.float16,
            trust_remote_code=True,
            token=tok,
            low_cpu_mem_usage=True,
        )
    model = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
    model.eval()

    prompts = [
        "What did the latest security scan find?",
        "Who are you?",
        "Explain the SEIR epidemic model.",
        "How do you handle prompt injection?",
    ]

    device = next(model.parameters()).device
    for prompt in prompts:
        chat = f"<|im_start|>system\nYou are Sancta, an autonomous AI security analyst.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(chat, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nQ: {prompt}")
        print(f"A: {response[:300]}")


def main():
    # Line-buffered stdout so Windows PowerShell shows progress instead of "stuck for hours"
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    setup_huggingface_auth()
    parser = argparse.ArgumentParser(description="Fine-tune Sancta LLM")
    parser.add_argument(
        "--base-model",
        default=None,
        help=f"HuggingFace model ID (default: env SANCTA_BASE_MODEL or {_DEFAULT_HF_BASE})",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after N optimizer steps (debug/smoke). -1 = use epochs only.",
    )
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Max sequence length (reduce if OOM); passed as SFTConfig max_length")
    parser.add_argument(
        "--fp16-lora",
        action="store_true",
        help="Train LoRA in fp16 without bitsandbytes 4-bit (use if 4-bit load hangs overnight on Windows/Python 3.13)",
    )
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA into base model after training")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing LoRA in sancta-lora-out (skip training)")
    parser.add_argument("--test", action="store_true",
                        help="Run test generation after training")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run test generation (skip training)")
    parser.add_argument("--export-gguf", action="store_true",
                        help="After training/merge, convert merged HF model to Q4_K_M GGUF")
    parser.add_argument("--export-only", action="store_true",
                        help="Only run GGUF export (merge first if needed); skips training")
    parser.add_argument("--gguf-out", default="",
                        help=f"GGUF output path (default: ./{DEFAULT_GGUF_NAME})")
    args = parser.parse_args()

    if args.base_model is None:
        args.base_model = os.environ.get("SANCTA_BASE_MODEL", _DEFAULT_HF_BASE).strip()
    if not args.base_model:
        args.base_model = _DEFAULT_HF_BASE

    if args.export_only and args.test_only:
        print("ERROR: Use either --export-only or --test-only, not both.")
        sys.exit(1)

    if args.test_only:
        assert_hf_access_for_gated_model(args.base_model)
        check_prerequisites(need_corpus=False)
        test_generation(args)
        return

    if args.merge_only:
        assert_hf_access_for_gated_model(args.base_model)
        check_prerequisites(need_corpus=False)
        if not _lora_adapter_ready():
            print(f"ERROR: No LoRA adapter in {OUTPUT_DIR}. Train first.")
            sys.exit(1)
        merge_and_export(args)
        if args.export_gguf:
            raise SystemExit(export_gguf(args))
        return

    if args.export_only:
        check_prerequisites(need_corpus=False)
        if not (MERGED_DIR / "config.json").is_file():
            if not _lora_adapter_ready():
                print("ERROR: Nothing to export. Train first or run --merge-only.")
                sys.exit(1)
            print("Merged folder missing; merging LoRA before GGUF export...")
            merge_and_export(args)
        raise SystemExit(export_gguf(args))

    check_prerequisites(need_corpus=True)
    assert_hf_access_for_gated_model(args.base_model)

    model, tokenizer = train(args)

    if args.test:
        test_generation(args)

    if args.merge:
        merge_and_export(args)

    if args.export_gguf:
        if not (MERGED_DIR / "config.json").is_file():
            print("Merged model not found; merging now...")
            merge_and_export(args)
        raise SystemExit(export_gguf(args))

    print(f"\n{'='*60}")
    print("  DONE! Next steps:")
    print("  1. Merge:   python train_sancta_llm.py --merge-only")
    print("  2. Test:    python train_sancta_llm.py --test-only")
    print("  3. GGUF:    python train_sancta_llm.py --export-gguf")
    print("  4. Ollama:  docs/FINETUNE_GUIDE.md (Modelfile + ollama create)")
    print("  5. Sancta:  USE_LOCAL_LLM=true  LOCAL_MODEL=sancta-analyst  in .env")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
