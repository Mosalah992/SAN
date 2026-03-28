#!/usr/bin/env python3
"""
Train LoRA adapter on Sancta-style data.

Uses QLoRA (4-bit base + LoRA) for 6GB VRAM compatibility.
Base model: TinyLlama-1.1B-Chat (or Phi-2 with more VRAM).
Output: checkpoints/sancta_lora/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing from sancta root
_SCRIPT_DIR = Path(__file__).resolve().parent
_SANCTA_ROOT = _SCRIPT_DIR.parent
_BACKEND = _SANCTA_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

CHECKPOINT_DIR = _SANCTA_ROOT / "checkpoints" / "sancta_lora"
DEFAULT_DATA_PATH = _SANCTA_ROOT / "lora_train.jsonl"
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _load_dataset(path: Path) -> list[dict]:
    """Load JSONL dataset."""
    examples: list[dict] = []
    if not path.exists():
        return examples
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def _format_chat(example: dict, tokenizer) -> str:
    """Format messages for the model's chat template."""
    messages = example.get("messages", [])
    if not messages:
        return ""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    # Fallback: concat as text
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA on Sancta data (QLoRA)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                        help="Input JSONL from prepare_lora_data.py")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Base model (TinyLlama or Phi-2)")
    parser.add_argument("--output", type=Path, default=CHECKPOINT_DIR,
                        help="Output adapter directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import Dataset
    except ImportError as e:
        print("Install: pip install transformers peft bitsandbytes trl datasets")
        raise SystemExit(1) from e

    examples = _load_dataset(args.data)
    if not examples:
        print(f"No examples in {args.data}. Run prepare_lora_data.py first.")
        raise SystemExit(1)

    print(f"Loaded {len(examples)} examples")

    # 4-bit config for QLoRA (6GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build dataset: format each example as text for causal LM
    def _format_ex(ex):
        text = _format_chat(ex, tokenizer)
        return {"text": text}

    formatted = [_format_ex(ex) for ex in examples]
    dataset = Dataset.from_list(formatted)

    output_dir = args.output
    if not output_dir.is_absolute():
        output_dir = _SANCTA_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        packing=False,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
