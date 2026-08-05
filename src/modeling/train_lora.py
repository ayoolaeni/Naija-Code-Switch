"""
LoRA/QLoRA fine-tuning (README §5.4) via `transformers` + `peft`, driven by
configs/training_config.yaml.

Two ways to run it:

  Real run (needs a GPU + the base model weights):
    python -m src.modeling.train_lora --config configs/training_config.yaml

  Smoke test (proves this code path is correct, runs on CPU in seconds,
  uses a tiny public model instead of Llama/Qwen/Gemma so it needs no GPU
  and no multi-GB download):
    python -m src.modeling.train_lora --smoke-test
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data_utils import build_examples, format_prompt_and_target

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "training_config.yaml"

SMOKE_TEST_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ChatDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = []
        for ex in examples:
            prompt_text, full_text = format_prompt_and_target(tokenizer, ex)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_length]
            if len(full_ids) <= len(prompt_ids):
                continue  # target got truncated away entirely, skip
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
            self.rows.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def make_collate_fn(pad_token_id: int):
    def collate(batch):
        max_len = max(len(r["input_ids"]) for r in batch)
        input_ids, attention_mask, labels = [], [], []
        for r in batch:
            pad_len = max_len - len(r["input_ids"])
            input_ids.append(r["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append([1] * len(r["input_ids"]) + [0] * pad_len)
            labels.append(r["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def load_config(path: Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def run(config: dict, smoke_test: bool = False):
    set_seed(config["training"]["seed"])

    model_id = config["model"]["base_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    quantization_config = None
    if config["model"].get("use_qlora") and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif config["model"].get("use_qlora"):
        print("[train_lora] use_qlora=true but no CUDA GPU available -- falling back to full precision")

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_examples = build_examples(ROOT / config["data"]["train_path"])
    val_examples = build_examples(ROOT / config["data"]["val_path"])
    if smoke_test:
        train_examples = train_examples[:4] or train_examples
        val_examples = val_examples[:2] or val_examples

    max_length = config["training"]["max_seq_length"]
    train_ds = ChatDataset(train_examples, tokenizer, max_length)
    val_ds = ChatDataset(val_examples, tokenizer, max_length) if val_examples else None

    output_dir = ROOT / config["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=1 if smoke_test else config["training"]["num_train_epochs"],
        max_steps=2 if smoke_test else -1,
        learning_rate=config["training"]["learning_rate"],
        logging_steps=1,
        eval_strategy="steps" if (smoke_test and val_ds) else ("epoch" if val_ds else "no"),
        eval_steps=1 if smoke_test else None,
        save_strategy="no" if smoke_test else "epoch",
        load_best_model_at_end=not smoke_test and val_ds is not None,
        metric_for_best_model="eval_loss" if val_ds else None,
        report_to=[],
        seed=config["training"]["seed"],
    )

    callbacks = []
    if val_ds is not None and not smoke_test:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config["training"]["early_stopping_patience"]))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    train_result = trainer.train()

    if not smoke_test:
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"Saved LoRA adapter to {output_dir}")

    # overfitting guardrail (README §5.4): flag verbatim reproduction of
    # training examples in a couple of sample generations.
    _check_verbatim_reproduction(model, tokenizer, train_examples[:3])

    return train_result


def _check_verbatim_reproduction(model, tokenizer, examples):
    model.eval()
    for ex in examples:
        prompt_text, _ = format_prompt_and_target(tokenizer, ex)
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if generated.strip() and generated.strip() in ex.target:
            print(f"[overfitting-guardrail] WARNING: near-verbatim reproduction detected: {generated!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke_test:
        config = json.loads(json.dumps(config))  # deep copy
        config["model"]["base_model_id"] = SMOKE_TEST_MODEL_ID
        config["model"]["use_qlora"] = False
        config["lora"]["target_modules"] = ["q_proj", "v_proj"]
        print(f"[train_lora] SMOKE TEST MODE -- using {SMOKE_TEST_MODEL_ID} instead of {load_config(args.config)['model']['base_model_id']}")

    run(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
