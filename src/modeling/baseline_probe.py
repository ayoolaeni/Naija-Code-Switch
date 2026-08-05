"""
Model-selection comparison (README §5.2): for each of the three candidate
families (Llama 3 / Qwen2.5 / Gemma 2), measure:
  1. Baseline zero/few-shot competence on a validation sample (via the HF
     Inference API -- needs HF_TOKEN and access to each model).
  2. Tokenizer fragmentation on a fixed Pidgin sample (works offline once
     tokenizer files are cached, no model weights needed).
  3. Ecosystem/license notes (recorded statically below; verify current
     terms before relying on them).

Produces a markdown comparison table. Any step that needs network/HF access
that isn't available degrades gracefully (reports "unavailable") rather than
crashing the whole comparison.

Run: python -m src.modeling.baseline_probe
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .data_utils import build_examples
from .prompt_design import get_frozen_system_prompt

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
VAL_PATH = ROOT / "data" / "splits" / "val.jsonl"

CANDIDATES = {
    "Llama 3": {
        "tokenizer_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "inference_model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "license": "Llama 3 Community License (permits research use + result reporting; custom, not OSI-approved)",
    },
    "Qwen2.5": {
        "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
        "inference_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "license": "Apache 2.0 (permissive)",
    },
    "Gemma 2": {
        "tokenizer_id": "google/gemma-2-9b-it",
        "inference_model_id": "google/gemma-2-9b-it",
        "license": "Gemma Terms of Use (permits research use + result reporting; custom, not OSI-approved)",
    },
}

# Fixed Pidgin sample for fragmentation measurement (README §5.2 point 3).
PIDGIN_SAMPLE = (
    "Abeg help me check my account balance, I don tire to dey wait for reply. "
    "Wetin dey happen na, make we settle this matter sharp sharp before I vex."
)


def measure_tokenizer_fragmentation(tokenizer_id: str) -> dict:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as e:  # noqa: BLE001 - best-effort, gated/offline models
        return {"status": "unavailable", "error": str(e)}

    words = PIDGIN_SAMPLE.split()
    token_ids = tok(PIDGIN_SAMPLE, add_special_tokens=False)["input_ids"]
    return {
        "status": "ok",
        "n_words": len(words),
        "n_tokens": len(token_ids),
        "tokens_per_word": round(len(token_ids) / len(words), 2),
    }


def probe_hf_inference(model_id: str, system_prompt: str, val_examples, max_examples: int = 3) -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"status": "unavailable", "reason": "HF_TOKEN not set"}
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=model_id, token=token)
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "error": str(e)}

    results = []
    for ex in val_examples[:max_examples]:
        messages = [{"role": "system", "content": system_prompt}] + ex.messages[1:]
        try:
            response = client.chat_completion(messages=messages, max_tokens=128)
            generated = response.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - gated model, quota, etc.
            return {"status": "error", "error": str(e)}
        results.append({"prompt_tail": ex.messages[-1]["content"], "reference": ex.target, "generated": generated})
    return {"status": "ok", "samples": results}


def run() -> list[dict]:
    system_prompt = get_frozen_system_prompt()
    val_examples = build_examples(VAL_PATH, system_prompt) if VAL_PATH.exists() else []

    report = []
    for family, spec in CANDIDATES.items():
        row = {
            "family": family,
            "license": spec["license"],
            "tokenizer_fragmentation": measure_tokenizer_fragmentation(spec["tokenizer_id"]),
            "baseline_probe": probe_hf_inference(spec["inference_model_id"], system_prompt, val_examples),
        }
        report.append(row)
    return report


def to_markdown(report: list[dict]) -> str:
    def _one_line(s: str, limit: int = 60) -> str:
        return " ".join(s.split())[:limit]

    lines = ["| Family | License | Tokens/word (Pidgin sample) | Baseline probe |", "|---|---|---|---|"]
    for row in report:
        frag = row["tokenizer_fragmentation"]
        frag_str = f"{frag['tokens_per_word']}" if frag.get("status") == "ok" else f"unavailable ({_one_line(frag.get('error', frag.get('status')))})"
        probe = row["baseline_probe"]
        probe_str = "ok" if probe.get("status") == "ok" else f"unavailable ({_one_line(str(probe.get('reason', probe.get('error', probe.get('status')))))})"
        lines.append(f"| {row['family']} | {row['license']} | {frag_str} | {probe_str} |")
    return "\n".join(lines)


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2))
    print()
    print(to_markdown(report))
