"""
Comparative evaluation harness (README §7.4): runs the same evaluation
identically across configured systems on data/splits/test.jsonl.

`Backend`-compatible systems (anything with `.generate(messages) -> str`,
including everything in src/app/inference_engine.py) can be registered by
name. General-purpose commercial chatbot baselines need to be added by the
user as their own Backend-shaped wrapper (no such API keys are configured in
this build) -- see the README section "Scope of this build".

Run: python -m src.evaluation.compare --systems mock
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..modeling.data_utils import build_examples
from ..modeling.prompt_design import get_frozen_system_prompt
from .automatic_metrics import compute_all
from .cs_metrics import switching_density_comparison

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data" / "splits" / "test.jsonl"
COMPARISON_DIR = ROOT / "logs" / "comparisons"


def _resolve_backend(name: str):
    from ..app.inference_engine import HFInferenceBackend, LocalTransformersBackend, MockBackend
    if name == "mock":
        return MockBackend()
    if name == "hf":
        return HFInferenceBackend()
    if name == "local":
        import os
        base_model_id = os.environ.get("LOCAL_BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        adapter_dir = os.environ.get("LOCAL_LORA_ADAPTER_DIR") or None
        return LocalTransformersBackend(base_model_id, adapter_dir)
    raise ValueError(f"unknown system name: {name!r} (use mock | hf | local, or register your own)")


def run(system_names: list[str], test_path: Path = TEST_PATH, max_examples: int | None = None) -> dict:
    system_prompt = get_frozen_system_prompt()
    examples = build_examples(test_path, system_prompt)
    if max_examples:
        examples = examples[:max_examples]
    references = [ex.target for ex in examples]

    report = {"n_examples": len(examples), "systems": {}}
    generations_by_system = {}

    for name in system_names:
        backend = _resolve_backend(name)
        hypotheses = [backend.generate(ex.messages) for ex in examples]
        generations_by_system[name] = hypotheses

        metrics = compute_all(hypotheses, references) if hypotheses else {}
        cs = switching_density_comparison(hypotheses, references) if hypotheses else {}
        report["systems"][name] = {"automatic_metrics": metrics, "cs_metrics": cs}

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = COMPARISON_DIR / f"{stamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "report": report,
                "references": references,
                "generations_by_system": generations_by_system,
                "prompts": [ex.messages[-1]["content"] if len(ex.messages) > 1 else "" for ex in examples],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    report["output_path"] = str(out_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=["mock"])
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    report = run(args.systems, max_examples=args.max_examples)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
