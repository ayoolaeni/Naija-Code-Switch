"""
Dialogue-level train/val/test partitioning (README §6.5): 80/10/10, split at
the whole-dialogue level (never individual turns, to prevent leakage),
stratified by source so authored vs. retrieved material is distributed
proportionally across all three partitions.

Run: python -m src.data_pipeline.split
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = ROOT / "data" / "processed" / "dialogues.jsonl"
SPLITS_DIR = ROOT / "data" / "splits"

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_split(dialogues: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for d in dialogues:
        by_source[d["source"]].append(d)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for source, group in by_source.items():
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * RATIOS["train"])
        n_val = round(n * RATIOS["val"])
        # remainder to test, guarantees all dialogues are placed exactly once
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])

    for name in splits:
        rng.shuffle(splits[name])
    return splits


def run(seed: int = 42) -> dict:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found -- run `python -m src.data_pipeline.pipeline` first"
        )
    dialogues = _load(PROCESSED_PATH)
    splits = stratified_split(dialogues, seed=seed)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, items in splits.items():
        out_path = SPLITS_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for d in items:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        summary[name] = len(items)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
