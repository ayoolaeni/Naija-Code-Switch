"""
Inter-annotator agreement (README §6.4):
  - Fleiss' kappa for token-level language tag agreement (>=2 annotators).
  - Krippendorff's alpha for ordinal human-eval quality judgments.

Both are reported honestly even when low, per the README's explicit
instruction that a low score is itself a finding.

Run: python -m src.annotation.agreement --demo
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from .schema import DialogueAnnotation

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED_DIR = ROOT / "data" / "processed" / "annotated"

CATEGORIES = ["english", "pidgin", "language_independent"]


def _load_annotator_file(path: Path) -> dict[str, DialogueAnnotation]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = DialogueAnnotation.from_dict(json.loads(line))
            out[d.dialogue_id] = d
    return out


def fleiss_kappa_token_tags(annotator_files: list[Path]) -> dict:
    """Computes Fleiss' kappa over turns/tokens annotated by ALL given
    annotators (the intersection of dialogue ids across files), i.e. the
    double-annotated overlap sample (README: >=10% of the dataset)."""
    from statsmodels.stats.inter_rater import fleiss_kappa

    annotators = {p.stem: _load_annotator_file(p) for p in annotator_files}
    if len(annotators) < 2:
        raise ValueError("need at least 2 annotator files to compute agreement")

    shared_ids = set.intersection(*(set(a.keys()) for a in annotators.values()))
    if not shared_ids:
        raise ValueError("no overlapping double-annotated dialogues found across annotator files")

    rows = []  # each row: counts per category across annotators, for one token
    for dialogue_id in sorted(shared_ids):
        per_annotator_turns = [annotators[name][dialogue_id].turns for name in annotators]
        n_turns = min(len(t) for t in per_annotator_turns)
        for turn_idx in range(n_turns):
            per_annotator_tags = [turns[turn_idx].token_tags for turns in per_annotator_turns]
            n_tokens = min(len(t) for t in per_annotator_tags)
            for tok_idx in range(n_tokens):
                counts = [0] * len(CATEGORIES)
                for tags in per_annotator_tags:
                    cat_idx = CATEGORIES.index(tags[tok_idx].tag)
                    counts[cat_idx] += 1
                rows.append(counts)

    table = np.array(rows)
    kappa = fleiss_kappa(table, method="fleiss")
    return {
        "n_annotators": len(annotators),
        "n_double_annotated_dialogues": len(shared_ids),
        "n_tokens_compared": len(rows),
        "fleiss_kappa": float(kappa),
    }


def krippendorff_alpha_ordinal(ratings: list[list[float]]) -> dict:
    """`ratings`: one row per rater, one column per rated item; use NaN for
    items a given rater did not score."""
    import krippendorff

    alpha = krippendorff.alpha(reliability_data=ratings, level_of_measurement="ordinal")
    return {"n_raters": len(ratings), "n_items": len(ratings[0]) if ratings else 0, "krippendorff_alpha": float(alpha)}


def _demo():
    """Builds a synthetic double-annotation (auto langid tags vs. a randomly
    perturbed copy) purely to exercise the agreement math end-to-end without
    requiring real human annotators. Clearly not a real agreement study."""
    from ..data_pipeline.pipeline import PROCESSED_DIR
    from .annotate_cli import run as annotate_run

    dialogues_path = PROCESSED_DIR / "dialogues.jsonl"
    if not dialogues_path.exists():
        print("run `python -m src.data_pipeline.pipeline` first")
        return

    # Annotator A: plain auto-tags.
    annotate_run("demo_auto_a", dialogues_path, auto=True, sample_fraction=1.0)

    # Annotator B: auto-tags with ~15% of tokens randomly perturbed, to
    # simulate a second (imperfect) annotator and produce a non-trivial
    # kappa instead of a vacuous 1.0.
    rng = random.Random(7)
    path_a = ANNOTATED_DIR / "demo_auto_a.jsonl"
    path_b = ANNOTATED_DIR / "demo_auto_b.jsonl"
    with path_a.open(encoding="utf-8") as fin, path_b.open("w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            d["annotator_id"] = "demo_auto_b"
            for turn in d["turns"]:
                for tok in turn["token_tags"]:
                    if rng.random() < 0.15:
                        tok["tag"] = rng.choice(CATEGORIES)
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    kappa_result = fleiss_kappa_token_tags([path_a, path_b])
    print("Fleiss' kappa (token-level language tags, synthetic double-annotation demo):")
    print(json.dumps(kappa_result, indent=2))

    # Synthetic ordinal human-eval ratings demo for Krippendorff's alpha.
    demo_ratings = [
        [4, 5, 3, 4, 2, 5, 4, 3, 4, 5],
        [4, 4, 3, 5, 2, 4, 4, 3, 3, 5],
    ]
    alpha_result = krippendorff_alpha_ordinal(demo_ratings)
    print("\nKrippendorff's alpha (synthetic ordinal human-eval ratings demo):")
    print(json.dumps(alpha_result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run the synthetic end-to-end demo")
    parser.add_argument("--annotator-files", nargs="*", type=Path, default=None)
    args = parser.parse_args()

    if args.demo or not args.annotator_files:
        _demo()
    else:
        result = fleiss_kappa_token_tags(args.annotator_files)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
