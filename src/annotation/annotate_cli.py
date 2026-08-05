"""
Lightweight annotation tool (README §6.2 step 5, §6.3): reviews/corrects the
`langid.py` auto-tags for each turn in data/processed/dialogues.jsonl and
writes per-annotator annotation files to data/processed/annotated/.

Two modes:
  --auto           non-interactive: accepts the langid.py auto-tags as-is
                    (useful for bootstrapping and for CI/smoke-testing).
  (default)         interactive CLI: shows each turn's auto-tags token by
                    token and lets the annotator accept (Enter) or correct
                    (type e/p/i) each one.

Run: python -m src.annotation.annotate_cli --annotator-id human1 --auto
"""
import argparse
import json
from pathlib import Path

from ..data_pipeline.langid import tag_utterance
from .schema import DialogueAnnotation, TokenTag, TurnAnnotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "dialogues.jsonl"
ANNOTATED_DIR = ROOT / "data" / "processed" / "annotated"

TAG_SHORTHAND = {"e": "english", "p": "pidgin", "i": "language_independent"}


def _load_dialogues(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def auto_annotate(dialogue: dict, annotator_id: str) -> DialogueAnnotation:
    turns = []
    for t in dialogue["turns"]:
        tags = tag_utterance(t["text"])
        turns.append(TurnAnnotation(t["speaker"], t["text"], [TokenTag(x.text, x.tag) for x in tags]))
    return DialogueAnnotation(dialogue["id"], dialogue["domain"], annotator_id, turns)


def interactive_annotate(dialogue: dict, annotator_id: str) -> DialogueAnnotation:
    print(f"\n=== Dialogue {dialogue['id']} ({dialogue['domain']}) ===")
    turns = []
    for t in dialogue["turns"]:
        print(f"\n[{t['speaker']}] {t['text']}")
        auto_tags = tag_utterance(t["text"])
        token_tags = []
        for tok in auto_tags:
            resp = input(f"  '{tok.text}' auto={tok.tag}  [Enter=accept, e/p/i=correct]: ").strip().lower()
            tag = TAG_SHORTHAND.get(resp, tok.tag)
            token_tags.append(TokenTag(tok.text, tag))
        turns.append(TurnAnnotation(t["speaker"], t["text"], token_tags))
    return DialogueAnnotation(dialogue["id"], dialogue["domain"], annotator_id, turns)


def run(annotator_id: str, input_path: Path = DEFAULT_INPUT, auto: bool = False,
        sample_fraction: float = 1.0) -> Path:
    dialogues = _load_dialogues(input_path)
    n_sample = max(1, round(len(dialogues) * sample_fraction))
    dialogues = dialogues[:n_sample]

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANNOTATED_DIR / f"{annotator_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in dialogues:
            annotation = auto_annotate(d, annotator_id) if auto else interactive_annotate(d, annotator_id)
            f.write(json.dumps(annotation.to_dict(), ensure_ascii=False) + "\n")
    print(f"Wrote {len(dialogues)} annotated dialogues to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--auto", action="store_true", help="accept langid.py tags without prompting")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    args = parser.parse_args()
    run(args.annotator_id, args.input, args.auto, args.sample_fraction)


if __name__ == "__main__":
    main()
