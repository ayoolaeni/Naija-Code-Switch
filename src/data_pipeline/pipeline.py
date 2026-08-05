"""
Orchestrates README §6.2: load -> filter (genuine mixing) -> privacy-strip
-> CMI-score -> write to data/processed/.

Run: python -m src.data_pipeline.pipeline
"""
import json
from pathlib import Path

from .cmi import cmi_utterance
from .langid import is_code_switched
from .privacy import scrub
from .sources.afrisenti import AfriSentiSource
from .sources.authored import AuthoredSource
from .sources.base import RawDialogue
from .sources.flores200 import Flores200Source
from .sources.naijasenti import NaijaSentiSource

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REFERENCE_DIR = ROOT / "data" / "processed" / "reference"

ALL_SOURCES = [AuthoredSource(), NaijaSentiSource(), AfriSentiSource(), Flores200Source()]


def process_dialogue(d: RawDialogue) -> dict:
    for t in d.turns:
        t.text = scrub(t.text)
    turn_cmis = [cmi_utterance(t.text) for t in d.turns]
    return {
        "id": d.id,
        "domain": d.domain,
        "source": d.source,
        "license": d.license,
        "metadata": d.metadata,
        "turns": [{"speaker": t.speaker, "text": t.text} for t in d.turns],
        "turn_cmi": turn_cmis,
        "mean_cmi": sum(turn_cmis) / len(turn_cmis) if turn_cmis else 0.0,
    }


def run(sources=None, min_turns: int = 2) -> dict:
    sources = sources if sources is not None else ALL_SOURCES
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    trainable_out = PROCESSED_DIR / "dialogues.jsonl"
    reference_out = REFERENCE_DIR / "flores200_reference.jsonl"

    n_trainable = 0
    n_reference = 0
    n_filtered_out = 0

    with trainable_out.open("w", encoding="utf-8") as train_f, \
            reference_out.open("w", encoding="utf-8") as ref_f:
        for source in sources:
            dialogues = source.load()
            for d in dialogues:
                if len(d.turns) < min_turns:
                    n_filtered_out += 1
                    continue
                # genuine-mixing filter (README §6.2 step 2): at least one
                # turn must show real English/Pidgin mixing.
                if source.trainable and not any(is_code_switched(t.text) for t in d.turns):
                    n_filtered_out += 1
                    continue
                record = process_dialogue(d)
                if source.trainable:
                    train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_trainable += 1
                else:
                    ref_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_reference += 1

    summary = {
        "trainable_dialogues": n_trainable,
        "reference_dialogues": n_reference,
        "filtered_out": n_filtered_out,
        "trainable_path": str(trainable_out),
        "reference_path": str(reference_out),
    }
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
