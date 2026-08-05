"""
Loader for AfriSenti (Muhammad et al., 2023) via the Hugging Face Hub
(README §6.1: "additional Pidgin lexical coverage... single utterances, no
conversational context"). Same synthetic-pairing caveat as
`naijasenti.py` -- this reconstructs 2-turn exchanges from isolated tweets,
it is not naturally occurring dialogue.
"""
from .base import RawDialogue, RawTurn, SourceLoader
from ..langid import is_code_switched
from ..privacy import scrub

CANDIDATE_DATASET_IDS = [
    ("shmuhammad/AfriSenti-twitter-sentiment", "pcm"),
    ("HausaNLP/afrisenti", "pcm"),
]

MAX_PAIRS = 100


class AfriSentiSource(SourceLoader):
    name = "afrisenti"
    license = "CC-BY-4.0 (research use; verify current terms before any redistribution)"
    trainable = True

    def load(self) -> list[RawDialogue]:
        try:
            from datasets import load_dataset
        except ImportError:
            print("[afrisenti] `datasets` not installed, skipping this source")
            return []

        ds = None
        used_id = None
        for hub_id, config in CANDIDATE_DATASET_IDS:
            try:
                ds = load_dataset(hub_id, config, split="train")
                used_id = hub_id
                break
            except Exception as e:  # noqa: BLE001 - best-effort external fetch
                print(f"[afrisenti] could not load {hub_id}/{config}: {e}")
        if ds is None:
            print("[afrisenti] no candidate dataset id worked, skipping source")
            return []

        text_col = "tweet" if "tweet" in ds.column_names else ds.column_names[0]
        mixed_texts = [scrub(row[text_col]) for row in ds if is_code_switched(row[text_col])]

        dialogues = []
        for i in range(0, min(len(mixed_texts) - 1, MAX_PAIRS * 2), 2):
            dialogues.append(
                RawDialogue(
                    id=f"afrisenti_{i // 2:04d}",
                    domain="everyday_chat",
                    turns=[
                        RawTurn("user", mixed_texts[i]),
                        RawTurn("assistant", mixed_texts[i + 1]),
                    ],
                    source=self.name,
                    license=self.license,
                    metadata={"hub_id": used_id, "synthetic_pairing": True},
                )
            )
        print(f"[afrisenti] built {len(dialogues)} synthetic 2-turn exchanges")
        return dialogues
