"""
Loader for NaijaSenti (Muhammad et al., 2022) via the Hugging Face Hub
(README §6.1: "authentic code-mixed tokens/orthography... isolated tweets,
not dialogue -- needs reconstruction into exchanges").

Isolated tweets are turned into synthetic 2-turn "exchanges" by pairing
consecutive genuinely code-mixed tweets (per `langid.is_code_switched`) --
this is a crude reconstruction, not real conversational data, and is clearly
tagged as such in `metadata["synthetic_pairing"]`.

Requires network access to the Hugging Face Hub. If unavailable (offline
sandbox, dataset renamed/gated, etc.), this loader logs a warning and
returns an empty list rather than failing the whole pipeline.
"""
from .base import RawDialogue, RawTurn, SourceLoader
from ..langid import is_code_switched
from ..privacy import scrub

# Candidate Hub ids -- NaijaSenti has been re-hosted under a couple of
# organizations over time; we try each and use the first that loads.
CANDIDATE_DATASET_IDS = [
    ("HausaNLP/NaijaSenti-Twitter", "pcm"),
    ("HausaNLP/naijasenti", "pcm"),
]

MAX_PAIRS = 150


class NaijaSentiSource(SourceLoader):
    name = "naijasenti"
    license = "CC-BY-4.0 (research use; verify current terms before any redistribution)"
    trainable = True

    def load(self) -> list[RawDialogue]:
        try:
            from datasets import load_dataset
        except ImportError:
            print("[naijasenti] `datasets` not installed, skipping this source")
            return []

        ds = None
        used_id = None
        for hub_id, config in CANDIDATE_DATASET_IDS:
            try:
                ds = load_dataset(hub_id, config, split="train")
                used_id = hub_id
                break
            except Exception as e:  # noqa: BLE001 - best-effort external fetch
                print(f"[naijasenti] could not load {hub_id}/{config}: {e}")
        if ds is None:
            print("[naijasenti] no candidate dataset id worked, skipping source")
            return []

        text_col = "tweet" if "tweet" in ds.column_names else ds.column_names[0]
        mixed_texts = [scrub(row[text_col]) for row in ds if is_code_switched(row[text_col])]

        dialogues = []
        for i in range(0, min(len(mixed_texts) - 1, MAX_PAIRS * 2), 2):
            dialogues.append(
                RawDialogue(
                    id=f"naijasenti_{i // 2:04d}",
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
        print(f"[naijasenti] built {len(dialogues)} synthetic 2-turn exchanges")
        return dialogues
