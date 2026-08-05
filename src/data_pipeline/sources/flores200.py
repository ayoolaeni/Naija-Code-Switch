"""
Loader for FLORES-200 (NLLB Team, 2022) -- README §6.1: "reference parallel
English-Pidgin sentences... translated, not naturally occurring dialogue --
reference only." `trainable = False`: `pipeline.py` and `split.py` must
exclude anything loaded here from train/val/test, per README's explicit
"Explicitly excluded" list (§6.1) treating it as reference-only.

Not every FLORES-200 mirror includes a Nigerian Pidgin (`pcm_Latn`) config;
if it's unavailable this loader logs a warning and returns an empty list.
"""
from .base import RawDialogue, RawTurn, SourceLoader
from ..privacy import scrub

CANDIDATE_DATASET_IDS = [
    ("facebook/flores", "eng_Latn-pcm_Latn"),
    ("openlanguagedata/flores_plus", "pcm_Latn"),
]

MAX_REFERENCES = 50


class Flores200Source(SourceLoader):
    name = "flores200"
    license = "CC-BY-SA-4.0 (reference only, not for training)"
    trainable = False  # reference set only -- see docstring

    def load(self) -> list[RawDialogue]:
        try:
            from datasets import load_dataset
        except ImportError:
            print("[flores200] `datasets` not installed, skipping this source")
            return []

        ds = None
        used_id = None
        for hub_id, config in CANDIDATE_DATASET_IDS:
            try:
                ds = load_dataset(hub_id, config, split="dev")
                used_id = hub_id
                break
            except Exception as e:  # noqa: BLE001 - best-effort external fetch
                print(f"[flores200] could not load {hub_id}/{config}: {e}")
        if ds is None:
            print("[flores200] no candidate dataset/config worked (pcm_Latn may not be mirrored), skipping")
            return []

        dialogues = []
        for i, row in enumerate(ds):
            if i >= MAX_REFERENCES:
                break
            eng = scrub(row.get("sentence_eng_Latn", ""))
            pcm = scrub(row.get("sentence_pcm_Latn", ""))
            if not eng or not pcm:
                continue
            dialogues.append(
                RawDialogue(
                    id=f"flores_{i:04d}",
                    domain="reference",
                    turns=[RawTurn("english_reference", eng), RawTurn("pidgin_reference", pcm)],
                    source=self.name,
                    license=self.license,
                    metadata={"hub_id": used_id, "reference_only": True},
                )
            )
        print(f"[flores200] loaded {len(dialogues)} reference pairs (excluded from train/val/test)")
        return dialogues
