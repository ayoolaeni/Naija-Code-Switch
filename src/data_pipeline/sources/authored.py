"""
Loader for the hand-authored + script-generated authored dialogues (README
§6.1 "Manually authored dialogues" -- the core training data). Loads every
*.jsonl file under data/authored/ (not a single hardcoded filename), so
additional batches can be dropped in as their own file without needing to be
merged into any one file by hand. See scripts/author_seed_dialogues.py (the
original 29 hand-authored dialogues, contributor ids c1-c4) and
scripts/generate_synthetic_dialogues.py (the ~770-dialogue script-generated
batch, contributor ids synth_c1-synth_c4 -- kept in a separate file so the
hand-authored file's provenance claim stays accurate).
"""
import json
from pathlib import Path

from .base import RawDialogue, RawTurn, SourceLoader

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "authored"


class AuthoredSource(SourceLoader):
    name = "authored"
    license = "project-owned (hand-authored + script-generated synthetic dialogues for this project)"
    trainable = True

    def __init__(self, directory: Path = DEFAULT_DIR):
        self.directory = directory

    def load(self) -> list[RawDialogue]:
        if not self.directory.exists():
            print(f"[authored] no directory at {self.directory}, run scripts/author_seed_dialogues.py first")
            return []
        paths = sorted(self.directory.glob("*.jsonl"))
        if not paths:
            print(f"[authored] no *.jsonl files under {self.directory}, run scripts/author_seed_dialogues.py first")
            return []
        dialogues = []
        for path in paths:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    dialogues.append(
                        RawDialogue(
                            id=d["id"],
                            domain=d["domain"],
                            turns=[RawTurn(t["speaker"], t["text"]) for t in d["turns"]],
                            source=self.name,
                            license=self.license,
                            metadata={"contributor": d.get("contributor", "unknown")},
                        )
                    )
        return dialogues
