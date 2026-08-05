"""
Loader for the hand-authored seed dialogues (README §6.1 "Manually authored
dialogues" -- the core training data). See scripts/author_seed_dialogues.py.
"""
import json
from pathlib import Path

from .base import RawDialogue, RawTurn, SourceLoader

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "authored" / "dialogues.jsonl"


class AuthoredSource(SourceLoader):
    name = "authored"
    license = "project-owned (hand-authored for this project)"
    trainable = True

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path

    def load(self) -> list[RawDialogue]:
        if not self.path.exists():
            print(f"[authored] no file at {self.path}, run scripts/author_seed_dialogues.py first")
            return []
        dialogues = []
        with self.path.open(encoding="utf-8") as f:
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
