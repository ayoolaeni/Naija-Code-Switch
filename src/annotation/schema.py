"""
Per-turn annotation schema (README §6.4): conversational domain, speaker
role, and token-level language tags.
"""
from dataclasses import dataclass, field


@dataclass
class TokenTag:
    text: str
    tag: str  # "english" | "pidgin" | "language_independent"


@dataclass
class TurnAnnotation:
    speaker: str
    text: str
    token_tags: list[TokenTag] = field(default_factory=list)


@dataclass
class DialogueAnnotation:
    dialogue_id: str
    domain: str
    annotator_id: str
    turns: list[TurnAnnotation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dialogue_id": self.dialogue_id,
            "domain": self.domain,
            "annotator_id": self.annotator_id,
            "turns": [
                {
                    "speaker": t.speaker,
                    "text": t.text,
                    "token_tags": [{"text": tt.text, "tag": tt.tag} for tt in t.token_tags],
                }
                for t in self.turns
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DialogueAnnotation":
        return cls(
            dialogue_id=d["dialogue_id"],
            domain=d["domain"],
            annotator_id=d["annotator_id"],
            turns=[
                TurnAnnotation(
                    speaker=t["speaker"],
                    text=t["text"],
                    token_tags=[TokenTag(tt["text"], tt["tag"]) for tt in t["token_tags"]],
                )
                for t in d["turns"]
            ],
        )
