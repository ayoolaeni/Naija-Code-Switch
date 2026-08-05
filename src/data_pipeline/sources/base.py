"""
Common interface for dataset source loaders (README §6.1/§6.2 step 1: every
usable source gets a loader/adapter and a license tag).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RawTurn:
    speaker: str
    text: str


@dataclass
class RawDialogue:
    id: str
    domain: str
    turns: list[RawTurn]
    source: str = "unknown"
    license: str = "unknown"
    metadata: dict = field(default_factory=dict)


class SourceLoader(ABC):
    """Every source adapter implements `load()` and declares its license."""

    name: str = "base"
    license: str = "unknown"
    trainable: bool = True  # False for reference-only sources (e.g. FLORES-200)

    @abstractmethod
    def load(self) -> list[RawDialogue]:
        ...
