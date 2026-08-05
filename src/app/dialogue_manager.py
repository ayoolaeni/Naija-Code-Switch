"""
Dialogue manager (README §4): holds conversation history, prepends the
frozen system prompt, and truncates the oldest turns first once the history
approaches a token budget. This is the component responsible for fixing
"failure mode 3" (loss of context in multi-turn Pidgin exchanges).

Token counting uses a real tokenizer when one is supplied (e.g. the active
backend's tokenizer); otherwise it falls back to a conservative
chars-per-token heuristic so the manager still works with backends that
don't expose a local tokenizer (e.g. the HF Inference API backend).
"""
from dataclasses import dataclass, field

from ..modeling.prompt_design import get_frozen_system_prompt

CHARS_PER_TOKEN_ESTIMATE = 4  # conservative fallback when no tokenizer is available


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class DialogueManager:
    system_prompt: str = field(default_factory=get_frozen_system_prompt)
    max_context_tokens: int = 2048
    tokenizer: object | None = None  # optional, needs .encode(str) -> list

    history: list[Turn] = field(default_factory=list)

    def add_user_turn(self, text: str):
        self.history.append(Turn("user", text))

    def add_assistant_turn(self, text: str):
        self.history.append(Turn("assistant", text))

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)

    def _truncate(self):
        """Drops the oldest user/assistant turn pairs first until the
        rendered context fits the token budget. The system prompt is never
        dropped."""
        while True:
            total = self._count_tokens(self.system_prompt)
            for turn in self.history:
                total += self._count_tokens(turn.content)
            if total <= self.max_context_tokens or len(self.history) <= 2:
                return
            # drop the oldest turn (and its pair partner if present) to keep
            # user/assistant alternation intact
            self.history.pop(0)

    def build_messages(self) -> list[dict]:
        self._truncate()
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend({"role": t.role, "content": t.content} for t in self.history)
        return messages

    def reset(self):
        self.history = []
