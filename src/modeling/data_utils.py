"""
Converts data/splits/*.jsonl dialogues into instruction-response training
examples that preserve preceding conversational turns as context (README
§5.4: "not single-turn pairs -- this is essential to teaching context
retention").

For a dialogue with turns [u1, a1, u2, a2, ...], one training example is
built per assistant turn a_i, with all turns before it (including the
system prompt) as context and a_i as the generation target.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from .prompt_design import get_frozen_system_prompt

ROLE_MAP = {"user": "user", "assistant": "assistant"}


@dataclass
class TrainingExample:
    messages: list[dict]  # context, roles "system"/"user"/"assistant"
    target: str  # the assistant response to generate


def load_dialogues(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dialogue_to_examples(dialogue: dict, system_prompt: str) -> list[TrainingExample]:
    examples = []
    history: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in dialogue["turns"]:
        role = ROLE_MAP.get(turn["speaker"])
        if role is None:
            # skip non user/assistant speaker turns (e.g. flores reference rows)
            continue
        if role == "assistant" and history[-1]["role"] == "user":
            examples.append(TrainingExample(messages=list(history), target=turn["text"]))
        history.append({"role": role, "content": turn["text"]})
    return examples


def build_examples(path: Path, system_prompt: str | None = None) -> list[TrainingExample]:
    system_prompt = system_prompt or get_frozen_system_prompt()
    examples = []
    for dialogue in load_dialogues(path):
        examples.extend(dialogue_to_examples(dialogue, system_prompt))
    return examples


def format_prompt_and_target(tokenizer, example: TrainingExample) -> tuple[str, str]:
    """Renders (prompt_text, full_text) using the tokenizer's chat template
    when available, falling back to a plain role-tagged template for
    tokenizers without one (e.g. tiny smoke-test models)."""
    full_messages = example.messages + [{"role": "assistant", "content": example.target}]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            example.messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
        return prompt_text, full_text

    def render(msgs):
        parts = []
        for m in msgs:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        return "\n".join(parts) + "\n"

    prompt_text = render(example.messages) + "<|assistant|>\n"
    full_text = render(full_messages)
    return prompt_text, full_text
