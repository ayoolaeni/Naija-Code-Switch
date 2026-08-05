"""
System prompt design (README §5.3): role/tone, instruction to mirror the
user's English/Pidgin mixture, and 2-4 few-shot code-switched exchanges.

Iterate `PROMPT_VARIANTS` against the validation split only; `FROZEN_VARIANT`
records which one won and is used everywhere else (train_lora.py,
baseline_probe.py, the chat app) so the test set is never touched during
prompt iteration.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTHORED_PATH = ROOT / "data" / "authored" / "dialogues.jsonl"

BASE_ROLE = (
    "You are Naija-Switch, a friendly Nigerian conversational assistant. "
    "You are equally comfortable in English and Nigerian Pidgin (Naija), "
    "and you understand code-switched input where a person mixes both "
    "languages within the same message or the same sentence."
)

MIRRORING_INSTRUCTION = (
    "Always respond in the same mixture of English and Pidgin that the "
    "user used. If the user writes mostly Pidgin, reply mostly in Pidgin. "
    "If they mix English and Pidgin mid-sentence, feel free to do the same "
    "in your reply -- do not default to pure English just because it feels "
    "safer. Keep responses natural, warm, and concise."
)

SCOPE_DISCLAIMER = (
    "You are a general-conversation assistant only. You do not give medical, "
    "legal, or financial advice; if asked, say so plainly and suggest the "
    "person consult a qualified professional."
)


def _load_few_shot_examples(n: int = 3) -> list[dict]:
    """Pulls short, clearly code-switched exchanges from the seed dataset to
    use as worked few-shot examples."""
    if not AUTHORED_PATH.exists():
        return []
    examples = []
    with AUTHORED_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            turns = d["turns"]
            for i in range(len(turns) - 1):
                if turns[i]["speaker"] == "user" and turns[i + 1]["speaker"] == "assistant":
                    examples.append({"user": turns[i]["text"], "assistant": turns[i + 1]["text"]})
                    break
            if len(examples) >= n:
                break
    return examples


def build_system_prompt(include_few_shot: bool = True, n_few_shot: int = 3) -> str:
    parts = [BASE_ROLE, MIRRORING_INSTRUCTION, SCOPE_DISCLAIMER]
    if include_few_shot:
        examples = _load_few_shot_examples(n_few_shot)
        if examples:
            parts.append("Here are some example exchanges:")
            for ex in examples:
                parts.append(f'User: "{ex["user"]}"\nAssistant: "{ex["assistant"]}"')
    return "\n\n".join(parts)


# Two candidate variants to iterate on the validation split (README §5.3).
PROMPT_VARIANTS = {
    "with_few_shot": lambda: build_system_prompt(include_few_shot=True, n_few_shot=3),
    "instruction_only": lambda: build_system_prompt(include_few_shot=False),
}

# Frozen after validation-split iteration -- used by train_lora.py,
# baseline_probe.py, and the chat app's dialogue_manager.
FROZEN_VARIANT = "with_few_shot"


def get_frozen_system_prompt() -> str:
    return PROMPT_VARIANTS[FROZEN_VARIANT]()


if __name__ == "__main__":
    print(get_frozen_system_prompt())
