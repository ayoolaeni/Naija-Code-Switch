"""
Pluggable inference backend (README §4): the dialogue manager talks to a
`Backend` regardless of whether it's a hosted API, a local model, or (for
zero-config demoability) a deterministic mock.

Backend selection (`select_backend()`):
  - INFERENCE_BACKEND env var, if set, forces "hf" | "local" | "mock".
  - Otherwise: "hf" if HF_TOKEN is set, else "mock".
"""
import os
import random
import re
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, messages: list[dict], **gen_params) -> str:
        """`messages`: [{"role": "system"|"user"|"assistant", "content": str}, ...]"""
        ...


class HFInferenceBackend(Backend):
    """Hugging Face Inference API -- the default backend once HF_TOKEN is set."""

    name = "hf_inference"

    def __init__(self, model_id: str | None = None, token: str | None = None):
        self.model_id = model_id or os.environ.get("INFERENCE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        self.token = token or os.environ["HF_TOKEN"]  # raises clearly if missing
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model=self.model_id, token=self.token)

    def generate(self, messages: list[dict], **gen_params) -> str:
        max_tokens = gen_params.get("max_new_tokens", 256)
        temperature = gen_params.get("temperature", 0.7)
        response = self.client.chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return response.choices[0].message.content


class LocalTransformersBackend(Backend):
    """Local `transformers` + PEFT generation -- for GPU environments."""

    name = "local_transformers"

    def __init__(self, base_model_id: str, lora_adapter_dir: str | None = None):
        from ..modeling.inference import LocalGenerator
        self._generator = LocalGenerator(base_model_id, lora_adapter_dir)

    def generate(self, messages: list[dict], **gen_params) -> str:
        from ..modeling.inference import GenerationParams
        params = GenerationParams(
            max_new_tokens=gen_params.get("max_new_tokens", 256),
            temperature=gen_params.get("temperature", 0.7),
            top_p=gen_params.get("top_p", 0.9),
        )
        return self._generator.generate(messages, params)


class MockBackend(Backend):
    """Deterministic, Pidgin-aware canned responses. No network, no model
    weights -- lets the whole app (UI, dialogue manager, logging) be
    verified as working with zero configuration."""

    name = "mock"

    _PIDGIN_MARKERS = re.compile(
        r"\b(abeg|wetin|dey|dem|una|unu|sef|abi|wey|fit|sabi|don|wahala|"
        r"o|na|kai|gan|jare|oga|shey|ehen|nawa|chai|wan|make|waka|chop|"
        r"comot|vex|sha|pikin)\b",
        re.IGNORECASE,
    )
    _PIDGIN_REPLIES = [
        "I hear you o, {tail} No wahala, I dey here to help.",
        "Ehen, I sabi wetin you talk. {tail} Make we sort am out.",
        "Okay o, I don get am. {tail} Just tell me wetin you need next.",
    ]
    _ENGLISH_REPLIES = [
        "Got it -- {tail} Let me know how I can help further.",
        "Thanks for sharing that. {tail} I'm happy to help with the next step.",
        "Understood. {tail} What would you like to do next?",
    ]

    def generate(self, messages: list[dict], **gen_params) -> str:
        user_turns = [m for m in messages if m["role"] == "user"]
        last_user = user_turns[-1]["content"] if user_turns else ""
        is_pidgin = bool(self._PIDGIN_MARKERS.search(last_user))
        pool = self._PIDGIN_REPLIES if is_pidgin else self._ENGLISH_REPLIES
        rng = random.Random(hash(last_user) & 0xFFFFFFFF)
        template = rng.choice(pool)
        tail = f'you say "{last_user.strip()[:60]}".' if last_user else "I dey listen."
        return template.format(tail=tail)


def select_backend() -> Backend:
    forced = os.environ.get("INFERENCE_BACKEND", "").strip().lower()
    if forced == "hf":
        return HFInferenceBackend()
    if forced == "local":
        base_model_id = os.environ.get("LOCAL_BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        adapter_dir = os.environ.get("LOCAL_LORA_ADAPTER_DIR") or None
        return LocalTransformersBackend(base_model_id, adapter_dir)
    if forced == "mock":
        return MockBackend()

    # auto-select
    if os.environ.get("HF_TOKEN"):
        return HFInferenceBackend()
    return MockBackend()
