"""
Local generation: base model + optional LoRA adapter via PEFT (README §4
"Inference Engine... Loads the base model + trained LoRA adapters"). Used by
`src.app.inference_engine.LocalTransformersBackend`.
"""
import copy
import os
import threading
from dataclasses import dataclass
from typing import Iterator

import torch
import transformers
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

# float32 weights for a 1.5B model are ~6GB; below this much free memory,
# fall back to bfloat16 (~3GB) so a low-RAM machine gets slower replies
# instead of the container being OOM-killed.
FP32_MIN_AVAILABLE_BYTES = 8 * 1024**3

_DTYPE_KWARG = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"


@dataclass
class GenerationParams:
    # Chat replies don't need to be long, and fewer tokens means
    # proportionally faster generation on CPU. temperature/top_p pulled in
    # from 0.7/0.9: a 1.5B model fine-tuned on a mostly templated dataset
    # drifts off-topic more easily at higher sampling freedom.
    max_new_tokens: int = 120
    temperature: float = 0.5
    top_p: float = 0.85
    do_sample: bool = True


def _pick_cpu_dtype() -> torch.dtype:
    """float32 when there's room for it, else bfloat16.

    Measured on a CPU without native bfloat16 support (common on laptops a
    few years old): float32 generated the same reply ~4x faster. Newer
    transformers versions default to the checkpoint's bfloat16, so this has
    to be chosen explicitly. LOCAL_DTYPE=float32|bfloat16 overrides.
    """
    override = os.environ.get("LOCAL_DTYPE", "").strip().lower()
    if override in ("float32", "fp32"):
        return torch.float32
    if override in ("bfloat16", "bf16"):
        return torch.bfloat16
    try:
        import psutil
        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001
        return torch.bfloat16
    return torch.float32 if available >= FP32_MIN_AVAILABLE_BYTES else torch.bfloat16


class _StopWhenSet(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


class LocalGenerator:
    def __init__(self, base_model_id: str, lora_adapter_dir: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # PyTorch's default thread count (physical cores) is deliberately
        # left alone: forcing it to os.cpu_count() includes hyperthreads,
        # which measured ~20% *slower* on a 2-core/4-thread laptop CPU.
        dtype = _pick_cpu_dtype() if self.device == "cpu" else "auto"
        self.dtype_name = str(dtype).replace("torch.", "")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(base_model_id, **{_DTYPE_KWARG: dtype})
        if lora_adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_adapter_dir)
            # Fold the adapter into the base weights: same outputs, but no
            # extra adapter matmuls on every generation step.
            model = model.merge_and_unload()
        self.model = model.to(self.device)
        self.model.eval()

        self._lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self._prefix = None  # (system_prompt, prefix_input_ids, kv_cache)

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        text = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages) + "\n"
        return text + ("<|assistant|>\n" if add_generation_prompt else "")

    def prime(self, system_prompt: str):
        """Run the fixed system prompt through the model once and keep its
        KV cache. Every request starts with this same prompt, and on a CPU
        re-reading it was most of the wait before the first word appeared;
        reusing the cache means only the new conversation text is processed
        per reply. Doubles as a warm-up, so the first real message doesn't
        also pay the one-off cost of paging the weights in."""
        text = self._render([{"role": "system", "content": system_prompt}], add_generation_prompt=False)
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            cache = self.model(**ids, use_cache=True).past_key_values
        self._prefix = (system_prompt, ids["input_ids"], cache)

    def _prepare(self, messages: list[dict]):
        inputs = self.tokenizer(
            self._render(messages, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        past = None
        if self._prefix is not None and messages and messages[0]["role"] == "system" \
                and messages[0]["content"] == self._prefix[0]:
            prefix_ids = self._prefix[1]
            n = prefix_ids.shape[1]
            full = inputs["input_ids"]
            # Only reuse the cache if the tokens really line up; otherwise
            # silently fall back to processing the whole prompt.
            if full.shape[1] > n and torch.equal(full[:, :n], prefix_ids):
                past = copy.deepcopy(self._prefix[2])
        return inputs, past

    def _generate_into(self, inputs, past, params, streamer, cancel, errors):
        # The lock is held by this worker thread, not by the consuming
        # generator, so an abandoned consumer can never leave it locked.
        with self._lock:
            try:
                if cancel.is_set():
                    return
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        past_key_values=past,
                        max_new_tokens=params.max_new_tokens,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        do_sample=params.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([_StopWhenSet(cancel)]),
                    )
            except Exception as e:  # noqa: BLE001 - re-raised in the consumer
                errors.append(e)
            finally:
                # Without this, an exception inside generate() would leave
                # the consumer blocked on the streamer forever.
                streamer.end()

    def generate_stream(self, messages: list[dict], params: GenerationParams = GenerationParams()) -> Iterator[str]:
        """Yields text chunks as they're generated.

        One generation runs at a time, and starting a new one cancels any
        still in flight: a reply whose page was refreshed or rerun keeps
        running in its background thread otherwise, and on a CPU a few of
        those competing for the same cores meant no reply ever finished.
        """
        if self._active_cancel is not None:
            self._active_cancel.set()
        cancel = threading.Event()
        self._active_cancel = cancel

        inputs, past = self._prepare(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        errors: list[Exception] = []
        threading.Thread(
            target=self._generate_into, args=(inputs, past, params, streamer, cancel, errors), daemon=True
        ).start()
        try:
            for chunk in streamer:
                yield chunk
        finally:
            # Reached on normal completion and also when the consumer goes
            # away mid-stream: stop the background generation either way.
            cancel.set()
        if errors:
            raise errors[0]

    def generate(self, messages: list[dict], params: GenerationParams = GenerationParams()) -> str:
        return "".join(self.generate_stream(messages, params)).strip()
