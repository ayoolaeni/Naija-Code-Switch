"""
Local generation: base model + optional LoRA adapter via PEFT (README §4
"Inference Engine... Loads the base model + trained LoRA adapters"). Used by
`src.app.inference_engine.LocalTransformersBackend`.
"""
import os
import threading
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


@dataclass
class GenerationParams:
    # Chat replies don't need to be long, and fewer tokens means
    # proportionally faster generation on CPU -- 120 is enough for a
    # normal reply without the multi-minute waits a 256-token cap can
    # produce. temperature/top_p pulled in slightly from the earlier
    # 0.7/0.9: a 1.5B model fine-tuned on a mostly-small/templated dataset
    # drifts off-topic more easily at higher sampling freedom, and tighter
    # sampling measurably reduces that without needing retraining.
    max_new_tokens: int = 120
    temperature: float = 0.5
    top_p: float = 0.85
    do_sample: bool = True


class LocalGenerator:
    def __init__(self, base_model_id: str, lora_adapter_dir: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            # PyTorch doesn't always default to using every available core;
            # explicit is safer than hoping the environment set this up,
            # and generation speed on CPU is directly threadcount-bound.
            torch.set_num_threads(os.cpu_count() or 1)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(base_model_id)
        if lora_adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_adapter_dir)
        self.model = model.to(self.device)
        self.model.eval()

    def _build_prompt_inputs(self, messages: list[dict]):
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages) + "\n<|assistant|>\n"
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

    def generate(self, messages: list[dict], params: GenerationParams = GenerationParams()) -> str:
        inputs = self._build_prompt_inputs(messages)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                do_sample=params.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_stream(self, messages: list[dict], params: GenerationParams = GenerationParams()) -> Iterator[str]:
        """Yields text chunks as they're generated, instead of blocking
        until the full reply is done -- the single biggest perceived-speed
        win on slow (CPU) hardware, since the user sees words appear
        immediately rather than staring at a spinner for the whole reply.
        Total generation time is unchanged; only when the user sees output
        changes."""
        inputs = self._build_prompt_inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            do_sample=params.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join()
