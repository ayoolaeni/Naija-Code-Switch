"""
Local generation: base model + optional LoRA adapter via PEFT (README §4
"Inference Engine... Loads the base model + trained LoRA adapters"). Used by
`src.app.inference_engine.LocalTransformersBackend`.
"""
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationParams:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class LocalGenerator:
    def __init__(self, base_model_id: str, lora_adapter_dir: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(base_model_id)
        if lora_adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_adapter_dir)
        self.model = model.to(self.device)
        self.model.eval()

    def generate(self, messages: list[dict], params: GenerationParams = GenerationParams()) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages) + "\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
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
