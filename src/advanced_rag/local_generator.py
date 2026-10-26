"""Optional local-only answer generation; loaded only when explicitly requested."""
from __future__ import annotations


class LocalAnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "auto",
                 max_new_tokens: int = 256) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for local generation") from exc
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"low_cpu_mem_usage": True}
        if device == "auto":
            kwargs["device_map"] = "auto"
        elif device in {"cuda", "mps"}:
            kwargs["device_map"] = device
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if device not in {"auto", "cuda", "mps"}:
            self.model.to(device)

    def generate(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": (
                "Answer using only the retrieved context. If the context does not support an answer, "
                "say you do not have enough information. Cite source names when available."
            )},
            {"role": "user", "content": f"Retrieved context:\n{context}\n\nQuestion: {query}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                     do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
