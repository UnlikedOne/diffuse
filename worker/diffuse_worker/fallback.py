import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalFallback:
    def __init__(self, model_id: str, device: str = "cpu", cache_dir: str | None = None):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch.float32
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)