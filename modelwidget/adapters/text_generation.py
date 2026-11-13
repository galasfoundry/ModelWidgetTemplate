"""Text generation with a transformers causal language model."""

from __future__ import annotations

from pathlib import Path

import torch

from modelwidget.adapters.base import ModelAdapter, register
from modelwidget.adapters.loading import from_pretrained


@register("text-generation", "transformers")
class TransformersTextGenerator(ModelAdapter):
    input_kind = "text"

    def __init__(self, settings, snapshot: Path):
        self.settings = settings
        self.snapshot = Path(snapshot)
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.dtype = torch.float32

    def load(self, *, device: str, dtype: torch.dtype) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = from_pretrained(
            AutoModelForCausalLM,
            self.snapshot,
            dtype=dtype,
            low_cpu_mem_usage=self.settings.low_cpu_mem_usage,
        )
        if self.model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype

    def predict(self, *, image=None, text: str | None = None) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The text generation adapter is not loaded.")
        prompt = (text or "").strip()
        if not prompt:
            raise ValueError("This model expects text.")
        prompt = prompt[: self.settings.max_input_chars]
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        max_new = self.settings.max_new_tokens
        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new,
                min_new_tokens=min(8, max_new),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {
            "results": [{"label": "completion", "text": generated, "score": None, "rank": 1}],
            "text": generated,
        }
