"""Text classification with a transformers sequence-classification model."""

from __future__ import annotations

from pathlib import Path

import torch

from modelwidget.adapters.base import ModelAdapter, register, topk_rows
from modelwidget.adapters.loading import from_pretrained
from modelwidget.labels import labels_from_id2label


@register("text-classification", "transformers")
class TransformersTextClassifier(ModelAdapter):
    input_kind = "text"

    def __init__(self, settings, snapshot: Path):
        self.settings = settings
        self.snapshot = Path(snapshot)
        self.model = None
        self.tokenizer = None
        self.labels = []
        self.device = "cpu"
        self.dtype = torch.float32

    def load(self, *, device: str, dtype: torch.dtype) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
        self.model = from_pretrained(
            AutoModelForSequenceClassification,
            self.snapshot,
            dtype=dtype,
            low_cpu_mem_usage=self.settings.low_cpu_mem_usage,
        )
        self.model.to(device)
        self.model.eval()
        self.labels = labels_from_id2label(
            dict(self.model.config.id2label or {}),
            getattr(self.model.config, "num_labels", None),
        )
        self.device = device
        self.dtype = dtype

    def predict(self, *, image=None, text: str | None = None) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The text classification adapter is not loaded.")
        prompt = (text or "").strip()
        if not prompt:
            raise ValueError("This model expects text.")
        prompt = prompt[: self.settings.max_input_chars]
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        probs = torch.softmax(logits[0].float(), dim=-1).cpu()
        return {"results": topk_rows(probs, self.labels, self.settings.top_k), "text": None}
