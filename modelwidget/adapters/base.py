"""Adapter plug-in point.

A task adapter loads one local snapshot and turns an image or a string into
JSON-ready rows. Register it with :func:`register` and import the module from
``modelwidget/adapters/__init__.py``.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from modelwidget.labels import ClassLabel

REGISTRY: dict[tuple[str, str], type[ModelAdapter]] = {}


class ModelAdapter:
    """Base class for one model task and one library backend."""

    task: ClassVar[str]
    backend: ClassVar[str]
    input_kind: ClassVar[str]  # "image" or "text"

    def load(self, *, device: str, dtype: torch.dtype) -> None:
        raise NotImplementedError

    def predict(self, *, image=None, text: str | None = None) -> dict:
        raise NotImplementedError


def register(task: str, backend: str):
    """Decorator that records an adapter class under ``(task, backend)``."""

    def decorator(cls: type[ModelAdapter]) -> type[ModelAdapter]:
        cls.task = task
        cls.backend = backend
        REGISTRY[(task, backend)] = cls
        return cls

    return decorator


def get_adapter_class(task: str, backend: str) -> type[ModelAdapter]:
    key = (task, backend)
    cls = REGISTRY.get(key)
    if cls is None:
        known = ", ".join(f"{name} via {lib}" for name, lib in sorted(REGISTRY)) or "none"
        raise ValueError(
            f"No adapter for task {task!r} with backend {backend!r}. Registered adapters: {known}."
        )
    return cls


def create_adapter(settings, snapshot) -> ModelAdapter:
    cls = get_adapter_class(settings.model_task, settings.model_backend)
    return cls(settings, snapshot)


def topk_rows(probs: torch.Tensor, labels: list[ClassLabel], top_k: int) -> list[dict]:
    """Highest-scoring classes as result rows."""
    if probs.ndim != 1:
        raise ValueError("topk_rows expects a 1-D probability vector.")
    count = int(probs.shape[0])
    if len(labels) < count:
        labels = list(labels) + [ClassLabel(f"class_{index}") for index in range(len(labels), count)]
    k = min(top_k, count)
    values, indices = torch.topk(probs, k=k)
    rows: list[dict] = []
    for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        label = labels[int(index)]
        row: dict = {"label": label.name, "score": float(score), "rank": rank}
        if label.description:
            row["description"] = label.description
        rows.append(row)
    return rows
