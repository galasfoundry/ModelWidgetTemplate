"""Image classification with timm or transformers.

The timm adapter reads ``config.json`` from the snapshot (architecture,
input size, normalization) and loads ``model.safetensors`` on CPU before
moving the module. That is the same shape as the woodsapp loader, with the
repo id taken from configuration.

The transformers adapter uses ``AutoImageProcessor`` and
``AutoModelForImageClassification`` with ``low_cpu_mem_usage``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from modelwidget.adapters.base import ModelAdapter, register, topk_rows
from modelwidget.adapters.loading import from_pretrained
from modelwidget.labels import ClassLabel, labels_from_id2label, resolve_labels


def _checkpoint_file(snapshot: Path) -> Path:
    safe = snapshot / "model.safetensors"
    if safe.is_file():
        return safe
    binary = snapshot / "pytorch_model.bin"
    if binary.is_file():
        return binary
    safetensors = sorted(path for path in snapshot.glob("*.safetensors") if path.is_file())
    if safetensors:
        return safetensors[0]
    raise FileNotFoundError(f"No checkpoint file in {snapshot}")


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.name.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {path} did not contain a state dict.")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = key[len("module.") :] if key.startswith("module.") else key
        cleaned[name] = value
    return cleaned


def _best_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_keys = set(model.state_dict())
    candidates = [state]
    keys = list(state)
    if keys and all(key.startswith("model.") for key in keys):
        candidates.append({key[len("model.") :]: value for key, value in state.items()})

    def score(candidate: dict[str, torch.Tensor]) -> int:
        return len(model_keys & set(candidate))

    return max(candidates, key=score)


def _interpolation(name: str):
    table = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "nearest": transforms.InterpolationMode.NEAREST,
    }
    return table.get(name.lower(), transforms.InterpolationMode.BICUBIC)


def _timm_transform(pretrained: dict) -> transforms.Compose:
    size = pretrained.get("input_size") or [3, 224, 224]
    image_size = int(size[-1])
    mean = pretrained.get("mean") or [0.485, 0.456, 0.406]
    std = pretrained.get("std") or [0.229, 0.224, 0.225]
    crop_pct = pretrained.get("crop_pct") or 1.0
    interpolation = _interpolation(str(pretrained.get("interpolation") or "bicubic"))
    steps: list = []
    if crop_pct and float(crop_pct) < 1:
        resize_to = int(math.floor(image_size / float(crop_pct)))
        steps.append(transforms.Resize(resize_to, interpolation=interpolation))
        steps.append(transforms.CenterCrop(image_size))
    else:
        steps.append(transforms.Resize((image_size, image_size), interpolation=interpolation))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(steps)


@register("image-classification", "timm")
class TimmImageClassifier(ModelAdapter):
    input_kind = "image"

    def __init__(self, settings, snapshot: Path):
        self.settings = settings
        self.snapshot = Path(snapshot)
        self.model = None
        self.transform = None
        self.labels: list[ClassLabel] = []
        self.device = "cpu"
        self.dtype = torch.float32

    def load(self, *, device: str, dtype: torch.dtype) -> None:
        import timm

        config_path = self.snapshot / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"timm snapshot is missing config.json: {self.snapshot}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        architecture = config.get("architecture")
        if not architecture:
            raise RuntimeError(f"{config_path} has no architecture field.")
        num_classes = int(config.get("num_classes") or 1000)
        model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
        state = _best_state(model, _load_state_dict(_checkpoint_file(self.snapshot)))
        incompatible = model.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if "num_batches_tracked" not in key]
        if missing or incompatible.unexpected_keys:
            preview = ", ".join(missing[:6])
            unexpected = ", ".join(incompatible.unexpected_keys[:6])
            raise RuntimeError(
                "timm checkpoint did not match "
                f"{architecture}. Missing: {preview or 'none'}. Unexpected: {unexpected or 'none'}."
            )
        model.eval()
        model.to(device=device, dtype=dtype)
        self.model = model
        self.transform = _timm_transform(config.get("pretrained_cfg") or {})
        self.labels = resolve_labels(self.settings.label_map_path, num_classes)
        self.device = device
        self.dtype = dtype

    def predict(self, *, image=None, text: str | None = None) -> dict:
        if not isinstance(image, Image.Image):
            raise ValueError("This model expects an image.")
        if self.model is None or self.transform is None:
            raise RuntimeError("The timm adapter is not loaded.")
        batch = self.transform(image.convert("RGB")).unsqueeze(0)
        batch = batch.to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            logits = self.model(batch)
        probs = torch.softmax(logits[0].float(), dim=-1).cpu()
        return {"results": topk_rows(probs, self.labels, self.settings.top_k), "text": None}


@register("image-classification", "transformers")
class TransformersImageClassifier(ModelAdapter):
    input_kind = "image"

    def __init__(self, settings, snapshot: Path):
        self.settings = settings
        self.snapshot = Path(snapshot)
        self.model = None
        self.processor = None
        self.labels: list[ClassLabel] = []
        self.device = "cpu"
        self.dtype = torch.float32

    def load(self, *, device: str, dtype: torch.dtype) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.processor = AutoImageProcessor.from_pretrained(self.snapshot, local_files_only=True)
        self.model = from_pretrained(
            AutoModelForImageClassification,
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
        if not isinstance(image, Image.Image):
            raise ValueError("This model expects an image.")
        if self.model is None or self.processor is None:
            raise RuntimeError("The transformers image adapter is not loaded.")
        encoded = self.processor(images=image.convert("RGB"), return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if "pixel_values" in encoded and encoded["pixel_values"].dtype.is_floating_point:
            encoded["pixel_values"] = encoded["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        probs = torch.softmax(logits[0].float(), dim=-1).cpu()
        return {"results": topk_rows(probs, self.labels, self.settings.top_k), "text": None}
