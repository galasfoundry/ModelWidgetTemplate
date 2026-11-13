"""Class labels for image and text classifiers.

``LABEL_MAP_PATH`` may be a JSON list, a map of class index to a name, or a
map of class index to ``{"name": "...", "description": "..."}``. A woodsapp
app-config file (an object with a ``label_map`` field) is accepted too.
When no file is set and the head has 1,000 classes, ImageNet-1k names from
torchvision are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassLabel:
    name: str
    description: str | None = None


def _as_label(item: object, index: int) -> ClassLabel:
    if isinstance(item, str):
        return ClassLabel(item)
    if isinstance(item, dict):
        name = item.get("name") or item.get("label") or f"class_{index}"
        description = item.get("description")
        if description in (None, "", "N/A"):
            description = None
        return ClassLabel(str(name), None if description is None else str(description))
    return ClassLabel(str(item))


def parse_label_map(data: object, num_classes: int) -> list[ClassLabel]:
    """Turn JSON data into one label per class index."""
    if isinstance(data, dict) and isinstance(data.get("label_map"), (dict, list)):
        data = data["label_map"]
    labels = [ClassLabel(f"class_{index}") for index in range(num_classes)]
    if isinstance(data, list):
        for index, item in enumerate(data):
            if index >= num_classes:
                break
            labels[index] = _as_label(item, index)
        return labels
    if isinstance(data, dict):
        for key, item in data.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= index < num_classes:
                labels[index] = _as_label(item, index)
        return labels
    raise ValueError("LABEL_MAP_PATH must contain a JSON list or object.")


def imagenet_1k_labels() -> list[ClassLabel] | None:
    """Return the 1,000 ImageNet class names bundled with torchvision."""
    try:
        from torchvision.models import MobileNet_V3_Small_Weights
    except Exception:
        return None
    categories = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta.get("categories")
    if not categories or len(categories) != 1000:
        return None
    return [ClassLabel(str(name)) for name in categories]


def labels_from_id2label(id2label: dict, num_labels: int | None = None) -> list[ClassLabel]:
    """Build one label per logit from a transformers ``id2label`` map."""
    numeric_keys: list[int] = []
    for key in id2label:
        try:
            numeric_keys.append(int(key))
        except (TypeError, ValueError):
            continue
    declared = int(num_labels or 0)
    if declared <= 0 and numeric_keys:
        declared = max(numeric_keys) + 1
    labels: list[ClassLabel] = []
    for index in range(declared):
        name = id2label.get(index, id2label.get(str(index), f"class_{index}"))
        labels.append(ClassLabel(str(name)))
    return labels


def resolve_labels(path: Path | None, num_classes: int) -> list[ClassLabel]:
    """Load labels from disk, or fall back to ImageNet-1k, then ``class_<n>``."""
    if num_classes < 0:
        raise ValueError("num_classes must be zero or greater.")
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"LABEL_MAP_PATH does not exist: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return parse_label_map(data, num_classes)
    if num_classes == 1000:
        imagenet = imagenet_1k_labels()
        if imagenet is not None:
            return imagenet
    return [ClassLabel(f"class_{index}") for index in range(num_classes)]
