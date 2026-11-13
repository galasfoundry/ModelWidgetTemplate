"""The shipped adapters stay registered under stable task names."""

from __future__ import annotations

import pytest

import modelwidget.adapters  # noqa: F401  (registers adapters)
from modelwidget.adapters.base import REGISTRY, get_adapter_class
from modelwidget.devices import resolve_device, resolve_dtype
from modelwidget.settings import Settings, infer_backend


def test_shipped_adapters_cover_image_and_text():
    assert ("image-classification", "timm") in REGISTRY
    assert ("image-classification", "transformers") in REGISTRY
    assert ("text-generation", "transformers") in REGISTRY
    assert ("text-classification", "transformers") in REGISTRY
    assert get_adapter_class("image-classification", "timm").input_kind == "image"
    assert get_adapter_class("text-generation", "transformers").input_kind == "text"


def test_unknown_adapter_names_the_registered_ones():
    with pytest.raises(ValueError) as caught:
        get_adapter_class("speech", "transformers")
    message = str(caught.value)
    assert "speech" in message
    assert "image-classification via timm" in message


def test_backend_inference_follows_the_repo_org():
    assert infer_backend("image-classification", "timm/mobilenetv3_small_100.lamb_in1k") == "timm"
    assert infer_backend("image-classification", "google/mobilenet_v2_1.0_224") == "transformers"
    assert infer_backend("text-generation", "roneneldan/TinyStories-1M") == "transformers"


def test_cpu_half_precision_stays_on_float32_unless_allowed():
    assert resolve_device("cpu") == "cpu"
    dtype, name = resolve_dtype("float16", "cpu", allow_cpu_half=False)
    assert name == "float32"
    dtype, name = resolve_dtype("float16", "cpu", allow_cpu_half=True)
    assert name == "float16"
    assert dtype is not None


def test_settings_defaults_point_at_the_tiny_timm_model(monkeypatch):
    for key in (
        "MODEL_ID",
        "MODEL_REVISION",
        "MODEL_TASK",
        "MODEL_BACKEND",
        "DEVICE",
        "TORCH_DTYPE",
        "LABEL_MAP_PATH",
        "WEIGHTS_DIR",
        "MODEL_WEIGHTS",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env()
    assert settings.model_id == "timm/mobilenetv3_small_100.lamb_in1k"
    assert settings.model_revision == "76b303ec28c466e9df37f4745dfb7aef20ab87ae"
    assert settings.model_task == "image-classification"
    assert settings.model_backend == "timm"
    assert settings.device == "auto"
    assert settings.lazy_load is True
    assert settings.low_cpu_mem_usage is True
    assert settings.weights_dir.name == "weights"
