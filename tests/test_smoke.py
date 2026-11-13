"""End-to-end predictions with the tiny models.

Each test downloads a snapshot into ``weights/`` on first run and then loads
it through the Flask app. Later runs reuse that snapshot.
"""

from __future__ import annotations

import gc
import io
from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from app import create_app
from modelwidget.settings import Settings


@pytest.fixture(autouse=True)
def _no_weight_override(monkeypatch):
    monkeypatch.delenv("MODEL_WEIGHTS", raising=False)


def _settings(**kwargs) -> Settings:
    base = Settings.from_env()
    values = {
        "device": "cpu",
        "torch_dtype": "float32",
        "lazy_load": True,
        "low_cpu_mem_usage": True,
        "allow_cpu_half": False,
        "top_k": 5,
        "max_new_tokens": 16,
    }
    values.update(kwargs)
    return replace(base, **values)


def _jpeg() -> bytes:
    image = Image.new("RGB", (256, 256), (30, 120, 180))
    ImageDraw.Draw(image).ellipse((70, 50, 190, 170), fill=(220, 180, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _release(app) -> None:
    app.config["SERVICE"]._adapter = None
    app.config["SERVICE"].snapshot = None
    del app
    gc.collect()


def test_timm_image_classification_endpoint():
    app = create_app(
        _settings(
            model_id="timm/mobilenetv3_small_100.lamb_in1k",
            model_revision="76b303ec28c466e9df37f4745dfb7aef20ab87ae",
            model_task="image-classification",
            model_backend="timm",
        )
    )
    client = app.test_client()
    response = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(_jpeg()), "sample.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["backend"] == "timm"
    assert payload["device"] == "cpu"
    assert payload["dtype"] == "float32"
    assert len(payload["results"]) == 5
    scores = [row["score"] for row in payload["results"]]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 <= scores[0] <= 1.0
    labels = [row["label"] for row in payload["results"]]
    assert "balloon" in labels
    _release(app)


def test_transformers_image_classification_endpoint():
    app = create_app(
        _settings(
            model_id="google/mobilenet_v2_1.0_224",
            model_revision="75e607b00aeae1297cc89d026a118bce012f5c5a",
            model_task="image-classification",
            model_backend="transformers",
        )
    )
    client = app.test_client()
    response = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(_jpeg()), "sample.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["backend"] == "transformers"
    assert payload["task"] == "image-classification"
    assert payload["results"]
    assert payload["results"][0]["label"]
    assert payload["results"][0]["score"] >= payload["results"][-1]["score"]
    _release(app)


def test_text_generation_endpoint():
    app = create_app(
        _settings(
            model_id="roneneldan/TinyStories-1M",
            model_revision="77f1b168e219585646439073245fe87e56b3023e",
            model_task="text-generation",
            model_backend="transformers",
        )
    )
    client = app.test_client()
    prompt = "Once upon a time"
    response = client.post("/api/predict", json={"text": prompt})
    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["task"] == "text-generation"
    assert payload["input_kind"] == "text"
    assert isinstance(payload["text"], str)
    assert len(payload["text"].strip()) > len(prompt)
    assert payload["results"][0]["label"] == "completion"
    _release(app)


def test_text_classification_endpoint():
    app = create_app(
        _settings(
            model_id="Facececersek/bedrock-sentiment-tiny",
            model_revision="16175109a6e476cee0c388d2aac37429ecf964dd",
            model_task="text-classification",
            model_backend="transformers",
            top_k=2,
        )
    )
    client = app.test_client()
    response = client.post("/api/predict", json={"text": "The homemade soup was warm and comforting."})
    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["task"] == "text-classification"
    labels = {row["label"] for row in payload["results"]}
    assert labels == {"POSITIVE", "NEGATIVE"}
    assert payload["results"][0]["label"] == "POSITIVE"
    assert all(0.0 <= row["score"] <= 1.0 for row in payload["results"])
    assert abs(sum(row["score"] for row in payload["results"]) - 1.0) < 0.01
    _release(app)
