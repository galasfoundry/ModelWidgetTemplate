"""HTTP behavior that does not load a checkpoint."""

from __future__ import annotations

import io
from dataclasses import replace

from app import create_app
from modelwidget.settings import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, Settings


def _image_settings() -> Settings:
    base = Settings.from_env()
    return replace(
        base,
        model_id=DEFAULT_MODEL_ID,
        model_revision=DEFAULT_MODEL_REVISION,
        model_task="image-classification",
        model_backend="timm",
        device="cpu",
        lazy_load=True,
    )


def test_index_and_health_and_info():
    app = create_app(_image_settings())
    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "predict-form" in body
    assert client.get("/api/health").get_json()["status"] == "ok"
    info = client.get("/api/info").get_json()
    assert info["input_kind"] == "image"
    assert info["loaded"] is False
    assert info["model_id"] == DEFAULT_MODEL_ID


def test_missing_or_unreadable_image_is_400():
    client = create_app(_image_settings()).test_client()
    missing = client.post("/api/predict", data={})
    assert missing.status_code == 400
    assert "image" in missing.get_json()["error"].lower()

    bad = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"this is not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert bad.status_code == 400
    assert "read" in bad.get_json()["error"].lower()


def test_empty_text_is_400_before_any_download():
    settings = replace(
        _image_settings(),
        model_id="roneneldan/TinyStories-1M",
        model_revision="77f1b168e219585646439073245fe87e56b3023e",
        model_task="text-generation",
        model_backend="transformers",
    )
    client = create_app(settings).test_client()
    response = client.post("/api/predict", json={"text": "   "})
    assert response.status_code == 400
    assert "text" in response.get_json()["error"].lower()
