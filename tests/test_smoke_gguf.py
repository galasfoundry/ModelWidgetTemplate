"""Opt-in end-to-end check of the GGUF path: real download + one prompt.

Downloads gemma-3-270m Q4_K_M (~253 MB) into weights/ and needs
llama-cpp-python. Run with: RUN_GGUF_SMOKE=1 python -m pytest tests/test_smoke_gguf.py
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import replace

import pytest

from app import create_app
from modelwidget import catalog, downloads
from modelwidget.settings import Settings

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GGUF_SMOKE") != "1" or importlib.util.find_spec("llama_cpp") is None,
    reason="set RUN_GGUF_SMOKE=1 and pip install llama-cpp-python",
)


def test_gemma_270m_downloads_and_answers(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_WEIGHTS", raising=False)
    for key in catalog.SETTING_KEYS:
        monkeypatch.setenv(key, "")
    settings = Settings.from_env()
    entry = next(m for m in catalog.load_catalog()["models"] if m["id"] == "gemma-3-270m-it-gguf")
    snapshot, _ = downloads.download_entry(entry, settings.weights_dir)
    assert (snapshot / entry["gguf_file"]).is_file()
    app = create_app(replace(settings, max_new_tokens=24), override_env_path=tmp_path / ".env")
    client = app.test_client()
    assert client.post("/api/models/select", json={"id": entry["id"]}).status_code == 200
    reply = client.post("/api/predict", json={"text": "Say hello in five words."}).get_json()
    assert reply["backend"] == "llama_cpp" and reply["text"].strip()
