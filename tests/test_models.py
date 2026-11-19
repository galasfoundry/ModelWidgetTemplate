"""Model catalog, capability detection, and the picker API. Offline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import paths
from app import create_app
from modelwidget import catalog
from modelwidget.capabilities import detect_capabilities
from modelwidget.settings import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, Settings

TINY = {
    "id": "tinystories-1m",
    "name": "TinyStories-1M",
    "hf_id": "roneneldan/TinyStories-1M",
    "revision": "77f1b168e219585646439073245fe87e56b3023e",
    "task": "text-generation",
    "backend": "transformers",
    "approx_size_mb": 5,
    "min_ram_gb": 2,
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_WEIGHTS", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))


def _settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        model_id=DEFAULT_MODEL_ID,
        model_revision=DEFAULT_MODEL_REVISION,
        model_task="image-classification",
        model_backend="timm",
        device="cpu",
        lazy_load=True,
        weights_dir=tmp_path / "weights",
    )


def _catalog(tmp_path: Path, models: list) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"version": 1, "models": models}), encoding="utf-8")
    return path


def _fake_snapshot(weights_dir: Path, entry: dict) -> Path:
    snap = weights_dir / paths.model_dir_name(entry["hf_id"]) / "snapshots" / entry["revision"]
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"x")
    return snap


def test_shipped_catalog_loads():
    data = catalog.load_catalog(catalog.CATALOG_PATH)
    ids = [m["id"] for m in data["models"]]
    assert ids == [
        "mobilenetv3-small", "tinystories-1m", "bedrock-sentiment-tiny", "mobilenetv2",
        "gemma-3-270m-it-gguf", "smollm2-360m-instruct-gguf",
        "qwen2.5-0.5b-instruct-gguf", "llama-3.2-1b-instruct-gguf",
    ]
    assert data["invalid"] == []


def test_capabilities_shape(tmp_path):
    caps = detect_capabilities(tmp_path / "missing" / "weights")
    assert caps["ram_total_gb"] is None or caps["ram_total_gb"] > 0
    assert caps["disk_free_mb"] is None or caps["disk_free_mb"] >= 0
    assert isinstance(caps["cuda_available"], bool)
    assert set(caps["backends"]) == {"timm", "transformers", "llama_cpp"}


def test_evaluate_filters_ram_disk_backend(tmp_path):
    caps = {"ram_total_gb": 3.8, "disk_free_mb": 100.0, "backends": {"timm": True, "transformers": False}}
    weights = tmp_path / "weights"
    ok = catalog.evaluate(dict(TINY, backend="timm", task="image-classification"), caps, weights)
    assert ok["compatible"] and not ok["installed"]
    big_ram = catalog.evaluate(dict(TINY, backend="timm", task="image-classification", min_ram_gb=8), caps, weights)
    assert not big_ram["compatible"] and "RAM" in big_ram["reasons"][0]
    big_disk = catalog.evaluate(dict(TINY, backend="timm", task="image-classification", approx_size_mb=500), caps, weights)
    assert not big_disk["compatible"]
    no_backend = catalog.evaluate(TINY, caps, weights)
    assert not no_backend["compatible"]
    bad_task = catalog.evaluate(dict(TINY, task="speech"), {"backends": {}}, weights)
    assert not bad_task["compatible"]


def test_malformed_entries_are_reported_not_fatal(tmp_path):
    path = _catalog(tmp_path, [TINY, {"id": "broken"}, TINY])
    data = catalog.load_catalog(path)
    assert [m["id"] for m in data["models"]] == ["tinystories-1m"]
    assert len(data["invalid"]) == 2


def test_write_override_keeps_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# keep me\nDEVICE=cpu\nMODEL_ID=old\nexport MODEL_TASK=old\nMODEL_ID=dupe\n", encoding="utf-8")
    catalog.write_override({"MODEL_ID": "new/model", "MODEL_TASK": "text-generation", "MODEL_BACKEND": "transformers"}, env)
    text = env.read_text(encoding="utf-8")
    assert text.splitlines() == [
        "# keep me",
        "DEVICE=cpu",
        "MODEL_ID=new/model",
        "MODEL_TASK=text-generation",
        "MODEL_BACKEND=transformers",
    ]


def test_api_models_lists_and_marks_active_and_installed(tmp_path):
    settings = _settings(tmp_path)
    mobilenet = dict(TINY, id="mnv3", name="MNV3", hf_id=DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION,
                     task="image-classification", backend="timm")
    _fake_snapshot(settings.weights_dir, TINY)
    path = _catalog(tmp_path, [mobilenet, TINY, dict(TINY, id="huge", min_ram_gb=4096)])
    client = create_app(settings, override_env_path=tmp_path / ".env", catalog_file=path).test_client()
    body = client.get("/api/models").get_json()
    assert [m["id"] for m in body["models"]] == ["mnv3", "tinystories-1m"]
    assert [m["id"] for m in body["excluded"]] == ["huge"]
    assert body["active"]["id"] == "mnv3"
    assert "tinystories-1m" in body["installed"]
    assert "capabilities" in body
    assert "model-picker" in client.get("/").get_data(as_text=True)


def test_select_bad_entry_fails_gracefully_and_keeps_model(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    bogus = dict(TINY, id="bogus", name="Bogus", hf_id="nobody/does-not-exist")

    def fail(*_args, **_kwargs):
        raise RuntimeError("404 Repository Not Found")

    monkeypatch.setattr(paths, "download_weights", fail)
    env = tmp_path / ".env"
    client = create_app(settings, override_env_path=env, catalog_file=_catalog(tmp_path, [bogus])).test_client()
    response = client.post("/api/models/select", json={"id": "bogus"})
    assert response.status_code == 502
    assert "could not be downloaded" in response.get_json()["error"]
    assert not env.exists()
    assert client.get("/api/info").get_json()["model_id"] == DEFAULT_MODEL_ID

    assert client.post("/api/models/select", json={"id": "nope"}).status_code == 404
    assert client.post("/api/models/select", json={}).status_code == 400


def test_select_refused_when_model_weights_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_WEIGHTS", str(tmp_path))
    client = create_app(_settings(tmp_path), override_env_path=tmp_path / ".env",
                        catalog_file=_catalog(tmp_path, [TINY])).test_client()
    assert client.post("/api/models/select", json={"id": "tinystories-1m"}).status_code == 409


def test_select_uses_snapshot_on_disk_without_download(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    _fake_snapshot(settings.weights_dir, TINY)
    calls = []
    monkeypatch.setattr(paths, "download_weights", lambda *a, **k: calls.append(a))

    loaded = []
    from modelwidget import service as service_module

    monkeypatch.setattr(service_module.ModelService, "ensure_loaded", lambda self: loaded.append(self.settings.model_id))
    for key in catalog.SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    client = create_app(settings, override_env_path=env, catalog_file=_catalog(tmp_path, [TINY])).test_client()
    response = client.post("/api/models/select", json={"id": "tinystories-1m"})
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["downloaded"] is False and calls == []
    assert loaded == [TINY["hf_id"]]
    assert f"MODEL_ID={TINY['hf_id']}" in env.read_text(encoding="utf-8")
    info = client.get("/api/info").get_json()
    assert info["model_id"] == TINY["hf_id"] and info["input_kind"] == "text"
