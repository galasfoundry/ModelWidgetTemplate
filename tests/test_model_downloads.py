"""Background model downloads, their endpoints and the CLI. All offline: the
Hugging Face hub is replaced by a fake that writes files in the cache layout."""

from __future__ import annotations

import json
import types
from dataclasses import replace
from pathlib import Path

import pytest

import paths
from app import create_app
from modelwidget import catalog, discover, downloads
from modelwidget.downloads import DownloadError, DownloadManager
from modelwidget.settings import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, Settings
from scripts import download_models

REV = "a" * 40
VISION = {"id": "tiny-vision", "name": "Tiny Vision", "hf_id": "org/tiny-vision", "revision": REV,
          "task": "image-classification", "backend": "timm", "category": "vision",
          "approx_size_mb": 1, "min_ram_gb": 1}
GGUF = {"id": "tiny-gguf", "name": "Tiny GGUF", "hf_id": "org/tiny-GGUF", "revision": REV,
        "gguf_file": "tiny-Q4_K_M.gguf", "task": "text-generation", "backend": "llama_cpp",
        "category": "llm", "status": "downloadable", "approx_size_mb": 1, "min_ram_gb": 1}
REPO_FILES = {
    "org/tiny-vision": {"config.json": b"{}", "model.safetensors": b"w" * 3000, "README.md": b"r",
                        "tf_model.h5": b"x"},
    "org/tiny-GGUF": {"tiny-Q4_K_M.gguf": b"g" * 5000, "tiny-Q8_0.gguf": b"h" * 9000},
}


class FakeHub:
    """Stands in for huggingface_hub: model_info + hf_hub_download, with progress."""

    def __init__(self):
        self.downloaded: list[str] = []
        self.fail: str | None = None

    def module(self):
        hub = self

        class HfApi:
            def model_info(self, repo_id, revision=None, files_metadata=False):
                files = REPO_FILES[repo_id]
                siblings = [types.SimpleNamespace(rfilename=n, size=len(b)) for n, b in files.items()]
                return types.SimpleNamespace(siblings=siblings, sha=revision)

        def hf_hub_download(repo_id, filename, revision, cache_dir, tqdm_class=None):
            if hub.fail:
                raise ConnectionError(hub.fail)
            data = REPO_FILES[repo_id][filename]
            snap = Path(cache_dir) / paths.model_dir_name(repo_id) / "snapshots" / revision
            snap.mkdir(parents=True, exist_ok=True)
            if tqdm_class is not None:
                bar = tqdm_class(total=len(data), initial=0)
                for start in range(0, len(data), 1000):
                    bar.update(min(1000, len(data) - start))
                bar.close()
            (snap / filename).write_bytes(data)
            hub.downloaded.append(filename)
            return str(snap / filename)

        return types.SimpleNamespace(HfApi=HfApi, hf_hub_download=hf_hub_download)


@pytest.fixture()
def hub(monkeypatch, tmp_path):
    fake = FakeHub()
    monkeypatch.setattr(downloads, "_hub", fake.module)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hfcache"))
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODEL_WEIGHTS"):
        monkeypatch.delenv(key, raising=False)
    return fake


def _catalog(tmp_path, models):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"version": 1, "models": models}), encoding="utf-8")
    return path


def _settings(tmp_path) -> Settings:
    return replace(Settings.from_env(), model_id=DEFAULT_MODEL_ID, model_revision=DEFAULT_MODEL_REVISION,
                   model_task="image-classification", model_backend="timm", device="cpu",
                   lazy_load=True, weights_dir=tmp_path / "weights")


def test_download_entry_skips_extra_files_and_reports_progress(hub, tmp_path):
    seen = []
    plan = []
    snap, downloaded = downloads.download_entry(VISION, tmp_path / "w", on_plan=plan.extend,
                                                on_bytes=seen.append)
    assert downloaded and sorted(hub.downloaded) == ["config.json", "model.safetensors"]
    assert seen[-1] == 3002 and seen == sorted(seen) and dict(plan)["model.safetensors"] == 3000
    assert snap.name == REV and (snap / "model.safetensors").is_file()
    # Second call: on disk, no network.
    hub.fail = "should not be called"
    assert downloads.download_entry(VISION, tmp_path / "w") == (snap.resolve(), False)


def test_progress_never_exceeds_total_with_two_bars(hub, tmp_path, monkeypatch):
    real = hub.module()

    def doubled():
        mod = real

        def hf_hub_download(repo_id, filename, revision, cache_dir, tqdm_class=None):
            size = len(REPO_FILES[repo_id][filename])
            for _ in range(2):  # e.g. network bar + write bar
                bar = tqdm_class(total=size)
                bar.update(size)
            return mod.hf_hub_download(repo_id, filename, revision, cache_dir)

        return types.SimpleNamespace(HfApi=mod.HfApi, hf_hub_download=hf_hub_download)

    monkeypatch.setattr(downloads, "_hub", doubled)
    seen = []
    downloads.download_entry(GGUF, tmp_path / "w", on_bytes=seen.append)
    assert max(seen) == 5000


def test_gguf_downloads_only_the_pinned_file(hub, tmp_path):
    snap, _ = downloads.download_entry(GGUF, tmp_path / "w")
    assert hub.downloaded == ["tiny-Q4_K_M.gguf"]
    assert catalog.entry_snapshot(GGUF, tmp_path / "w") == snap.resolve()
    assert downloads.disk_size_mb(snap) == round(5000 / 1024**2, 1)
    with pytest.raises(DownloadError, match="has no file"):
        downloads.download_entry(dict(GGUF, gguf_file="nope.gguf"), tmp_path / "w2")


def test_offline_flag_and_network_errors_are_explained(hub, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(DownloadError, match="HF_HUB_OFFLINE"):
        downloads.download_entry(VISION, tmp_path / "w")
    monkeypatch.delenv("HF_HUB_OFFLINE")
    hub.fail = "Max retries exceeded: connection refused"
    with pytest.raises(DownloadError, match="internet connection"):
        downloads.download_entry(VISION, tmp_path / "w")


def test_manager_runs_jobs_in_background(hub, tmp_path):
    manager = DownloadManager(tmp_path / "w")
    job = manager.submit(GGUF)
    assert job["state"] in ("queued", "downloading", "done")
    done = manager.wait("tiny-gguf", timeout=10)
    assert done["state"] == "done" and done["downloaded"] is True
    assert done["bytes_done"] == done["bytes_total"] == 5000 and done["percent"] == 100.0
    assert done["files_done"] == done["files_total"] == 1
    hub.fail = "boom"
    bad = manager.submit(VISION)
    assert bad["id"] == "tiny-vision"
    failed = manager.wait("tiny-vision", timeout=10)
    assert failed["state"] == "error" and "boom" in failed["error"]
    assert {j["id"] for j in manager.all()} == {"tiny-gguf", "tiny-vision"}


def test_download_endpoints(hub, tmp_path):
    path = _catalog(tmp_path, [VISION, GGUF, dict(VISION, id="huge", min_ram_gb=4096)])
    app = create_app(_settings(tmp_path), override_env_path=tmp_path / ".env", catalog_file=path)
    client = app.test_client()
    assert client.get("/api/models/tiny-gguf/status").get_json()["state"] == "idle"
    started = client.post("/api/models/tiny-gguf/download")
    assert started.status_code == 202
    app.config["DOWNLOADS"].wait("tiny-gguf", timeout=10)
    status = client.get("/api/models/tiny-gguf/status").get_json()
    assert status["state"] == "done" and status["installed"] is True
    assert client.get("/api/models/downloads").get_json()["active"] == 0
    again = client.post("/api/models/tiny-gguf/download")
    assert again.status_code == 200 and again.get_json()["installed"] is True
    assert client.post("/api/models/huge/download").status_code == 409
    assert client.post("/api/models/nope/download").status_code == 404
    body = client.get("/api/models").get_json()
    rows = {m["id"]: m for m in body["models"] + body["excluded"]}
    assert rows["tiny-gguf"]["installed"] and rows["tiny-gguf"]["download"]["state"] == "done"
    assert rows["tiny-vision"]["downloadable"] and not rows["tiny-vision"]["installed"]


def test_select_refuses_while_downloading(hub, tmp_path, monkeypatch):
    path = _catalog(tmp_path, [VISION])
    app = create_app(_settings(tmp_path), override_env_path=tmp_path / ".env", catalog_file=path)
    monkeypatch.setattr(app.config["DOWNLOADS"], "busy", lambda model_id: True)
    response = app.test_client().post("/api/models/select", json={"id": "tiny-vision"})
    assert response.status_code == 409 and "still downloading" in response.get_json()["error"]


def test_gguf_needs_llama_cpp_to_activate_but_not_to_download(tmp_path):
    caps = {"ram_total_gb": 3.7, "disk_free_mb": 10000.0,
            "backends": {"timm": True, "transformers": True, "llama_cpp": False}}
    row = catalog.evaluate(dict(GGUF, approx_size_mb=808), caps, tmp_path)
    assert row["downloadable"] and not row["compatible"]
    assert "llama-cpp-python" in row["reasons"][0]
    assert row["warnings"] and "close other apps" in row["warnings"][0]
    ok = catalog.evaluate(GGUF, dict(caps, backends={"llama_cpp": True}), tmp_path)
    assert ok["compatible"] and ok["warnings"] == []


def test_select_writes_model_file_for_gguf(hub, tmp_path, monkeypatch):
    from modelwidget import service as service_module

    downloads.download_entry(GGUF, tmp_path / "weights")
    monkeypatch.setattr(service_module.ModelService, "ensure_loaded", lambda self: None)
    monkeypatch.setattr("app.detect_capabilities", lambda weights: {
        "ram_total_gb": 8, "disk_free_mb": 1e6, "backends": {"llama_cpp": True}})
    for key in catalog.SETTING_KEYS:
        # setenv first so monkeypatch restores the original state after the
        # app writes these keys into os.environ.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    env = tmp_path / ".env"
    client = create_app(_settings(tmp_path), override_env_path=env,
                        catalog_file=_catalog(tmp_path, [GGUF])).test_client()
    response = client.post("/api/models/select", json={"id": "tiny-gguf"})
    assert response.status_code == 200, response.get_json()
    text = env.read_text(encoding="utf-8")
    assert "MODEL_FILE=tiny-Q4_K_M.gguf" in text and "MODEL_BACKEND=llama_cpp" in text
    assert hub.downloaded == ["tiny-Q4_K_M.gguf"]


def test_llama_adapter_picks_the_model_file(tmp_path):
    from modelwidget.adapters.llama_cpp import LlamaCppTextGenerator

    settings = replace(_settings(tmp_path), model_file="")
    (tmp_path / "a.gguf").write_bytes(b"x")
    assert LlamaCppTextGenerator(settings, tmp_path).model_path().name == "a.gguf"
    (tmp_path / "b.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="MODEL_FILE"):
        LlamaCppTextGenerator(settings, tmp_path).model_path()
    chosen = replace(settings, model_file="b.gguf")
    assert LlamaCppTextGenerator(chosen, tmp_path).model_path().name == "b.gguf"


def test_cli_list_dry_run_and_download(hub, tmp_path, capsys):
    path = _catalog(tmp_path, [VISION, GGUF])
    weights = tmp_path / "w"
    common = ["--weights-dir", str(weights), "--catalog", str(path)]
    assert download_models.main(["--list", *common]) == 0
    out = capsys.readouterr().out
    assert "tiny-gguf" in out and "not downloaded" in out
    assert download_models.main(["--all", "--dry-run", *common]) == 0
    assert hub.downloaded == [] and "would download" in capsys.readouterr().out
    assert download_models.main(["tiny-gguf", *common]) == 0
    assert hub.downloaded == ["tiny-Q4_K_M.gguf"]
    assert "done" in capsys.readouterr().out
    assert download_models.main(["tiny-gguf", *common]) == 0
    assert "already on disk" in capsys.readouterr().out
    assert download_models.main(["missing", *common]) == 2
    hub.fail = "Max retries exceeded"
    assert download_models.main(["tiny-vision", *common]) == 1
    assert "FAILED" in capsys.readouterr().err


def test_discovered_model_is_listed_after_download_and_restart(hub, tmp_path, monkeypatch):
    """A search pick is a normal GGUF catalog entry: on disk, in the picker, and after restart."""
    repo = "someone/Small-Chat-GGUF"
    filename = "small-Q4_K_M.gguf"
    REPO_FILES[repo] = {filename: b"g" * 4000}
    catalog_file = _catalog(tmp_path, [VISION])
    candidate = {
        "hf_id": repo,
        "name": "Small Chat (GGUF Q4_K_M)",
        "revision": REV,
        "gguf_file": filename,
        "quant": "Q4_K_M",
        "approx_size_mb": 1,
        "estimated_ram_mb": 800,
        "fit": "fits",
        "reason": "about 800 MB RAM, file is 1 MB",
        "license": "apache-2.0",
    }
    entry = discover.add_candidate(candidate, catalog_file)
    assert entry["source"] == "user"
    assert entry["backend"] == "llama_cpp"
    assert entry["gguf_file"] == filename
    assert json.loads(catalog_file.read_text(encoding="utf-8"))["models"] == [VISION]

    monkeypatch.setattr("app.detect_capabilities", lambda weights_dir: {
        "ram_total_gb": 4.0, "disk_free_mb": 20000.0, "cuda_available": False,
        "backends": {"timm": True, "transformers": True, "llama_cpp": True},
    })
    from modelwidget import service as service_module

    monkeypatch.setattr(service_module.ModelService, "ensure_loaded", lambda self: None)
    for key in catalog.SETTING_KEYS:
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)

    settings = _settings(tmp_path)
    env = tmp_path / ".env"
    app = create_app(settings, override_env_path=env, catalog_file=catalog_file)
    client = app.test_client()
    started = client.post(f"/api/models/{entry['id']}/download")
    assert started.status_code == 202
    assert app.config["DOWNLOADS"].wait(entry["id"], timeout=10)["state"] == "done"
    listed = client.get("/api/models").get_json()
    rows = {row["id"]: row for row in listed["models"]}
    assert entry["id"] in rows
    assert rows[entry["id"]]["installed"] is True
    assert rows[entry["id"]]["source"] == "user"
    assert entry["id"] not in {row["id"] for row in listed["excluded"]}

    # A new process reads the same user catalog and the file already on disk.
    hub.fail = "should not touch the network"
    restarted = create_app(settings, override_env_path=env, catalog_file=catalog_file)
    again = restarted.test_client().get("/api/models").get_json()
    again_rows = {row["id"]: row for row in again["models"]}
    assert again_rows[entry["id"]]["installed"] is True
    assert again_rows[entry["id"]]["download"] is None

    chosen = restarted.test_client().post("/api/models/select", json={"id": entry["id"]})
    assert chosen.status_code == 200, chosen.get_json()
    text = env.read_text(encoding="utf-8")
    assert f"MODEL_FILE={filename}" in text
    assert "MODEL_BACKEND=llama_cpp" in text
    assert f"MODEL_ID={repo}" in text
    active = restarted.test_client().get("/api/models").get_json()
    assert active["active"]["id"] == entry["id"]
    picked = next(row for row in active["models"] if row["id"] == entry["id"])
    assert picked["active"] is True


def test_stale_partials_are_removed_before_retry(hub, tmp_path):
    import os
    import time

    blobs = tmp_path / "w" / paths.model_dir_name(GGUF["hf_id"]) / "blobs"
    blobs.mkdir(parents=True)
    old = blobs / "abc.1234.incomplete"
    old.write_bytes(b"x" * 100)
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    fresh = blobs / "def.5678.incomplete"
    fresh.write_bytes(b"y")
    downloads.download_entry(GGUF, tmp_path / "w")
    assert not old.exists() and fresh.exists()
