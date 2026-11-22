"""Hugging Face model discovery: fixtures only, no live network."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from app import create_app
from modelwidget import catalog, discover
from modelwidget.discover import DiscoveryError
from modelwidget.settings import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, Settings

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf_discover"

CAPS_4G = {
    "ram_total_gb": 4.0,
    "ram_available_gb": 2.5,
    "disk_free_mb": 20_000.0,
    "cuda_available": False,
    "backends": {"timm": True, "transformers": True, "llama_cpp": True},
}

CAPS_TINY_DISK = {**CAPS_4G, "disk_free_mb": 200.0}


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def fake_fetch(url: str, token: str | None = None, timeout: int = 20):
    parsed = urlparse(url)
    path = unquote(parsed.path)
    if path == "/api/models":
        return _load("search.json")
    if path.endswith("/tree") or "/tree/" in path:
        # /api/models/<repo>/tree/<rev>
        rest = path[len("/api/models/") :]
        repo, _, _rev = rest.partition("/tree/")
        trees = _load("trees.json")
        if repo not in trees:
            raise DiscoveryError(f"no fixture tree for {repo}")
        return trees[repo]
    if path.startswith("/api/models/"):
        repo = path[len("/api/models/") :]
        details = _load("details.json")
        if repo not in details:
            raise DiscoveryError(f"no fixture detail for {repo}")
        return details[repo]
    raise DiscoveryError(f"unexpected url {url}")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_WEIGHTS", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
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


def _catalog(tmp_path: Path, models: list | None = None) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"version": 1, "models": models or []}), encoding="utf-8")
    return path


def test_hub_urls_keep_repo_slash():
    assert "/bartowski/Qwen2.5-1.5B-Instruct-GGUF" in discover._detail_url(
        "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    )
    assert "%2F" not in discover._detail_url("bartowski/Qwen2.5-1.5B-Instruct-GGUF")
    tree = discover._tree_url("bartowski/Qwen2.5-1.5B-Instruct-GGUF", "abc123")
    assert "/bartowski/Qwen2.5-1.5B-Instruct-GGUF/tree/abc123" in tree


def test_estimate_is_calibrated_to_measured_rss():
    """size × 1.5 + 700 MB stays at or above measured process RSS."""
    # Gemma 3 270M Q4_K_M: 241 MB file, 672–728 MB RSS in this app.
    gemma = discover.estimated_ram_mb(241)
    assert gemma >= 728
    # Qwen 2.5 1.5B Q4_K_M: 940 MB file, 2091 MB RSS; 2381 MB was a two-model switch peak.
    qwen = discover.estimated_ram_mb(940)
    assert qwen >= 2091
    assert qwen == pytest.approx(940 * 1.5 + 700)
    # Conservative for the smaller model, close to the 1.5B measurement.
    assert gemma == pytest.approx(241 * 1.5 + 700)


def test_1_5b_is_tight_and_3b_will_not_fit_on_4gb():
    label, estimate, reason = discover.assess_fit(940, CAPS_4G)
    assert label == "tight"
    assert estimate == pytest.approx(940 * 1.5 + 700)
    assert "4.0 GB" in reason
    q3, _, _ = discover.assess_fit(1640, CAPS_4G)
    q4, _, _ = discover.assess_fit(2003, CAPS_4G)
    assert q3 == "won't fit"
    assert q4 == "won't fit"


def test_prefers_q4_k_m_when_it_fits():
    files = [
        {"path": "model-Q3_K_M.gguf", "size": 700 * 1024 * 1024},
        {"path": "model-Q4_K_M.gguf", "size": 800 * 1024 * 1024},
        {"path": "model-Q8_0.gguf", "size": 1500 * 1024 * 1024},
        {"path": "model-mmproj-f16.gguf", "size": 100 * 1024 * 1024},
    ]
    picked = discover._pick_file(files, CAPS_4G)
    assert picked["quant"] == "Q4_K_M"
    assert picked["fit"] == "fits"


def test_falls_back_to_q3_when_q4_will_not_fit():
    # Under the calibrated estimate, Q3 at ~1300 MB is tight on 4 GB and Q4 at ~1600 MB will not fit.
    files = [
        {"path": "model-Q3_K_M.gguf", "size": 1300 * 1024 * 1024},
        {"path": "model-Q4_K_M.gguf", "size": 1600 * 1024 * 1024},
        {"path": "model-Q5_K_M.gguf", "size": 2000 * 1024 * 1024},
    ]
    picked = discover._pick_file(files, CAPS_4G)
    assert picked["quant"] == "Q3_K_M"
    assert picked["fit"] == "tight"


def test_q5_only_with_spare_ram():
    # A ~600 MB Q5 is still comfortable on 4 GB, so it may be chosen when it is the only option.
    small = [{"path": "only-Q5_K_M.gguf", "size": 600 * 1024 * 1024}]
    picked = discover._pick_file(small, CAPS_4G)
    assert picked["quant"] == "Q5_K_M"
    assert picked["fit"] == "fits"
    # A larger Q5 that would only be tight must not beat a fitting Q3.
    both = [
        {"path": "also-Q5_K_M.gguf", "size": 1100 * 1024 * 1024},
        {"path": "also-Q3_K_M.gguf", "size": 800 * 1024 * 1024},
    ]
    picked = discover._pick_file(both, CAPS_4G)
    assert picked["quant"] == "Q3_K_M"


def test_wont_fit_when_disk_is_short():
    files = [{"path": "model-Q4_K_M.gguf", "size": 800 * 1024 * 1024}]
    picked = discover._pick_file(files, CAPS_TINY_DISK)
    assert picked["fit"] == "won't fit"
    assert "disk" in picked["reason"]


def test_search_filters_and_labels(tmp_path):
    catalog_file = _catalog(tmp_path)
    payload = discover.search_models(
        "qwen",
        CAPS_4G,
        fetch=fake_fetch,
        catalog_file=catalog_file,
    )
    ids = [row["hf_id"] for row in payload["results"]]
    assert "someone/all-minilm-L6-v2-GGUF" not in ids
    assert "someone/CustomCode-1B-Instruct-GGUF" not in ids
    assert "someone/Huge-70B-Instruct-GGUF" not in ids
    assert "someone/Secret-Chat-7B-GGUF" not in ids
    assert "unsloth/Llama-3.2-1B-Instruct-GGUF" in ids
    assert "bartowski/Qwen2.5-1.5B-Instruct-GGUF" in ids
    by_id = {row["hf_id"]: row for row in payload["results"]}
    llama = by_id["unsloth/Llama-3.2-1B-Instruct-GGUF"]
    assert llama["quant"] == "Q4_K_M"
    assert llama["fit"] == "fits"
    assert llama["license"] == "llama3.2"
    assert llama["gated"] is False
    qwen = by_id["bartowski/Qwen2.5-1.5B-Instruct-GGUF"]
    assert qwen["quant"] == "Q4_K_M"
    assert qwen["gguf_file"] == "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    assert qwen["fit"] == "tight"
    three = by_id["someone/Qwen2.5-3B-Instruct-GGUF"]
    # Both the Q3 and Q4 files are above the 4 GB ceiling, so the preferred Q4_K_M is shown as won't fit.
    assert three["quant"] == "Q4_K_M"
    assert three["fit"] == "won't fit"
    # Fits before tight before won't fit, then by downloads.
    assert ids[0] == "unsloth/Llama-3.2-1B-Instruct-GGUF"
    assert ids.index(qwen["hf_id"]) < ids.index(three["hf_id"])
    skipped = {row["id"]: row["reason"] for row in payload["skipped"]}
    assert "gated" in skipped["someone/Secret-Chat-7B-GGUF"]
    assert "trust_remote_code" in skipped["someone/CustomCode-1B-Instruct-GGUF"]
    assert "too large" in skipped["someone/Huge-70B-Instruct-GGUF"]


def test_gated_included_when_token_set(tmp_path):
    payload = discover.search_models(
        "qwen",
        CAPS_4G,
        token="hf_test",
        fetch=fake_fetch,
        catalog_file=_catalog(tmp_path),
    )
    ids = [row["hf_id"] for row in payload["results"]]
    assert "someone/Secret-Chat-7B-GGUF" in ids
    secret = next(row for row in payload["results"] if "Secret" in row["hf_id"])
    assert secret["gated"] is True
    assert secret["fit"] == "won't fit"


def test_cache_used_when_huggingface_is_down(tmp_path):
    catalog_file = _catalog(tmp_path)
    cache_file = tmp_path / "models_discover_cache.json"
    live = discover.discover(
        "llama 3.2 1b",
        CAPS_4G,
        fetch=fake_fetch,
        catalog_file=catalog_file,
        cache_file=cache_file,
    )
    assert live["online"] is True
    assert live["cached"] is False
    assert cache_file.is_file()

    def boom(url, token=None, timeout=20):
        raise DiscoveryError("Hugging Face could not be reached: timed out")

    offline = discover.discover(
        "llama 3.2 1b",
        CAPS_4G,
        fetch=boom,
        catalog_file=catalog_file,
        cache_file=cache_file,
    )
    assert offline["online"] is False
    assert offline["cached"] is True
    assert offline["cache_age_seconds"] is not None
    assert offline["results"][0]["hf_id"] == live["results"][0]["hf_id"]
    assert "could not be reached" in offline["error"].lower()

    mismatched = discover.discover(
        "something else",
        CAPS_4G,
        fetch=boom,
        catalog_file=catalog_file,
        cache_file=cache_file,
    )
    assert "not the new search" in mismatched["error"]
    assert mismatched["query"] == "llama 3.2 1b"


def test_add_writes_user_catalog_not_curated(tmp_path):
    curated = [
        {
            "id": "tinystories-1m",
            "name": "TinyStories-1M",
            "hf_id": "roneneldan/TinyStories-1M",
            "revision": "abc",
            "task": "text-generation",
            "backend": "transformers",
        }
    ]
    catalog_file = _catalog(tmp_path, curated)
    payload = discover.search_models(
        "llama", CAPS_4G, fetch=fake_fetch, catalog_file=catalog_file
    )
    llama = next(row for row in payload["results"] if "Llama-3.2-1B" in row["hf_id"])
    entry = discover.add_candidate(llama, catalog_file)
    user_path = discover.user_catalog_path(catalog_file)
    assert user_path.is_file()
    user_doc = json.loads(user_path.read_text(encoding="utf-8"))
    assert user_doc["models"][0]["id"] == entry["id"]
    assert entry["id"].startswith("user-")
    assert entry["source"] == "user"
    assert entry["hf_id"] == "unsloth/Llama-3.2-1B-Instruct-GGUF"
    assert entry["gguf_file"] == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    # Curated file is untouched.
    assert json.loads(catalog_file.read_text(encoding="utf-8"))["models"] == curated
    merged = catalog.load_catalog(catalog_file)
    ids = [item["id"] for item in merged["models"]]
    assert ids == ["tinystories-1m", entry["id"]]
    assert merged["models"][1]["source"] == "user"

    with pytest.raises(DiscoveryError, match="already"):
        discover.add_candidate(llama, catalog_file)


def test_remove_user_model_leaves_curated(tmp_path):
    catalog_file = _catalog(
        tmp_path,
        [
            {
                "id": "tinystories-1m",
                "name": "TinyStories-1M",
                "hf_id": "roneneldan/TinyStories-1M",
                "revision": "abc",
                "task": "text-generation",
                "backend": "transformers",
            }
        ],
    )
    payload = discover.search_models("llama", CAPS_4G, fetch=fake_fetch, catalog_file=catalog_file)
    llama = next(row for row in payload["results"] if "Llama-3.2-1B" in row["hf_id"])
    entry = discover.add_candidate(llama, catalog_file)
    with pytest.raises(DiscoveryError, match="curated"):
        discover.remove_user_model("tinystories-1m", catalog_file=catalog_file)
    with pytest.raises(DiscoveryError, match="Switch away"):
        discover.remove_user_model(entry["id"], catalog_file=catalog_file, active_id=entry["id"])
    removed = discover.remove_user_model(entry["id"], catalog_file=catalog_file)
    assert removed["id"] == entry["id"]
    assert catalog.load_catalog(catalog_file)["models"][0]["id"] == "tinystories-1m"
    assert discover.load_user_models(discover.user_catalog_path(catalog_file)) == []


def test_wont_fit_cannot_be_added(tmp_path):
    catalog_file = _catalog(tmp_path)
    payload = discover.search_models(
        "qwen", CAPS_4G, token="hf_test", fetch=fake_fetch, catalog_file=catalog_file
    )
    secret = next(row for row in payload["results"] if "Secret" in row["hf_id"])
    with pytest.raises(DiscoveryError, match="won't fit"):
        discover.add_candidate(secret, catalog_file)


def test_discover_api_and_cli(tmp_path, monkeypatch, capsys):
    catalog_file = _catalog(
        tmp_path,
        [
            {
                "id": "tinystories-1m",
                "name": "TinyStories-1M",
                "hf_id": "roneneldan/TinyStories-1M",
                "revision": "abc",
                "task": "text-generation",
                "backend": "transformers",
            }
        ],
    )
    monkeypatch.setattr(discover, "fetch_json", fake_fetch)
    # detect_capabilities would read this machine; pin 4 GB so labels match fixtures.
    monkeypatch.setattr(
        "modelwidget.capabilities.detect_capabilities",
        lambda weights_dir: CAPS_4G,
    )
    app = create_app(settings=_settings(tmp_path), catalog_file=catalog_file)
    client = app.test_client()

    response = client.get("/api/models/discover?q=llama+3.2+1b&task=chat")
    assert response.status_code == 200
    body = response.get_json()
    assert body["online"] is True
    assert body["results"]
    assert body["results"][0]["fit"] in {"fits", "tight", "won't fit"}

    added = client.post("/api/models/discover/add", json={"index": 1})
    assert added.status_code == 201, added.get_json()
    entry = added.get_json()["added"]
    listing = client.get("/api/models").get_json()
    ids = [row["id"] for row in listing["models"] + listing["excluded"]]
    assert entry["id"] in ids
    assert catalog_file.read_text(encoding="utf-8").count("Llama-3.2") == 0

    gone = client.post("/api/models/discover/remove", json={"id": entry["id"]})
    assert gone.status_code == 200
    listing = client.get("/api/models").get_json()
    ids = [row["id"] for row in listing["models"] + listing["excluded"]]
    assert entry["id"] not in ids

    # Offline path: cached results, plus a clear error.
    def boom(url, token=None, timeout=20):
        raise DiscoveryError("Hugging Face could not be reached: timed out")

    monkeypatch.setattr(discover, "fetch_json", boom)
    offline = client.get("/api/models/discover?q=llama+3.2+1b")
    assert offline.status_code == 200
    payload = offline.get_json()
    assert payload["online"] is False
    assert payload["cached"] is True
    assert payload["results"]

    from scripts.discover_models import main as discover_main

    monkeypatch.setattr(discover, "fetch_json", fake_fetch)
    assert discover_main(["--q", "llama 3.2 1b", "--catalog", str(catalog_file)]) == 0
    output = capsys.readouterr().out
    assert "Llama-3.2-1B" in output
    assert "Q4_K_M" in output


def _inkling():
    return _load("inkling.json")


def test_inkling_search_skips_bf16_drafter_and_warns(tmp_path):
    """The reported download was the small DSpark BF16 sidecar, not the model.

    That file finishes quickly and then cannot run as chat. The real GGUF in
    the same repo is far too big for 4 GB RAM, and the search should say so
    instead of offering the drafter.
    """
    pack = _inkling()
    repo = pack["repo"]

    def fetch(url, token=None, timeout=20):
        parsed = urlparse(url)
        path = unquote(parsed.path)
        if path == "/api/models":
            return [pack["listing"]]
        if path.endswith("/tree") or "/tree/" in path:
            return pack["tree"]
        if path.startswith("/api/models/"):
            return pack["detail"]
        raise DiscoveryError(f"unexpected url {url}")

    payload = discover.search_models("inkling", CAPS_4G, fetch=fetch, catalog_file=_catalog(tmp_path))
    assert [row["hf_id"] for row in payload["results"]] == [repo]
    row = payload["results"][0]
    assert row["gguf_file"] == "Inkling-2-Mini-XQ-Uncensored.gguf"
    assert "DSpark" not in row["gguf_file"] and "mmproj" not in row["gguf_file"]
    assert row["name"] == "Inkling 2 Mini XQ Uncensored (GGUF)"
    assert "BF16" not in row["name"]
    assert row["fit"] == "won't fit"
    assert "too big for this machine" in row["warning"]
    assert "4.0 GB" in row["warning"]
    assert "Q4_K_M" in row["warning"]
    with pytest.raises(DiscoveryError, match="won't fit"):
        discover.add_candidate(row, _catalog(tmp_path))


def test_smaller_quant_is_suggested_ahead_of_bf16_and_drafters():
    files = _inkling()["q4_over_bf16"]
    picked = discover._pick_file(files, CAPS_4G)
    assert picked["path"] == "Inkling-2-Mini-UD-Q4_K_M.gguf"
    assert picked["quant"] == "UD-Q4_K_M"
    assert picked["fit"] == "fits"
    assert "dspark" not in picked["path"].lower()


def test_sidecar_already_in_the_catalog_is_not_runnable(tmp_path, monkeypatch):
    """A BF16 drafter downloaded before this fix must not be offered as a chat model."""
    drafter = {
        "id": "user-inkling-2-mini-xq-uncensored-bf16",
        "name": "Inkling 2 Mini XQ Uncensored (GGUF BF16)",
        "hf_id": "IndrasMirror/Inkling-2-Mini-XQ-Uncensored",
        "revision": "e2a54ac8ca34b6fe9ac722094b9a020654f63832",
        "gguf_file": "Inkling-2-Mini-XQ-Uncensored-DSpark-BF16.gguf",
        "task": "text-generation",
        "backend": "llama_cpp",
        "status": "downloadable",
        "approx_size_mb": 618,
        "min_ram_gb": 2,
    }
    catalog_file = _catalog(tmp_path, [drafter])
    monkeypatch.setattr("app.detect_capabilities", lambda weights_dir: CAPS_4G)
    client = create_app(settings=_settings(tmp_path), catalog_file=catalog_file).test_client()
    listing = client.get("/api/models").get_json()
    assert [row["id"] for row in listing["models"]] == []
    excluded = {row["id"]: row for row in listing["excluded"]}
    assert drafter["id"] in excluded
    assert excluded[drafter["id"]]["downloadable"] is False
    assert any("drafter" in reason for reason in excluded[drafter["id"]]["reasons"])
    refused = client.post(f"/api/models/{drafter['id']}/download")
    assert refused.status_code == 409
    select = client.post("/api/models/select", json={"id": drafter["id"]})
    assert select.status_code == 409
    assert "drafter" in select.get_json()["error"]
