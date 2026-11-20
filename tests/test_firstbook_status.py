"""/api/status, home and navigation, library paths, planned models."""

from __future__ import annotations

import json

from app import create_app
from firstbook.library import Library, safe_child
from modelwidget import catalog
from tests.firstbook_helpers import image_settings, make_library, make_mbtiles


def test_status_reports_content_and_capabilities(tmp_path):
    lib = make_library(tmp_path)
    make_mbtiles(lib.maps_dir / "uk-osm.mbtiles")
    (lib.videos_dir / "a.mp4").write_bytes(b"x")
    (lib.guides_dir / "g.md").write_text("# G (placeholder)\n", encoding="utf-8")
    client = create_app(image_settings(tmp_path), library=lib).test_client()
    body = client.get("/api/status").get_json()
    assert body["offline"] is True
    assert body["maps"]["count"] == 1 and body["videos"]["count"] == 1
    assert body["guides"] == {"count": 1, "placeholders": 1}
    models = body["models"]
    data = catalog.load_catalog()
    curated = [m for m in data["models"] if m.get("source") != "user"]
    assert len(curated) == 8
    assert models["catalog_count"] == len(data["models"])
    assert models["planned"] == []
    assert sum(1 for m in curated if catalog.category_of(m) == "llm") == 5
    assert models["by_category"]["vision"] == 2
    assert "ram_total_gb" in body["capabilities"]
    assert "llama_cpp" in body["capabilities"]["optional_backends"]


def test_home_and_nav_on_every_page(tmp_path):
    client = create_app(image_settings(tmp_path), library=make_library(tmp_path)).test_client()
    for url in ("/home", "/maps", "/videos", "/guides", "/"):
        page = client.get(url)
        assert page.status_code == 200, url
        assert 'class="fb-nav"' in page.get_data(as_text=True), url
    assert "The First Book" in client.get("/home").get_data(as_text=True)


def test_library_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path / "lib"))
    monkeypatch.setenv("VIDEOS_DIR", str(tmp_path / "usb" / "videos"))
    lib = Library.from_env()
    assert lib.maps_dir == (tmp_path / "lib" / "maps").resolve()
    assert lib.videos_dir == (tmp_path / "usb" / "videos").resolve()


def test_safe_child_blocks_escape(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "ok.mp4").write_bytes(b"x")
    (tmp_path / "secret").write_text("x")
    assert safe_child(tmp_path / "in", "ok.mp4") is not None
    assert safe_child(tmp_path / "in", "../secret") is None
    assert safe_child(tmp_path / "in", str(tmp_path / "secret")) is None


def test_gguf_entries_are_downloadable_single_files():
    data = catalog.load_catalog(catalog.CATALOG_PATH)
    curated_gguf = [
        m for m in data["models"] if m.get("gguf_file") and m.get("source") != "user"
    ]
    assert len(curated_gguf) == 4 and data["planned"] == []
    for entry in data["models"]:
        if not entry.get("gguf_file"):
            continue
        assert entry["status"] == "downloadable" and entry["backend"] == "llama_cpp"
        assert entry["gguf_file"].endswith(".gguf") and len(entry["revision"]) == 40


def test_planned_models_are_never_downloaded_or_selected(tmp_path):
    planned = {"id": "future", "name": "Future", "hf_id": "a/b", "revision": "r" * 40,
               "task": "text-generation", "backend": "llama_cpp", "status": "planned"}
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"models": [planned]}), encoding="utf-8")
    client = create_app(image_settings(tmp_path), override_env_path=tmp_path / ".env",
                        catalog_file=path).test_client()
    body = client.get("/api/models").get_json()
    assert [p["id"] for p in body["planned"]] == ["future"]
    assert client.post("/api/models/future/download").status_code == 409
    assert client.post("/api/models/select", json={"id": "future"}).status_code == 404


def test_unknown_status_is_invalid(tmp_path):
    path = tmp_path / "c.json"
    entry = {"id": "x", "name": "X", "hf_id": "a/b", "revision": "r", "task": "text-generation",
             "backend": "transformers", "status": "maybe"}
    path.write_text(json.dumps({"models": [entry]}), encoding="utf-8")
    data = catalog.load_catalog(path)
    assert data["models"] == [] and data["invalid"][0]["error"].startswith("unknown status")
    assert catalog.category_of({"task": "image-classification"}) == "vision"
