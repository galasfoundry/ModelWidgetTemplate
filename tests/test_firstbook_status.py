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
    assert models["catalog_count"] == 4
    assert "qwen2.5-0.5b-instruct-gguf" in models["planned"]
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


def test_planned_models_are_never_installable(tmp_path):
    data = catalog.load_catalog(catalog.CATALOG_PATH)
    assert data["planned"], "expected planned GGUF entries"
    for entry in data["planned"]:
        assert entry["backend"] == "llama_cpp" and entry["gguf_file"].endswith(".gguf")
        assert len(entry["revision"]) == 40
    client = create_app(image_settings(tmp_path), override_env_path=tmp_path / ".env").test_client()
    body = client.get("/api/models").get_json()
    listed = {m["id"] for m in body["models"] + body["excluded"]}
    assert not listed & {p["id"] for p in body["planned"]}
    assert all(p["category"] == "llm" for p in body["planned"])
    response = client.post("/api/models/select", json={"id": data["planned"][0]["id"]})
    assert response.status_code == 404


def test_unknown_status_is_invalid(tmp_path):
    path = tmp_path / "c.json"
    entry = {"id": "x", "name": "X", "hf_id": "a/b", "revision": "r", "task": "text-generation",
             "backend": "transformers", "status": "maybe"}
    path.write_text(json.dumps({"models": [entry]}), encoding="utf-8")
    data = catalog.load_catalog(path)
    assert data["models"] == [] and data["invalid"][0]["error"].startswith("unknown status")
    assert catalog.category_of({"task": "image-classification"}) == "vision"
