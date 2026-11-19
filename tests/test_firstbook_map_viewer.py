"""The /maps viewer: vendored assets are served, packs carry what the JS needs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app import create_app
from firstbook import maps
from scripts import fetch_demo_map
from tests.firstbook_helpers import gz, image_settings, make_library, make_mbtiles

ROOT = Path(__file__).resolve().parent.parent


def _client(tmp_path):
    lib = make_library(tmp_path)
    return create_app(image_settings(tmp_path), library=lib).test_client(), lib


def test_maps_page_includes_local_viewer_when_packs_exist(tmp_path):
    client, lib = _client(tmp_path)
    empty = client.get("/maps").get_data(as_text=True)
    assert 'id="map"' not in empty and "fetch_demo_map" in empty
    (lib.maps_dir / "uk-overview.pmtiles").write_bytes(b"PMTiles" + bytes(64))
    page = client.get("/maps").get_data(as_text=True)
    assert 'id="map"' in page and 'id="map-pack"' in page
    for asset in ("vendor/maplibre-gl/maplibre-gl.js", "vendor/maplibre-gl/maplibre-gl.css",
                  "vendor/pmtiles/pmtiles.js", "vendor/protomaps-basemaps/basemaps.js", "js/maps.js"):
        assert f"/static/{asset}" in page
    assert "Demo: UK overview" in page


def test_vendored_assets_are_served(tmp_path):
    client, _ = _client(tmp_path)
    for url in ("/static/vendor/maplibre-gl/maplibre-gl.js",
                "/static/vendor/pmtiles/pmtiles.js",
                "/static/vendor/basemaps-assets/fonts/Noto%20Sans%20Regular/0-255.pbf",
                "/static/vendor/basemaps-assets/sprites/v4/light@2x.png"):
        response = client.get(url)
        assert response.status_code == 200, url
        response.close()


def test_viewer_script_uses_same_origin_only():
    text = (ROOT / "static" / "js" / "maps.js").read_text(encoding="utf-8")
    assert not re.search(r"https?://", text)
    assert "pmtiles://" in text and "window.location.origin" in text
    assert "/fonts/{fontstack}/{range}.pbf" in text


def test_vector_mbtiles_expose_layers_bounds_and_attribution(tmp_path):
    lib = make_library(tmp_path)
    path = make_mbtiles(lib.maps_dir / "zoomstack.mbtiles", fmt="pbf", tiles={(0, 0, 0): gz(b"\x1a\x00")})
    import sqlite3
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM metadata WHERE name='attribution'")
        conn.execute("INSERT INTO metadata VALUES ('json', ?)",
                     (json.dumps({"vector_layers": [{"id": "roads"}, {"id": "water"}, {"x": 1}]}),))
        conn.execute("INSERT INTO metadata VALUES ('center', '-2.6,51.47,10')")
    cat = maps.load_catalog(lib.maps_catalog)
    (pack,) = maps.list_packs(lib.maps_dir, cat)
    assert pack["metadata"]["vector_layers"] == [{"id": "roads"}, {"id": "water"}]
    assert "json" not in pack["metadata"]
    assert pack["metadata"]["center"] == [-2.6, 51.47, 10.0]
    assert pack["bounds"] == [-8.65, 49.86, 1.77, 60.86]
    assert pack["attribution"] == cat["attribution_default"] == "© OpenStreetMap contributors"


def test_demo_pmtiles_take_catalog_title_bounds_and_attribution(tmp_path):
    lib = make_library(tmp_path)
    (lib.maps_dir / "bristol.pmtiles").write_bytes(b"PMTiles")
    (pack,) = maps.list_packs(lib.maps_dir, maps.load_catalog(lib.maps_catalog))
    assert pack["catalog_id"] == "demo-bristol"
    assert pack["bounds"] == [-2.75, 51.38, -2.45, 51.56]
    assert "OpenStreetMap" in pack["attribution"]
    assert maps.load_catalog(lib.maps_catalog)["uk_bbox"] == maps.UK_BBOX


def test_fetch_script_matches_catalog_demo_packs():
    cat = maps.load_catalog(ROOT / "config" / "maps_catalog.json")
    demo = {p["filename"]: p for p in cat["packs"] if p.get("demo")}
    for name, (bbox, maxzoom) in fetch_demo_map.DEMO_PACKS.items():
        entry = demo[f"{name}.pmtiles"]
        assert [float(v) for v in bbox.split(",")] == entry["bounds"]
        assert f"maxzoom={maxzoom}" in entry["how_to_get"]
    assert fetch_demo_map._asset_name()[0].startswith("go-pmtiles")
