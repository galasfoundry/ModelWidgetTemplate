"""Offline map packs: listing, MBTiles tiles (TMS flip, gzip), PMTiles ranges."""

from __future__ import annotations

import json

from app import create_app
from firstbook import maps
from tests.firstbook_helpers import PNG_1PX, gz, image_settings, make_library, make_mbtiles


def _client(tmp_path):
    lib = make_library(tmp_path)
    return create_app(image_settings(tmp_path), library=lib).test_client(), lib


def test_shipped_maps_catalog_is_uk_and_attributed():
    data = maps.load_catalog(maps.Path(__file__).resolve().parent.parent / "config" / "maps_catalog.json")
    assert "error" not in data
    ids = {p["id"] for p in data["packs"]}
    assert {"uk-protomaps", "uk-osm-mbtiles", "os-open-zoomstack"} <= ids
    for pack in data["packs"]:
        assert pack["filename"].endswith((".mbtiles", ".pmtiles"))
        assert pack["license"] and pack["attribution"] and pack["source_url"].startswith("https://")


def test_broken_catalog_is_empty_not_fatal(tmp_path):
    bad = tmp_path / "maps.json"
    bad.write_text("{nope", encoding="utf-8")
    assert maps.load_catalog(bad)["packs"] == []
    assert maps.load_catalog(tmp_path / "missing.json")["packs"] == []


def test_list_packs_reads_metadata(tmp_path):
    lib = make_library(tmp_path)
    make_mbtiles(lib.maps_dir / "uk-osm.mbtiles")
    (lib.maps_dir / "uk.pmtiles").write_bytes(b"PMTiles" + bytes(100))
    (lib.maps_dir / "notes.txt").write_text("ignored")
    packs = maps.list_packs(lib.maps_dir, maps.load_catalog(lib.maps_catalog))
    by_name = {p["name"]: p for p in packs}
    assert set(by_name) == {"uk-osm", "uk"}
    osm = by_name["uk-osm"]
    assert osm["kind"] == "mbtiles" and osm["format"] == "png"
    assert osm["metadata"]["maxzoom"] == 2 and osm["metadata"]["bounds"][0] == -8.65
    assert osm["catalog_id"] == "uk-osm-mbtiles"
    assert osm["tile_url"] == "/tiles/uk-osm/{z}/{x}/{y}"
    assert by_name["uk"]["file_url"] == "/maps/files/uk.pmtiles"


def test_tile_route_flips_y_and_sets_type(tmp_path):
    client, lib = _client(tmp_path)
    # XYZ (1, 0, 0) is the top-left tile; MBTiles stores it at tile_row 1.
    make_mbtiles(lib.maps_dir / "test.mbtiles", tiles={(1, 0, 0): PNG_1PX, (1, 0, 1): b"\xff\xd8other"})
    top = client.get("/tiles/test/1/0/0")
    assert top.status_code == 200 and top.data == PNG_1PX
    assert top.headers["Content-Type"] == "image/png"
    assert client.get("/tiles/test/1/0/1.png").data.startswith(b"\xff\xd8")
    assert client.get("/tiles/test/1/1/1").status_code == 204  # inside range, not stored
    assert client.get("/tiles/test/1/5/0").status_code == 204  # out of range
    assert client.get("/tiles/missing/0/0/0").status_code == 404
    assert client.get("/tiles/test/0/0/abc").status_code == 404


def test_vector_tiles_are_sent_gzipped(tmp_path):
    client, lib = _client(tmp_path)
    make_mbtiles(lib.maps_dir / "vec.mbtiles", fmt="pbf", tiles={(0, 0, 0): gz(b"\x1a\x00")})
    response = client.get("/tiles/vec/0/0/0.pbf")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/x-protobuf"
    assert response.headers["Content-Encoding"] == "gzip"


def test_pmtiles_file_supports_range(tmp_path):
    client, lib = _client(tmp_path)
    (lib.maps_dir / "uk.pmtiles").write_bytes(bytes(range(256)))
    part = client.get("/maps/files/uk.pmtiles", headers={"Range": "bytes=10-19"})
    assert part.status_code == 206 and part.data == bytes(range(10, 20))
    assert client.get("/maps/files/..%2Fsecret.pmtiles").status_code == 404
    assert client.get("/maps/files/uk.mbtiles").status_code == 404


def test_api_maps_and_page(tmp_path):
    client, lib = _client(tmp_path)
    make_mbtiles(lib.maps_dir / "uk-osm.mbtiles")
    body = client.get("/api/maps").get_json()
    assert [p["name"] for p in body["packs"]] == ["uk-osm"]
    assert body["catalog"]["packs"]
    page = client.get("/maps").get_data(as_text=True)
    assert "uk-osm" in page and "OpenStreetMap" in page
    json.dumps(body)
