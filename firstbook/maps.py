"""Offline map packs: list what is on disk and serve MBTiles tiles.

MBTiles is a SQLite file (https://github.com/mapbox/mbtiles-spec), so the
standard library reads it. Tiles are stored in TMS order, so the XYZ ``y``
from the browser is flipped before lookup.

PMTiles packs are listed and served whole with HTTP Range support; a
locally vendored PMTiles/MapLibre viewer reads them by byte range, so the
server never has to decode them.

Nothing here touches the network. ``config/maps_catalog.json`` only records
where a pack can be fetched by hand and its licence.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

# West, south, east, north. The viewer opens here when a pack has no bounds.
UK_BBOX = [-8.65, 49.86, 1.77, 60.86]
PACK_SUFFIXES = (".mbtiles", ".pmtiles")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

FORMAT_MIMETYPES = {
    "pbf": "application/x-protobuf",
    "mvt": "application/x-protobuf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def load_catalog(path: Path) -> dict:
    """Read config/maps_catalog.json. A missing or broken file gives an empty catalog."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"version": None, "packs": [], "uk_bbox": UK_BBOX, "error": f"{path}: {exc}"}
    packs = data.get("packs") if isinstance(data, dict) else None
    if not isinstance(packs, list):
        return {"version": None, "packs": [], "uk_bbox": UK_BBOX, "error": f"{path}: no 'packs' list"}
    return {
        "version": data.get("version"),
        "description": data.get("description", ""),
        "attribution_default": data.get("attribution_default", ""),
        "uk_bbox": data.get("uk_bbox") or UK_BBOX,
        "packs": [p for p in packs if isinstance(p, dict) and p.get("id")],
    }


def _connect(path: Path) -> sqlite3.Connection:
    # Read-only URI so a bad request can never write to a pack.
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def mbtiles_metadata(path: Path) -> dict:
    """The ``metadata`` table as a dict, plus numeric zooms and bounds when present."""
    meta: dict = {}
    try:
        with _connect(path) as conn:
            for name, value in conn.execute("SELECT name, value FROM metadata"):
                meta[str(name)] = value
    except sqlite3.Error as exc:
        return {"error": f"not a readable MBTiles file: {exc}"}
    for key in ("minzoom", "maxzoom"):
        try:
            meta[key] = int(meta[key])
        except (KeyError, TypeError, ValueError):
            meta.pop(key, None)
    for key in ("bounds", "center"):
        if isinstance(meta.get(key), str):
            try:
                meta[key] = [float(v) for v in meta[key].split(",")]
            except ValueError:
                pass
    # Vector MBTiles describe their layers in a JSON blob; the viewer uses the
    # layer ids to build a plain style when it does not know the schema.
    if isinstance(meta.get("json"), str):
        try:
            blob = json.loads(meta.pop("json"))
        except json.JSONDecodeError:
            blob = {}
        layers = blob.get("vector_layers") if isinstance(blob, dict) else None
        if isinstance(layers, list):
            meta["vector_layers"] = [
                {"id": str(layer["id"])} for layer in layers if isinstance(layer, dict) and layer.get("id")
            ]
    return meta


def list_packs(maps_dir: Path, catalog: dict | None = None) -> list[dict]:
    """Every pack file in ``maps_dir``, joined to its catalog entry by filename."""
    by_file = {}
    default_attribution = (catalog or {}).get("attribution_default", "")
    for entry in (catalog or {}).get("packs", []):
        if entry.get("filename"):
            by_file[entry["filename"]] = entry
    packs: list[dict] = []
    maps_dir = Path(maps_dir)
    if not maps_dir.is_dir():
        return packs
    for file in sorted(maps_dir.iterdir()):
        if not file.is_file() or file.suffix.lower() not in PACK_SUFFIXES:
            continue
        if not _NAME_RE.match(file.stem):
            continue
        kind = file.suffix.lower().lstrip(".")
        entry = by_file.get(file.name, {})
        pack = {
            "name": file.stem,
            "file": file.name,
            "kind": kind,
            "size_mb": round(file.stat().st_size / 1024**2, 1),
            "catalog_id": entry.get("id"),
            "title": entry.get("title", file.stem),
            "attribution": entry.get("attribution", ""),
            "license": entry.get("license", ""),
        }
        if kind == "mbtiles":
            meta = mbtiles_metadata(file)
            pack["metadata"] = meta
            pack["format"] = meta.get("format")
            pack["tile_url"] = f"/tiles/{file.stem}/{{z}}/{{x}}/{{y}}"
            if not pack["attribution"]:
                pack["attribution"] = meta.get("attribution", "")
        else:
            pack["file_url"] = f"/maps/files/{file.name}"
        if not pack["attribution"]:
            pack["attribution"] = default_attribution
        # Packs that ship whole-UK coverage open on the UK; others on their bounds.
        pack["bounds"] = (pack.get("metadata") or {}).get("bounds") or entry.get("bounds")
        packs.append(pack)
    return packs


def find_pack(maps_dir: Path, name: str, kind: str) -> Path | None:
    if not _NAME_RE.match(name or ""):
        return None
    path = Path(maps_dir) / f"{name}.{kind}"
    return path if path.is_file() else None


def read_tile(path: Path, z: int, x: int, y: int) -> tuple[bytes, dict] | None:
    """Return (data, headers) for XYZ tile z/x/y, or None when it is not in the pack."""
    if z < 0 or z > 30 or not (0 <= x < 2**z) or not (0 <= y < 2**z):
        return None
    tms_y = (2**z - 1) - y
    try:
        with _connect(path) as conn:
            row = conn.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, tms_y),
            ).fetchone()
            fmt_row = conn.execute("SELECT value FROM metadata WHERE name='format'").fetchone()
    except sqlite3.Error:
        return None
    if row is None or row[0] is None:
        return None
    data = bytes(row[0])
    fmt = (fmt_row[0] if fmt_row else "") or _sniff(data)
    headers = {"Content-Type": FORMAT_MIMETYPES.get(str(fmt).lower(), "application/octet-stream")}
    if data[:2] == b"\x1f\x8b":
        # Vector tiles in MBTiles are normally gzipped; let the browser inflate them.
        headers["Content-Encoding"] = "gzip"
    headers["Cache-Control"] = "public, max-age=86400"
    return data, headers


def _sniff(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "png"
    if data.startswith(b"\xff\xd8"):
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "pbf"
