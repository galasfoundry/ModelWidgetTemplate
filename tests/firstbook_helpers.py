"""Shared builders for The First Book tests. Everything is generated in tmp dirs."""

from __future__ import annotations

import gzip
import sqlite3
from dataclasses import replace
from pathlib import Path

from firstbook.library import Library
from modelwidget.settings import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, Settings

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


def make_mbtiles(path: Path, fmt: str = "png", tiles: dict | None = None) -> Path:
    """Write a tiny MBTiles file. ``tiles`` maps XYZ (z, x, y) to bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = tiles if tiles is not None else {(0, 0, 0): PNG_1PX}
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE metadata (name text, value text)")
    conn.execute("CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob)")
    meta = {"name": path.stem, "format": fmt, "minzoom": "0", "maxzoom": "2",
            "bounds": "-8.65,49.86,1.77,60.86", "attribution": "© OpenStreetMap contributors"}
    conn.executemany("INSERT INTO metadata VALUES (?, ?)", meta.items())
    for (z, x, y), data in tiles.items():
        conn.execute("INSERT INTO tiles VALUES (?, ?, ?, ?)", (z, x, (2**z - 1) - y, data))
    conn.commit()
    conn.close()
    return path


def gz(data: bytes) -> bytes:
    return gzip.compress(data)


def make_library(tmp_path: Path) -> Library:
    root = tmp_path / "library"
    for name in ("maps", "videos", "guides"):
        (root / name).mkdir(parents=True)
    lib = Library.from_env(root=root, weights_dir=tmp_path / "weights")
    return replace(lib, maps_catalog=Path(__file__).resolve().parent.parent / "config" / "maps_catalog.json")


def image_settings(tmp_path: Path) -> Settings:
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
