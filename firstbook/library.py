"""Where The First Book keeps its content.

The data root defaults to ``library/`` in the project and can be moved with
``LIBRARY_DIR`` (for example to a USB stick or a second disk). Each section
can also be moved on its own:

    library/
      maps/      *.mbtiles and *.pmtiles map packs          (MAPS_DIR)
      videos/    *.mp4 / *.webm / *.mkv plus videos.json      (VIDEOS_DIR)
      guides/    *.md and *.txt guides                        (GUIDES_DIR)

Model weights stay where the model picker already puts them (``weights/``,
``WEIGHTS_DIR``), so nothing about the existing install path changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import paths

DEFAULT_LIBRARY_DIR = paths.PROJECT_ROOT / "library"
MAPS_CATALOG_PATH = paths.PROJECT_ROOT / "config" / "maps_catalog.json"
VIDEOS_CATALOG_PATH = paths.PROJECT_ROOT / "config" / "videos_catalog.json"


@dataclass(frozen=True)
class Library:
    root: Path
    maps_dir: Path
    videos_dir: Path
    guides_dir: Path
    weights_dir: Path
    maps_catalog: Path

    @classmethod
    def from_env(cls, root: Path | None = None, weights_dir: Path | None = None) -> "Library":
        """Resolve every folder. ``root`` beats ``LIBRARY_DIR``; section env vars beat both."""
        if root is not None:
            base = Path(root).resolve()
            sections = {"maps": base / "maps", "videos": base / "videos", "guides": base / "guides"}
        else:
            base = paths.resolve_path("LIBRARY_DIR", DEFAULT_LIBRARY_DIR)
            sections = {
                name: paths.resolve_path(f"{name.upper()}_DIR", base / name)
                for name in ("maps", "videos", "guides")
            }
        weights = Path(weights_dir) if weights_dir is not None else paths.resolve_path(
            "WEIGHTS_DIR", paths.WEIGHTS_DIR
        )
        return cls(
            root=base,
            maps_dir=sections["maps"],
            videos_dir=sections["videos"],
            guides_dir=sections["guides"],
            weights_dir=weights,
            maps_catalog=paths.resolve_path("MAPS_CATALOG_PATH", MAPS_CATALOG_PATH),
        )

    def describe(self) -> dict:
        return {
            "root": str(self.root),
            "maps": str(self.maps_dir),
            "videos": str(self.videos_dir),
            "guides": str(self.guides_dir),
            "weights": str(self.weights_dir),
        }


def safe_child(base: Path, relative: str) -> Path | None:
    """Resolve ``relative`` under ``base``; None if it escapes or does not exist."""
    if not relative or "\x00" in relative or os.path.isabs(relative):
        return None
    base = Path(base).resolve()
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None
