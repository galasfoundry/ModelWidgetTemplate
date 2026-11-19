"""Local video library: index, search and stream files from the videos folder.

Metadata lives in plain JSON, never in the video files:

* ``videos.json`` in the videos folder is the manifest (see
  ``docs/ARCHITECTURE.md`` and ``scripts/build_video_manifest.py``)::

      {"version": 1,
       "videos": [{"file": "electronics/soldering-basics.mp4",
                   "title": "Soldering basics", "topic": "electronics",
                   "tags": ["soldering", "repair"], "duration_s": 754,
                   "description": "", "source": "", "license": ""}]}

* An optional sidecar ``<video>.json`` next to one file (for example
  ``soldering-basics.mp4.json``) overrides that file's manifest entry, so
  a video and its notes can be copied around together.

Files on disk with no metadata still show up, titled from the filename.
Streaming goes through Flask's ``send_file(conditional=True)``, which answers
``Range`` requests with ``206 Partial Content`` so players can seek.
"""

from __future__ import annotations

import json
from pathlib import Path

VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".m4v")
MANIFEST_NAME = "videos.json"

# "Tech-hacky": practical how-to, repair and making. A video may use any topic;
# these are the suggested ones the page groups by.
TOPICS = (
    "repair",
    "electronics",
    "power",
    "radio",
    "computing",
    "3d-printing",
    "tools",
    "making",
    "water",
    "food",
    "shelter",
    "first-aid",
    "navigation",
    "other",
)

MIMETYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_manifest(videos_dir: Path) -> dict[str, dict]:
    """Manifest entries keyed by their relative POSIX path."""
    data = _read_json(Path(videos_dir) / MANIFEST_NAME) or {}
    entries = {}
    for item in data.get("videos", []) if isinstance(data.get("videos"), list) else []:
        if isinstance(item, dict) and isinstance(item.get("file"), str):
            entries[item["file"].replace("\\", "/")] = item
    return entries


def title_from_filename(name: str) -> str:
    stem = Path(name).stem.replace("_", " ").replace("-", " ").strip()
    return stem[:1].upper() + stem[1:] if stem else name


def _normalise(rel: str, file: Path, meta: dict) -> dict:
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    duration = meta.get("duration_s")
    topic = str(meta.get("topic") or "other").strip().lower()
    return {
        "id": rel,
        "file": rel,
        "title": str(meta.get("title") or title_from_filename(file.name)),
        "topic": topic,
        "tags": [str(t).strip().lower() for t in tags if str(t).strip()],
        "duration_s": duration if isinstance(duration, (int, float)) and not isinstance(duration, bool) else None,
        "description": str(meta.get("description") or ""),
        "source": str(meta.get("source") or ""),
        "license": str(meta.get("license") or ""),
        "size_mb": round(file.stat().st_size / 1024**2, 1),
        "mimetype": MIMETYPES.get(file.suffix.lower(), "application/octet-stream"),
        "url": f"/videos/file/{rel}",
        "has_metadata": bool(meta),
    }


def scan_files(videos_dir: Path) -> list[Path]:
    root = Path(videos_dir)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and not any(
            part.startswith(".") for part in p.relative_to(root).parts
        )
    )


def index(videos_dir: Path) -> list[dict]:
    """Every video file on disk with its merged metadata."""
    root = Path(videos_dir)
    manifest = load_manifest(root)
    out = []
    for file in scan_files(root):
        rel = file.relative_to(root).as_posix()
        meta = dict(manifest.get(rel, {}))
        sidecar = _read_json(file.with_name(file.name + ".json"))
        if sidecar:
            meta.update(sidecar)
        out.append(_normalise(rel, file, meta))
    return out


def search(items: list[dict], query: str = "", topic: str = "", tag: str = "") -> list[dict]:
    """Keyword AND-search over title, description, topic and tags."""
    words = [w for w in query.lower().split() if w]
    topic = topic.strip().lower()
    tag = tag.strip().lower()
    results = []
    for item in items:
        if topic and item["topic"] != topic:
            continue
        if tag and tag not in item["tags"]:
            continue
        haystack = " ".join(
            [item["title"], item["description"], item["topic"], item["file"], *item["tags"]]
        ).lower()
        if all(word in haystack for word in words):
            results.append(item)
    return results


def topics_present(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item["topic"]] = counts.get(item["topic"], 0) + 1
    return dict(sorted(counts.items()))
