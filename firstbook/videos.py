"""Local video library: index, search and stream files from the videos folder.

Metadata lives in plain JSON, never in the video files:

* ``videos.json`` in the videos folder is the manifest (see
  ``docs/ARCHITECTURE.md`` and ``scripts/build_video_manifest.py``)::

      {"version": 1,
       "videos": [{"file": "electronics/soldering-basics.mp4",
                   "title": "Soldering basics", "topic": "electronics",
                   "tags": ["soldering", "repair"], "duration_s": 754,
                   "description": "", "source": "", "license": "",
                   "creator": "", "creator_url": "", "license_url": "",
                   "attribution": ""}]}

* An optional sidecar ``<video>.json`` next to one file (for example
  ``soldering-basics.mp4.json``) overrides that file's manifest entry, so
  a video and its notes can be copied around together.

* ``config/videos_catalog.json`` lists videos that ``scripts/fetch_videos.py``
  can download while online (Creative Commons Attribution only). The fetch
  script copies each entry's title, creator and attribution line into the
  manifest, so credits travel with the library, not with the catalog.

* First Book originals live in ``library/videos/originals/``. The ``.mp4.json``
  sidecar and ``.srt`` are in git; the ``.mp4`` is copied in at setup and is
  not committed. The watch page serves the sidecar credits and turns the
  ``.srt`` into WebVTT for a ``<track>`` element.

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
    "hydro",
    "batteries",
    "starlink-off-grid",
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

# Display names for topics the page groups by; others show their slug.
TOPIC_NAMES = {
    "hydro": "DIY off-grid hydro",
    "batteries": "Battery storage",
    "starlink-off-grid": "Starlink off-grid",
    "power": "Power",
    "fire": "Fire",
}


def branch_id(value: object) -> str:
    """Slug for a branch lane. Blank stays blank."""
    return str(value or "").strip().lower()


def parent_episode(value: object) -> str | None:
    """Episode number this item builds on, or None when it has no parent."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text

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
        "license_url": str(meta.get("license_url") or ""),
        "creator": str(meta.get("creator") or ""),
        "creator_url": str(meta.get("creator_url") or ""),
        "attribution": str(meta.get("attribution") or ""),
        "fact_check": str(meta.get("fact_check") or ""),
        "size_mb": round(file.stat().st_size / 1024**2, 1),
        "mimetype": MIMETYPES.get(file.suffix.lower(), "application/octet-stream"),
        "url": f"/videos/file/{rel}",
        "has_metadata": bool(meta),
        "subtitles": "",
        "subtitles_url": "",
        "episode": str(meta.get("episode") or "").strip(),
        "branch": branch_id(meta.get("branch")),
        "parent": parent_episode(meta.get("parent", meta.get("parent_episode"))),
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


def find_subtitles(videos_dir: Path, file: Path, meta: dict) -> Path | None:
    """A .srt or .vtt next to the video, or the path in the sidecar ``subtitles`` field."""
    root = Path(videos_dir).resolve()
    candidates: list[Path] = []
    named = meta.get("subtitles")
    if isinstance(named, str) and named.strip():
        candidates.append((file.parent / named.strip()).resolve())
    candidates.append(file.with_suffix(".vtt").resolve())
    candidates.append(file.with_suffix(".srt").resolve())
    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            cand.relative_to(root)
        except ValueError:
            continue
        if cand.is_file() and cand.suffix.lower() in {".srt", ".vtt"}:
            return cand
    return None


def srt_to_vtt(text: str) -> str:
    """Turn SubRip captions into WebVTT for the HTML5 <track> element."""
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if "-->" in line:
            line = line.replace(",", ".")
        lines.append(line)
    body = "\n".join(lines).lstrip("\n")
    if not body.startswith("WEBVTT"):
        body = "WEBVTT\n\n" + body
    return body if body.endswith("\n") else body + "\n"


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
        item = _normalise(rel, file, meta)
        sub = find_subtitles(root, file, meta)
        if sub:
            item["subtitles"] = sub.resolve().relative_to(root.resolve()).as_posix()
            item["subtitles_url"] = f"/videos/subtitles/{item['subtitles']}"
        out.append(item)
    return out


def search(items: list[dict], query: str = "", topic: str = "", tag: str = "") -> list[dict]:
    """Keyword AND-search over title, description, topic, creator and tags."""
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
            [item["title"], item["description"], item["topic"], item["file"], item.get("creator", ""), *item["tags"]]
        ).lower()
        if all(word in haystack for word in words):
            results.append(item)
    return results


CATALOG_FIELDS = (
    "title", "topic", "tags", "duration_s", "description", "creator", "creator_url",
    "license", "license_url", "attribution",
)


def load_catalog(path: Path) -> dict:
    """Read config/videos_catalog.json. A missing or broken file gives an empty catalog.

    Entries without an id, url, file or attribution are skipped, and so is
    any whose licence is not CC BY: the catalog is only for videos that may be
    kept offline with credit.
    """
    data = _read_json(Path(path)) or {}
    out = []
    for item in data.get("videos", []) if isinstance(data.get("videos"), list) else []:
        if not isinstance(item, dict):
            continue
        if not all(isinstance(item.get(k), str) and item[k].strip() for k in ("id", "url", "file", "attribution")):
            continue
        if not is_cc_by(item.get("license", "")):
            continue
        file = item["file"].replace("\\", "/")
        if file.startswith("/") or ".." in file.split("/") or Path(file).suffix.lower() not in VIDEO_SUFFIXES:
            continue
        out.append({**item, "file": file, "topic": str(item.get("topic") or "other").strip().lower()})
    topics = data.get("topics") if isinstance(data.get("topics"), dict) else {}
    return {"version": data.get("version", 1), "topics": topics, "videos": out}


def is_cc_by(license_text: str) -> bool:
    """True for CC BY wording (catalog "CC BY 3.0" or YouTube's
    "Creative Commons Attribution license (reuse allowed)"), not for -NC/-ND/-SA."""
    text = str(license_text or "").lower().replace("-", " ")
    if not ("cc by" in text or "creative commons attribution" in text):
        return False
    return not any(word in text for word in ("noncommercial", "non commercial", "noderiv", "no deriv",
                                              "sharealike", "share alike", "by nc", "by nd", "by sa"))


def manifest_entry(item: dict) -> dict:
    """The videos.json entry for one catalog item (``file`` plus its metadata)."""
    entry = {"file": item["file"], "source": item["url"]}
    for key in CATALOG_FIELDS:
        if key in item:
            entry[key] = item[key]
    return entry


def topics_present(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item["topic"]] = counts.get(item["topic"], 0) + 1
    return dict(sorted(counts.items()))
