"""Remote, downloadable video library for The First Book.

Playback stays offline: the watch page only reads files already in the
videos folder. An explicit "Download library" action may go online to
fetch a versioned manifest and then selected files, each streamed to a
``.part`` file, checked with sha256, and renamed into place atomically.

The library base URL is ``FIRSTBOOK_LIBRARY_URL`` (and the same key in
``config/.env``). The default is the published GitHub Release
``episodes-v1`` under galasfoundry (the four First Book originals, with
branch and parent fields). ``library-v1`` was the earlier release of
episodes 001 and 002. ``FIRSTBOOK_LIBRARY_MANIFEST`` overrides the URL
with a local file or a direct manifest URL, which is how tests and a
USB-key copy work. Publishing a new episode is: build the manifest,
upload the files, no code change. ``scripts/fetch_videos.py`` (yt-dlp)
stays as a fallback for the third-party Creative Commons catalog.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

from firstbook import videos
from firstbook.library import VIDEOS_CATALOG_PATH

SCHEMA = "firstbook.library.v1"
DEFAULT_LIBRARY_URL = (
    "https://github.com/galasfoundry/ModelWidgetTemplate/releases/download/episodes-v1/"
)
# An old or mistaken setting may still point here. It is not a live host.
PLACEHOLDER_LIBRARY_URL = (
    "https://github.com/galasfoundry/ModelWidgetTemplate/releases/download/library-placeholder/"
)
MANIFEST_FILENAME = "manifest.json"
STATE_NAME = ".library-state.json"
CACHED_MANIFEST_NAME = ".library-manifest.json"
PARTIAL_DIR = ".downloading"
CHUNK = 64 * 1024
USER_AGENT = "TheFirstBook/1.0 (+https://github.com/galasfoundry/ModelWidgetTemplate)"
MANIFEST_MAX_BYTES = 8 * 1024 * 1024
ENV_URL = "FIRSTBOOK_LIBRARY_URL"
ENV_MANIFEST = "FIRSTBOOK_LIBRARY_MANIFEST"

METADATA_FIELDS = (
    "title",
    "topic",
    "tags",
    "duration_s",
    "description",
    "source",
    "source_url",
    "license",
    "license_url",
    "creator",
    "creator_url",
    "attribution",
    "credit",
    "fact_check",
    "subtitles",
    "episode",
    "branch",
    "parent",
    "narration_voice",
)


class ManifestError(ValueError):
    """The manifest is missing, the wrong schema, or has an unsafe entry."""


class DownloadError(RuntimeError):
    """A file could not be fetched, verified, or written."""


class HashMismatch(DownloadError):
    """The downloaded bytes did not match the manifest sha256."""


@dataclass(frozen=True)
class LibrarySettings:
    base_url: str
    manifest_spec: str
    is_placeholder: bool

    @property
    def manifest_url(self) -> str:
        if self.manifest_spec:
            return self.manifest_spec
        return manifest_url_from_base(self.base_url)

    @property
    def display_url(self) -> str:
        return self.manifest_spec or self.base_url or DEFAULT_LIBRARY_URL


def library_settings() -> LibrarySettings:
    """Read the library URL from the environment (and so from config/.env)."""
    base = os.environ.get(ENV_URL, "").strip() or DEFAULT_LIBRARY_URL
    if not base.endswith("/") and not base.lower().endswith(".json"):
        base += "/"
    manifest = os.environ.get(ENV_MANIFEST, "").strip()
    placeholder = is_placeholder_url(manifest or base)
    return LibrarySettings(base_url=base, manifest_spec=manifest, is_placeholder=placeholder)


def is_placeholder_url(url: str) -> bool:
    text = (url or "").lower()
    return "library-placeholder" in text or "/placeholder/" in text


def manifest_url_from_base(base: str) -> str:
    if base.lower().endswith(".json"):
        return base
    return urljoin(base if base.endswith("/") else base + "/", MANIFEST_FILENAME)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def size_label(n: int | float | None) -> str:
    if n is None:
        return ""
    n = float(n)
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{int(n)} B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def state_path(videos_dir: Path) -> Path:
    return Path(videos_dir) / STATE_NAME


def cached_manifest_path(videos_dir: Path) -> Path:
    return Path(videos_dir) / CACHED_MANIFEST_NAME


def load_state(videos_dir: Path) -> dict:
    data = _read_json(state_path(videos_dir)) or {}
    return {
        "last_checked": data.get("last_checked"),
        "last_checked_ok": bool(data.get("last_checked_ok")),
        "last_success": data.get("last_success"),
        "manifest_url": str(data.get("manifest_url") or ""),
        "schema": str(data.get("schema") or ""),
        "error": data.get("error"),
        "video_count": data.get("video_count"),
    }


def save_state(videos_dir: Path, **fields: object) -> dict:
    path = state_path(videos_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _read_json(path) or {}
    current.update(fields)
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return load_state(videos_dir)


def _safe_relative(raw: str, *, suffixes: tuple[str, ...] | None = None) -> str | None:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        return None
    rel = raw.replace("\\", "/").strip()
    if rel.startswith("/") or rel.startswith("~"):
        return None
    parts = [p for p in rel.split("/") if p and p != "."]
    if not parts or any(p == ".." or p.startswith(".") for p in parts):
        return None
    rel = "/".join(parts)
    if suffixes is not None and Path(rel).suffix.lower() not in suffixes:
        return None
    return rel


def _hex_sha(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if len(text) == 64 and all(c in "0123456789abcdef" for c in text):
        return text
    return None


def _int_size(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def _absolute_url(value: str, base: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return value
    if parsed.scheme:
        return ""
    if not base:
        return ""
    return urljoin(base if base.endswith("/") else base + "/", value.lstrip("/"))


def _entry_from_raw(raw: dict, base: str) -> dict:
    rel = _safe_relative(str(raw.get("file") or ""), suffixes=videos.VIDEO_SUFFIXES)
    ident = str(raw.get("id") or (Path(rel).stem if rel else "")).strip()
    sha = _hex_sha(raw.get("sha256"))
    size = _int_size(raw.get("size") if raw.get("size") is not None else raw.get("bytes"))
    file_url = raw.get("url") or raw.get("file_url") or raw.get("href")
    file_url = _absolute_url(str(file_url or ""), base)
    if not rel or not ident or not sha or size is None or not file_url:
        raise ManifestError(
            f"Video entry {ident or rel or '(unknown)'} needs id, a safe file path, "
            "an http(s) file URL, size and sha256."
        )
    sidecar = raw.get("sidecar") if isinstance(raw.get("sidecar"), dict) else {}
    license_text = str(raw.get("license") or raw.get("licence") or sidecar.get("license") or sidecar.get("licence") or "")
    credit = str(raw.get("credit") or raw.get("attribution") or sidecar.get("credit") or sidecar.get("attribution") or "")
    source = str(raw.get("source_url") or raw.get("source") or sidecar.get("source_url") or sidecar.get("source") or "")
    duration = raw.get("duration_s") if raw.get("duration_s") is not None else raw.get("duration")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        duration = sidecar.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        duration = None
    subtitles_rel = raw.get("subtitles") or sidecar.get("subtitles") or ""
    subtitles_rel = _safe_relative(str(subtitles_rel), suffixes=(".srt", ".vtt")) if subtitles_rel else ""
    if not subtitles_rel:
        # Default: a caption file sitting next to the video, if the manifest gives a URL.
        pass
    subtitles_url = _absolute_url(str(raw.get("subtitles_url") or sidecar.get("subtitles_url") or ""), base)
    if subtitles_url and not subtitles_rel:
        name = posixpath.basename(urlparse(subtitles_url).path)
        guess = _safe_relative(str(Path(rel).parent / name) if name else "", suffixes=(".srt", ".vtt"))
        subtitles_rel = guess or ""
    title = str(raw.get("title") or sidecar.get("title") or videos.title_from_filename(Path(rel).name))
    topic = str(raw.get("topic") or sidecar.get("topic") or "other").strip().lower() or "other"
    tags = raw.get("tags") if isinstance(raw.get("tags"), list) else sidecar.get("tags")
    tags = [str(t).strip().lower() for t in (tags or []) if str(t).strip()]
    meta = {k: sidecar[k] for k in sidecar if k not in {"url", "file_url", "sha256", "size"}}
    for key in METADATA_FIELDS:
        if key in raw and key not in ("sidecar",):
            meta.setdefault(key, raw[key])
    meta.update({
        "file": rel,
        "title": title,
        "topic": topic,
        "tags": tags,
        "duration_s": duration,
        "license": license_text,
        "license_url": str(raw.get("license_url") or raw.get("licence_url") or meta.get("license_url") or ""),
        "credit": credit,
        "attribution": credit or str(meta.get("attribution") or ""),
        "source": source,
        "source_url": source,
        "creator": str(raw.get("creator") or meta.get("creator") or ""),
        "creator_url": str(raw.get("creator_url") or meta.get("creator_url") or ""),
        "description": str(raw.get("description") or meta.get("description") or ""),
        "subtitles": subtitles_rel,
    })
    if "branch" in raw:
        branch = videos.branch_id(raw.get("branch"))
    else:
        branch = videos.branch_id(meta.get("branch"))
    if "parent" in raw:
        parent = videos.parent_episode(raw.get("parent"))
    elif "parent_episode" in raw:
        parent = videos.parent_episode(raw.get("parent_episode"))
    else:
        parent = videos.parent_episode(meta.get("parent", meta.get("parent_episode")))
    episode = raw.get("episode") if raw.get("episode") not in (None, "") else meta.get("episode")
    episode = str(episode or "").strip()
    meta["branch"] = branch
    meta["parent"] = parent
    meta["episode"] = episode
    return {
        "id": ident,
        "file": rel,
        "title": title,
        "topic": topic,
        "license": license_text,
        "license_url": meta["license_url"],
        "credit": credit,
        "attribution": meta["attribution"],
        "source": source,
        "source_url": source,
        "url": file_url,
        "size": size,
        "size_label": size_label(size),
        "sha256": sha,
        "duration_s": duration,
        "subtitles": subtitles_rel,
        "subtitles_url": subtitles_url,
        "tags": tags,
        "description": meta["description"],
        "creator": meta["creator"],
        "creator_url": meta["creator_url"],
        "branch": branch,
        "parent": parent,
        "episode": episode,
        "sidecar": meta,
    }


def parse_manifest(data: object, *, base_url: str = "") -> dict:
    """Validate a remote library manifest and return a normalised copy."""
    if not isinstance(data, dict):
        raise ManifestError("The library manifest must be a JSON object.")
    schema = str(data.get("schema") or data.get("schema_version") or "").strip()
    if schema != SCHEMA:
        raise ManifestError(
            f"Unsupported library manifest schema {schema or '(missing)'!r}. "
            f"Expected {SCHEMA}."
        )
    raw_videos = data.get("videos")
    if not isinstance(raw_videos, list):
        raise ManifestError("The library manifest needs a 'videos' list.")
    declared_base = str(data.get("base_url") or data.get("base") or "").strip() or base_url
    items = []
    seen: set[str] = set()
    for raw in raw_videos:
        if not isinstance(raw, dict):
            raise ManifestError("Each video entry must be a JSON object.")
        entry = _entry_from_raw(raw, declared_base)
        if entry["id"] in seen or entry["file"] in seen:
            raise ManifestError(f"Duplicate video id or file in the manifest: {entry['id']}.")
        seen.add(entry["id"])
        seen.add(entry["file"])
        items.append(entry)
    return {
        "schema": SCHEMA,
        "version": data.get("version", 1),
        "generated_at": data.get("generated_at") or "",
        "base_url": declared_base,
        "videos": items,
    }


def load_manifest_file(path: Path, *, base_url: str = "") -> dict:
    data = _read_json(path)
    if data is None:
        raise ManifestError(f"Could not read a library manifest from {path}.")
    return parse_manifest(data, base_url=base_url)


def open_remote(url: str, *, range_header: str | None = None, timeout: float = 30.0):
    """GET a library file, following one redirect and keeping a Range header.

    GitHub release URLs answer 302 to a short-lived file host. The player
    needs that host to honour Range, because the episode mp4 stores its
    index at the end of the file.
    """

    class _StopRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: A003
            return None

    opener = urllib.request.build_opener(_StopRedirect)

    def once(target: str):
        headers = {"User-Agent": USER_AGENT}
        if range_header:
            headers["Range"] = range_header
        request = urllib.request.Request(target, headers=headers)
        return opener.open(request, timeout=timeout)

    try:
        return once(url)
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            location = exc.headers.get("Location")
            exc.close()
            if not location:
                raise DownloadError(f"Redirect from the library file had no Location.")
            try:
                return once(urljoin(url, location))
            except urllib.error.HTTPError as exc2:
                raise DownloadError(f"Could not open the library file ({exc2.code}).") from exc2
            except urllib.error.URLError as exc2:
                raise DownloadError("Could not open the library file: offline or blocked.") from exc2
        raise DownloadError(f"Could not open the library file ({exc.code}).") from exc
    except urllib.error.URLError as exc:
        raise DownloadError("Could not open the library file: offline or blocked.") from exc


def _open(url: str, headers: dict | None = None, timeout: float = 30.0):
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(url, headers=req_headers)
    return urllib.request.urlopen(request, timeout=timeout)


def _is_local_path(spec: str) -> bool:
    if not spec:
        return False
    parsed = urlparse(spec)
    if parsed.scheme in {"http", "https"}:
        return False
    if parsed.scheme == "file":
        return True
    return True


def fetch_manifest(spec: str | None = None, *, settings: LibrarySettings | None = None) -> dict:
    """Load and parse the remote (or local) library manifest."""
    settings = settings or library_settings()
    spec = (spec or settings.manifest_url).strip()
    if not spec:
        raise ManifestError("No library manifest URL or path is configured.")
    if _is_local_path(spec):
        path = Path(urlparse(spec).path if spec.startswith("file:") else spec)
        if not path.is_absolute():
            from paths import PROJECT_ROOT

            path = (PROJECT_ROOT / path).resolve()
        return load_manifest_file(path, base_url=settings.base_url)
    try:
        with _open(spec) as resp:
            raw = resp.read(MANIFEST_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        hint = ""
        if settings.is_placeholder or is_placeholder_url(spec):
            hint = (
                " This FIRSTBOOK_LIBRARY_URL is a placeholder, not a live host. "
                f"Use the published library at {DEFAULT_LIBRARY_URL} "
                "or set FIRSTBOOK_LIBRARY_MANIFEST to a local manifest file."
            )
        raise ManifestError(f"Could not fetch the library manifest ({exc.code}).{hint}") from exc
    except urllib.error.URLError as exc:
        hint = ""
        if settings.is_placeholder or is_placeholder_url(spec):
            hint = (
                " This FIRSTBOOK_LIBRARY_URL is a placeholder, not a live host. "
                f"The published library is {DEFAULT_LIBRARY_URL}."
            )
        raise ManifestError(f"Could not reach the library manifest (offline or blocked).{hint}") from exc
    if len(raw) > MANIFEST_MAX_BYTES:
        raise ManifestError("The library manifest is larger than 8 MB; it looks wrong.")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("The library manifest is not valid JSON.") from exc
    base = settings.base_url
    if not settings.manifest_spec and not settings.base_url.lower().endswith(".json"):
        base = settings.base_url
    elif spec.lower().endswith(".json"):
        base = spec.rsplit("/", 1)[0] + "/"
    return parse_manifest(data, base_url=base)


def file_status(entry: dict, videos_dir: Path) -> str:
    """'new', 'updated' or 'current' for one manifest entry against the disk."""
    dest = Path(videos_dir) / entry["file"]
    if not dest.is_file():
        return "new"
    try:
        if dest.stat().st_size != entry["size"]:
            return "updated"
        if sha256_file(dest) != entry["sha256"]:
            return "updated"
    except OSError:
        return "updated"
    return "current"


def compare_manifest(manifest: dict, videos_dir: Path) -> dict:
    items = []
    counts = {"new": 0, "updated": 0, "current": 0}
    total_new = 0
    for entry in manifest["videos"]:
        status = file_status(entry, videos_dir)
        counts[status] += 1
        if status != "current":
            total_new += entry["size"]
        items.append({
            "id": entry["id"],
            "file": entry["file"],
            "title": entry["title"],
            "topic": entry["topic"],
            "license": entry["license"],
            "credit": entry["credit"],
            "size": entry["size"],
            "size_label": entry["size_label"],
            "sha256": entry["sha256"],
            "duration_s": entry["duration_s"],
            "status": status,
        })
    return {
        "schema": SCHEMA,
        "generated_at": manifest.get("generated_at") or "",
        "videos": items,
        "counts": counts,
        "download_bytes": total_new,
        "download_label": size_label(total_new) if total_new else "0 B",
        "total": len(items),
    }


def part_path(videos_dir: Path, rel: str) -> Path:
    return Path(videos_dir) / PARTIAL_DIR / f"{rel}.part"


def _write_sidecar(videos_dir: Path, entry: dict) -> None:
    dest = Path(videos_dir) / entry["file"]
    meta = dict(entry.get("sidecar") or {})
    meta.setdefault("file", entry["file"])
    meta.setdefault("title", entry["title"])
    meta.setdefault("topic", entry["topic"])
    meta.setdefault("license", entry.get("license") or "")
    meta.setdefault("license_url", entry.get("license_url") or "")
    meta.setdefault("attribution", entry.get("attribution") or entry.get("credit") or "")
    meta.setdefault("credit", entry.get("credit") or meta.get("attribution") or "")
    meta.setdefault("source", entry.get("source") or "")
    meta.setdefault("creator", entry.get("creator") or "")
    meta.setdefault("creator_url", entry.get("creator_url") or "")
    meta.setdefault("description", entry.get("description") or "")
    meta.setdefault("duration_s", entry.get("duration_s"))
    meta.setdefault("tags", entry.get("tags") or [])
    if entry.get("subtitles"):
        meta.setdefault("subtitles", Path(entry["subtitles"]).name if "/" not in str(entry["subtitles"]) or Path(entry["subtitles"]).parent == Path(entry["file"]).parent else entry["subtitles"])
        # Keep a path relative to the video file when they share a folder.
        sub_rel = entry["subtitles"]
        try:
            meta["subtitles"] = Path(sub_rel).relative_to(Path(entry["file"]).parent).as_posix()
        except ValueError:
            meta["subtitles"] = Path(sub_rel).name
    sidecar_path = dest.with_name(dest.name + ".json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _merge_local_manifest(videos_dir: Path, entry: dict) -> None:
    path = Path(videos_dir) / videos.MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    records = [e for e in data.get("videos", []) if isinstance(e, dict) and isinstance(e.get("file"), str)]
    records = [e for e in records if e["file"] != entry["file"]]
    local = {
        "file": entry["file"],
        "title": entry["title"],
        "topic": entry["topic"],
        "tags": entry.get("tags") or [],
        "duration_s": entry.get("duration_s"),
        "description": entry.get("description") or "",
        "source": entry.get("source") or "",
        "license": entry.get("license") or "",
        "license_url": entry.get("license_url") or "",
        "creator": entry.get("creator") or "",
        "creator_url": entry.get("creator_url") or "",
        "attribution": entry.get("attribution") or entry.get("credit") or "",
    }
    if entry.get("subtitles"):
        local["subtitles"] = entry["subtitles"]
    if "branch" in entry:
        local["branch"] = entry.get("branch") or ""
    if "episode" in entry:
        local["episode"] = entry.get("episode") or ""
    if "parent" in entry:
        local["parent"] = entry.get("parent")
    records.append(local)
    path.write_text(json.dumps({"version": 1, "videos": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download_file(
    url: str,
    dest: Path,
    *,
    sha256: str,
    size: int,
    part: Path | None = None,
    progress: Callable[..., None] | None = None,
    timeout: float = 30.0,
) -> Path:
    """Stream ``url`` to ``dest`` via a ``.part`` file, with resume and sha256.

    The file is hashed as it arrives. A mismatch, or a final size that is
    wrong, deletes the partial and leaves any existing ``dest`` untouched.
    """
    dest = Path(dest)
    part = Path(part) if part is not None else Path(str(dest) + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256.lower().strip()

    def report(done: int) -> None:
        if progress is not None:
            progress(done, size)

    start = 0
    digest = hashlib.sha256()
    if part.is_file():
        existing = part.stat().st_size
        if existing > size:
            part.unlink(missing_ok=True)
        elif existing == size:
            got = sha256_file(part)
            if got == expected:
                report(size)
                os.replace(part, dest)
                return dest
            part.unlink(missing_ok=True)
        elif existing > 0:
            start = existing
            with part.open("rb") as fh:
                while True:
                    chunk = fh.read(CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)

    headers = {}
    if start:
        headers["Range"] = f"bytes={start}-"
    try:
        resp = _open(url, headers=headers, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if start and exc.code in {416, 200, 400, 404, 501}:
            part.unlink(missing_ok=True)
            start = 0
            digest = hashlib.sha256()
            try:
                resp = _open(url, timeout=timeout)
            except urllib.error.HTTPError as exc2:
                raise DownloadError(f"Could not download {url} ({exc2.code}).") from exc2
        else:
            raise DownloadError(f"Could not download {url} ({exc.code}).") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"Could not download {url}: offline or blocked.") from exc

    with resp:
        status = getattr(resp, "status", None) or resp.getcode()
        if start and status != 206:
            # Server ignored the range request; start the file again.
            start = 0
            digest = hashlib.sha256()
            mode = "wb"
        else:
            mode = "ab" if start else "wb"
        report(start)
        try:
            with part.open(mode) as fh:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    fh.flush()
                    digest.update(chunk)
                    report(part.stat().st_size)
        except OSError as exc:
            raise DownloadError(f"Could not write {part}: {exc}") from exc

    final_size = part.stat().st_size
    got = digest.hexdigest()
    if got != expected or final_size != size:
        part.unlink(missing_ok=True)
        if got != expected:
            raise HashMismatch(
                f"sha256 mismatch for {dest.name}: got {got}, expected {expected}."
            )
        raise DownloadError(
            f"Size mismatch for {dest.name}: got {final_size} bytes, expected {size}."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, dest)
    report(size)
    return dest


def _download_sidecar_file(url: str, dest: Path, part: Path, timeout: float = 30.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _open(url, timeout=timeout) as resp:
            data = resp.read(MANIFEST_MAX_BYTES + 1)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise DownloadError(f"Could not download subtitles from {url}: {exc}") from exc
    if len(data) > MANIFEST_MAX_BYTES:
        raise DownloadError(f"Subtitles file at {url} is unexpectedly large.")
    part.write_bytes(data)
    os.replace(part, dest)


def download_entry(
    entry: dict,
    videos_dir: Path,
    *,
    progress: Callable[..., None] | None = None,
    timeout: float = 30.0,
) -> Path:
    """Download one manifest entry (video, optional subtitles, sidecar + local index)."""
    videos_dir = Path(videos_dir)
    rel = entry["file"]
    dest = videos_dir / rel
    if dest.is_file() and dest.stat().st_size == entry["size"] and sha256_file(dest) == entry["sha256"]:
        if progress is not None:
            progress(entry["size"], entry["size"])
        _write_sidecar(videos_dir, entry)
        _merge_local_manifest(videos_dir, entry)
        return dest
    download_file(
        entry["url"],
        dest,
        sha256=entry["sha256"],
        size=entry["size"],
        part=part_path(videos_dir, rel),
        progress=progress,
        timeout=timeout,
    )
    if entry.get("subtitles_url") and entry.get("subtitles"):
        sub_rel = entry["subtitles"]
        # If the manifest only gave a filename, keep it next to the video.
        if "/" not in sub_rel:
            sub_rel = (Path(rel).parent / sub_rel).as_posix()
            entry = {**entry, "subtitles": Path(sub_rel).name}
        sub_dest = videos_dir / sub_rel
        _download_sidecar_file(
            entry["subtitles_url"],
            sub_dest,
            part_path(videos_dir, sub_rel),
            timeout=timeout,
        )
        entry = {**entry, "subtitles": sub_rel}
    _write_sidecar(videos_dir, entry)
    _merge_local_manifest(videos_dir, entry)
    return dest


def check_library(videos_dir: Path, *, settings: LibrarySettings | None = None, spec: str | None = None) -> dict:
    """Fetch the manifest, compare it with the disk, and remember the check time."""
    settings = settings or library_settings()
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest = fetch_manifest(spec, settings=settings)
    except ManifestError as exc:
        prev = load_state(videos_dir)
        last_success = prev.get("last_success") or (
            prev.get("last_checked") if prev.get("last_checked_ok") else None
        )
        save_state(
            videos_dir,
            last_checked=utc_now(),
            last_checked_ok=False,
            last_success=last_success,
            error=str(exc),
            manifest_url=settings.manifest_url,
        )
        state = load_state(videos_dir)
        return {
            "ok": False,
            "error": str(exc),
            "offline": True,
            "placeholder": settings.is_placeholder,
            "display_url": settings.display_url,
            "state": state,
            "comparison": None,
        }
    cache = cached_manifest_path(videos_dir)
    cache.write_text(json.dumps({"schema": manifest["schema"], "version": manifest.get("version", 1),
                                 "generated_at": manifest.get("generated_at") or "",
                                 "base_url": manifest.get("base_url") or "",
                                 "videos": manifest["videos"]}, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
    comparison = compare_manifest(manifest, videos_dir)
    now = utc_now()
    state = save_state(
        videos_dir,
        last_checked=now,
        last_checked_ok=True,
        last_success=now,
        error=None,
        manifest_url=settings.manifest_url,
        schema=SCHEMA,
        video_count=comparison["total"],
    )
    return {
        "ok": True,
        "error": None,
        "offline": False,
        "placeholder": settings.is_placeholder,
        "display_url": settings.display_url,
        "state": state,
        "comparison": comparison,
        "manifest": manifest,
    }


SHIPPED_MANIFEST_PATH = VIDEOS_CATALOG_PATH.parent / "library_manifest.json"


def _norm_base(url: str) -> str:
    return (url or "").strip().rstrip("/")


def load_shipped_manifest() -> dict | None:
    """The manifest committed in the repo, if it parses."""
    path = SHIPPED_MANIFEST_PATH
    if not path.is_file():
        return None
    try:
        return load_manifest_file(path)
    except ManifestError:
        return None


def active_manifest(videos_dir: Path, settings: LibrarySettings | None = None) -> dict | None:
    """Manifest for the video page. Does not use the network.

    A cache from the last successful check wins when it belongs to the
    configured host. Otherwise the shipped manifest is used when its base
    URL is that host, so a fresh install can list the episodes before the
    first check. A different ``FIRSTBOOK_LIBRARY_URL`` does not pick up
    the shipped catalog.
    """
    settings = settings or library_settings()
    cached = load_cached_manifest(videos_dir)
    shipped = load_shipped_manifest()
    configured = _norm_base(settings.base_url)
    if cached is not None and _norm_base(str(cached.get("base_url") or "")) == configured:
        return cached
    if shipped is not None and _norm_base(str(shipped.get("base_url") or "")) == configured:
        return shipped
    return cached


def load_cached_manifest(videos_dir: Path) -> dict | None:
    data = _read_json(cached_manifest_path(videos_dir))
    if not data:
        return None
    try:
        return parse_manifest(data, base_url=str(data.get("base_url") or ""))
    except ManifestError:
        return None


def _sidecar_for_build(file: Path, videos_dir: Path, catalog_by_file: dict[str, dict]) -> dict:
    rel = file.relative_to(videos_dir).as_posix()
    meta: dict = {}
    existing = videos.load_manifest(videos_dir).get(rel, {})
    if existing:
        meta.update(existing)
    sidecar = videos._read_json(file.with_name(file.name + ".json")) or {}
    if sidecar:
        meta.update(sidecar)
    catalog_item = catalog_by_file.get(rel)
    if catalog_item:
        for key in videos.CATALOG_FIELDS:
            if key in catalog_item and key not in meta:
                meta[key] = catalog_item[key]
        meta.setdefault("source", catalog_item.get("url") or "")
        meta.setdefault("id", catalog_item.get("id"))
    return meta


def build_remote_manifest(
    videos_dir: Path,
    *,
    base_url: str = DEFAULT_LIBRARY_URL,
    url_mode: str = "path",
    catalog_path: Path | None = None,
) -> dict:
    """Build a versioned remote manifest from local files, sidecars and the catalog."""
    videos_dir = Path(videos_dir)
    if not base_url.endswith("/"):
        base_url += "/"
    url_mode = (url_mode or "path").strip().lower()
    if url_mode not in {"path", "flat"}:
        raise ValueError("url_mode must be 'path' or 'flat'")
    catalog = videos.load_catalog(catalog_path or VIDEOS_CATALOG_PATH)
    catalog_by_file = {item["file"]: item for item in catalog["videos"]}
    entries = []
    for file in videos.scan_files(videos_dir):
        rel = file.relative_to(videos_dir).as_posix()
        meta = _sidecar_for_build(file, videos_dir, catalog_by_file)
        ident = str(meta.get("id") or meta.get("episode") or Path(rel).stem)
        remote_name = Path(rel).name if url_mode == "flat" else rel
        file_url = urljoin(base_url, remote_name)
        sub = videos.find_subtitles(videos_dir, file, meta)
        sub_rel = ""
        sub_url = ""
        if sub:
            sub_rel = sub.resolve().relative_to(videos_dir.resolve()).as_posix()
            sub_name = Path(sub_rel).name if url_mode == "flat" else sub_rel
            sub_url = urljoin(base_url, sub_name)
        credit = str(meta.get("credit") or meta.get("attribution") or "")
        source = str(meta.get("source_url") or meta.get("source") or "")
        branch = videos.branch_id(meta.get("branch"))
        parent = videos.parent_episode(meta.get("parent", meta.get("parent_episode")))
        episode = str(meta.get("episode") or "").strip()
        duration = meta.get("duration_s")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            duration = None
        sidecar = {
            k: meta[k]
            for k in meta
            if k not in {"file", "id", "url", "file_url", "sha256", "size"}
        }
        entry = {
            "id": ident,
            "file": rel,
            "title": str(meta.get("title") or videos.title_from_filename(file.name)),
            "topic": str(meta.get("topic") or "other").strip().lower() or "other",
            "license": str(meta.get("license") or meta.get("licence") or ""),
            "license_url": str(meta.get("license_url") or ""),
            "credit": credit,
            "source_url": source,
            "url": file_url,
            "size": file.stat().st_size,
            "sha256": sha256_file(file),
            "duration_s": duration,
            "subtitles_url": sub_url,
            "subtitles": sub_rel,
            "branch": branch,
            "parent": parent,
            "episode": episode,
            "sidecar": sidecar,
        }
        entries.append(entry)
    return {
        "schema": SCHEMA,
        "version": 1,
        "generated_at": utc_now(),
        "base_url": base_url,
        "videos": entries,
    }


# ---------------------------------------------------------------------------
# Background download of one batch (one worker, one job at a time)
# ---------------------------------------------------------------------------


@dataclass
class Job:
    state: str = "idle"  # idle | checking | downloading | done | error
    ids: list[str] = field(default_factory=list)
    current_id: str | None = None
    current_title: str = ""
    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0
    bytes_total: int = 0
    message: str = ""
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        total = self.bytes_total
        data["percent"] = round(100.0 * self.bytes_done / total, 1) if total else None
        data["mb_done"] = round(self.bytes_done / 1024**2, 1)
        data["mb_total"] = None if not total else round(total / 1024**2, 1)
        data["bytes_label"] = f"{size_label(self.bytes_done)} / {size_label(total)}" if total else size_label(self.bytes_done)
        end = self.finished_at or time.time()
        data["seconds"] = None if self.started_at is None else round(end - self.started_at, 1)
        return data


class LibraryDownloads:
    """One background worker that downloads a batch of library videos."""

    def __init__(self, videos_dir: Path):
        self.videos_dir = Path(videos_dir)
        self._lock = threading.Lock()
        self._job = Job()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    def snapshot(self) -> dict:
        with self._lock:
            return self._job.to_dict()

    def busy(self) -> bool:
        with self._lock:
            return self._job.state in {"checking", "downloading"}

    def start(self, manifest: dict, ids: list[str] | None = None, *, all_pending: bool = False) -> dict:
        if self.busy():
            return {"ok": False, "error": "A library download is already running.", "job": self.snapshot()}
        by_id = {v["id"]: v for v in manifest["videos"]}
        if all_pending or ids is None:
            chosen = [v for v in manifest["videos"] if file_status(v, self.videos_dir) != "current"]
        else:
            unknown = [i for i in ids if i not in by_id]
            if unknown:
                return {"ok": False, "error": f"Unknown video id(s): {', '.join(unknown)}.", "job": self.snapshot()}
            chosen = [by_id[i] for i in ids]
        # Skip ones already current when a mixed list is sent.
        chosen = [v for v in chosen if file_status(v, self.videos_dir) != "current"]
        if not chosen:
            with self._lock:
                self._job = Job(state="done", message="Everything selected is already on disk.", finished_at=time.time())
            return {"ok": True, "error": None, "job": self.snapshot()}
        job = Job(
            state="downloading",
            ids=[v["id"] for v in chosen],
            files_total=len(chosen),
            bytes_total=sum(v["size"] for v in chosen),
            message=f"Downloading {len(chosen)} video(s).",
            started_at=time.time(),
        )
        with self._lock:
            self._job = job
            self._cancel.clear()
        thread = threading.Thread(target=self._run, args=(list(chosen),), name="firstbook-library-download", daemon=True)
        self._thread = thread
        thread.start()
        return {"ok": True, "error": None, "job": self.snapshot()}

    def wait(self, timeout: float | None = None) -> dict:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self.snapshot()

    def _run(self, entries: list[dict]) -> None:
        offset = 0
        try:
            for index, entry in enumerate(entries, start=1):
                if self._cancel.is_set():
                    raise DownloadError("Download cancelled.")
                with self._lock:
                    self._job.current_id = entry["id"]
                    self._job.current_title = entry["title"]
                    self._job.message = f"Downloading “{entry['title']}” ({index} of {len(entries)})."

                def progress(done: int, total: int, *, _offset=offset, _entry=entry) -> None:
                    with self._lock:
                        self._job.bytes_done = _offset + min(done, _entry["size"])

                download_entry(entry, self.videos_dir, progress=progress)
                offset += entry["size"]
                with self._lock:
                    self._job.files_done = index
                    self._job.bytes_done = offset
            with self._lock:
                self._job.state = "done"
                self._job.current_id = None
                self._job.message = f"Downloaded {len(entries)} video(s)."
                self._job.finished_at = time.time()
        except (DownloadError, ManifestError, OSError) as exc:
            with self._lock:
                self._job.state = "error"
                self._job.error = str(exc)
                self._job.message = str(exc)
                self._job.finished_at = time.time()


def public_status(videos_dir: Path, manager: LibraryDownloads | None = None) -> dict:
    """Shape used by /videos and /api/videos/library (works fully offline)."""
    settings = library_settings()
    state = load_state(videos_dir)
    return {
        "display_url": settings.display_url,
        "manifest_url": settings.manifest_url,
        "placeholder": settings.is_placeholder,
        "last_checked": state.get("last_checked"),
        "last_checked_ok": state.get("last_checked_ok"),
        "last_success": state.get("last_success"),
        "error": state.get("error"),
        "video_count": state.get("video_count"),
        "job": manager.snapshot() if manager is not None else Job().to_dict(),
    }
