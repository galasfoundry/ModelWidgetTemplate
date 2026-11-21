"""Remote video library: manifest parsing, sha256, resume, offline fallback, button."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from functools import partial
from pathlib import Path

import pytest

from app import create_app
from firstbook import remote_library, videos
from scripts import build_video_manifest
from tests.firstbook_helpers import image_settings, make_library

SCHEMA = remote_library.SCHEMA


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(name: str, data: bytes, *, topic: str = "power", **over) -> tuple[dict, bytes]:
    rel = f"{topic}/{name}.mp4"
    entry = {
        "id": name,
        "file": rel,
        "title": name.replace("-", " ").title(),
        "topic": topic,
        "license": "CC BY 4.0",
        "credit": f"“{name}” by The First Book, CC BY 4.0",
        "source_url": "https://example.invalid/source",
        "url": rel,
        "size": len(data),
        "sha256": _sha(data),
        "duration_s": 12,
        "subtitles_url": f"{topic}/{name}.srt",
        "subtitles": f"{topic}/{name}.srt",
        "sidecar": {"creator": "The First Book / galasfoundry", "description": f"A clip called {name}."},
    }
    entry.update(over)
    return entry, data


class _RecordingHandler(http.server.SimpleHTTPRequestHandler):
    """Static files plus a log of Range headers, for the resume test."""

    ranges: list
    requests: list

    def log_message(self, fmt, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        type(self).ranges.append(self.headers.get("Range"))
        type(self).requests.append(self.path)
        return super().do_GET()


@pytest.fixture
def hosted(tmp_path):
    """A local HTTP server of two tiny fake videos and a valid manifest."""
    root = tmp_path / "host"
    clip_a = b"AAAA" * 2048  # 8 KiB
    clip_b = b"BBBB" * 4096  # 16 KiB
    srt = "1\n00:00:00,000 --> 00:00:01,000\nHello from the library\n"
    entry_a, _ = _entry("clip-a", clip_a)
    entry_b, _ = _entry("clip-b", clip_b, topic="hydro")
    for entry, payload in ((entry_a, clip_a), (entry_b, clip_b)):
        path = root / entry["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        (path.parent / f"{path.stem}.srt").write_text(srt, encoding="utf-8")
    manifest = {
        "schema": SCHEMA,
        "version": 1,
        "generated_at": "2026-11-20T12:00:00Z",
        "base_url": "",  # filled after the server starts
        "videos": [entry_a, entry_b],
    }

    handler = type(
        "Handler",
        (_RecordingHandler,),
        {"ranges": [], "requests": []},
    )
    # Bind on 127.0.0.1:0 first so we know the URL before serve_forever.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(root)))
    base = f"http://127.0.0.1:{server.server_address[1]}/"
    manifest["base_url"] = base
    # Resolve relative file URLs now, as parse_manifest would.
    for item in manifest["videos"]:
        item["url"] = base + item["url"]
        item["subtitles_url"] = base + item["subtitles_url"]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "base": base,
            "root": root,
            "manifest": manifest,
            "handler": handler,
            "clip_a": clip_a,
            "clip_b": clip_b,
            "entry_a": entry_a,
            "entry_b": entry_b,
            "srt": srt,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _client(tmp_path, lib, monkeypatch, *, manifest_url=None, manifest_file=None):
    if manifest_url:
        monkeypatch.setenv("FIRSTBOOK_LIBRARY_URL", manifest_url)
        monkeypatch.delenv("FIRSTBOOK_LIBRARY_MANIFEST", raising=False)
    if manifest_file:
        monkeypatch.setenv("FIRSTBOOK_LIBRARY_MANIFEST", str(manifest_file))
    return create_app(image_settings(tmp_path), library=lib).test_client()


def test_parse_manifest_accepts_aliases_and_rejects_bad_entries():
    data = {
        "schema": SCHEMA,
        "videos": [{
            "id": "demo",
            "file": "power/demo.mp4",
            "title": "Demo",
            "topic": "Power",
            "licence": "CC BY 4.0",
            "credit": "Demo credit",
            "source_url": "https://example.invalid/src",
            "file_url": "https://example.invalid/demo.mp4",
            "size": 12,
            "sha256": "a" * 64,
            "duration": 3.5,
            "subtitles_url": "https://example.invalid/demo.srt",
            "sidecar": {"creator": "Ada", "tags": ["Hydro"]},
        }],
    }
    parsed = remote_library.parse_manifest(data)
    item = parsed["videos"][0]
    assert item["license"] == "CC BY 4.0" and item["credit"] == "Demo credit"
    assert item["url"] == "https://example.invalid/demo.mp4"
    assert item["topic"] == "power" and item["tags"] == ["hydro"]
    assert item["duration_s"] == 3.5
    assert item["subtitles"].endswith("demo.srt")

    with pytest.raises(remote_library.ManifestError, match="schema"):
        remote_library.parse_manifest({"version": 1, "videos": []})
    with pytest.raises(remote_library.ManifestError, match="safe file path|sha256|id"):
        remote_library.parse_manifest({"schema": SCHEMA, "videos": [{
            "id": "x", "file": "../etc/passwd.mp4", "url": "https://example.invalid/x.mp4",
            "size": 1, "sha256": "a" * 64,
        }]})
    with pytest.raises(remote_library.ManifestError, match="sha256|safe"):
        remote_library.parse_manifest({"schema": SCHEMA, "videos": [{
            "id": "x", "file": "power/x.mp4", "url": "https://example.invalid/x.mp4",
            "size": 1, "sha256": "zzzz",
        }]})


def test_load_manifest_from_local_file(tmp_path, monkeypatch):
    entry, _ = _entry("local-only", b"zzzz")
    entry["url"] = "https://example.invalid/power/local-only.mp4"
    entry["subtitles_url"] = "https://example.invalid/power/local-only.srt"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": SCHEMA, "videos": [entry]}), encoding="utf-8")
    monkeypatch.setenv("FIRSTBOOK_LIBRARY_MANIFEST", str(path))
    monkeypatch.delenv("FIRSTBOOK_LIBRARY_URL", raising=False)
    loaded = remote_library.fetch_manifest()
    assert loaded["videos"][0]["id"] == "local-only"
    assert remote_library.library_settings().manifest_url == str(path)


def test_default_library_url_is_the_live_release(monkeypatch):
    monkeypatch.delenv("FIRSTBOOK_LIBRARY_URL", raising=False)
    monkeypatch.delenv("FIRSTBOOK_LIBRARY_MANIFEST", raising=False)
    settings = remote_library.library_settings()
    assert settings.is_placeholder is False
    assert settings.base_url == remote_library.DEFAULT_LIBRARY_URL
    assert settings.base_url == (
        "https://github.com/galasfoundry/ModelWidgetTemplate/releases/download/library-v1/"
    )
    assert settings.manifest_url == settings.base_url + "manifest.json"
    # A leftover placeholder URL is still flagged, and is not the default.
    assert remote_library.is_placeholder_url(remote_library.PLACEHOLDER_LIBRARY_URL)
    assert "library-placeholder" in remote_library.PLACEHOLDER_LIBRARY_URL


def test_sha_mismatch_rejects_and_leaves_existing_file(hosted, tmp_path):
    lib = make_library(tmp_path)
    entry = dict(hosted["entry_a"])
    entry["url"] = hosted["manifest"]["videos"][0]["url"]
    entry["sha256"] = "b" * 64  # wrong
    dest = lib.videos_dir / entry["file"]
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"keep-me")
    with pytest.raises(remote_library.HashMismatch, match="sha256"):
        remote_library.download_entry(entry, lib.videos_dir)
    assert dest.read_bytes() == b"keep-me"
    assert not remote_library.part_path(lib.videos_dir, entry["file"]).exists()
    assert not (dest.parent / (dest.name + ".part")).exists()


def test_resume_from_part_file_uses_range(hosted, tmp_path):
    lib = make_library(tmp_path)
    entry = dict(hosted["manifest"]["videos"][0])
    rel = entry["file"]
    part = remote_library.part_path(lib.videos_dir, rel)
    part.parent.mkdir(parents=True)
    part.write_bytes(hosted["clip_a"][:3000])
    hosted["handler"].ranges.clear()
    dest = remote_library.download_entry(entry, lib.videos_dir)
    assert dest.read_bytes() == hosted["clip_a"]
    assert dest.stat().st_size == entry["size"]
    assert remote_library.sha256_file(dest) == entry["sha256"]
    assert any(r and r.startswith("bytes=3000-") for r in hosted["handler"].ranges)
    assert not part.exists()
    # Sidecar and captions landed next to the video.
    assert (dest.with_name(dest.name + ".json")).is_file()
    assert "Hello from the library" in (dest.with_suffix(".srt")).read_text(encoding="utf-8")
    indexed = {v["id"]: v for v in videos.index(lib.videos_dir)}
    assert rel in indexed and indexed[rel]["creator"] == "The First Book / galasfoundry"


def test_download_skips_current_file_without_refetch(hosted, tmp_path):
    lib = make_library(tmp_path)
    entry = dict(hosted["manifest"]["videos"][0])
    dest = lib.videos_dir / entry["file"]
    dest.parent.mkdir(parents=True)
    dest.write_bytes(hosted["clip_a"])
    hosted["handler"].requests.clear()
    remote_library.download_entry(entry, lib.videos_dir)
    video_gets = [p for p in hosted["handler"].requests if p.endswith(".mp4")]
    assert video_gets == []


def test_offline_fallback_keeps_disk_and_last_success(hosted, tmp_path, monkeypatch):
    lib = make_library(tmp_path)
    (lib.videos_dir / "already.mp4").write_bytes(b"local-copy")
    monkeypatch.setenv("FIRSTBOOK_LIBRARY_URL", hosted["base"])
    monkeypatch.delenv("FIRSTBOOK_LIBRARY_MANIFEST", raising=False)
    first = remote_library.check_library(lib.videos_dir)
    assert first["ok"] and first["comparison"]["counts"]["new"] == 2
    success_at = first["state"]["last_checked"]
    assert first["state"]["last_checked_ok"] and success_at

    monkeypatch.setenv("FIRSTBOOK_LIBRARY_URL", "http://127.0.0.1:1/")  # closed port
    second = remote_library.check_library(lib.videos_dir)
    assert second["ok"] is False and second["offline"] is True
    assert second["state"]["last_success"] == success_at
    assert second["state"]["last_checked"]  # a later attempt was recorded
    assert second["state"]["last_checked_ok"] is False
    assert "offline" in (second["error"] or "").lower() or "reach" in (second["error"] or "").lower()

    client = _client(tmp_path, lib, monkeypatch, manifest_url="http://127.0.0.1:1/")
    page = client.get("/videos").get_data(as_text=True)
    assert "Download library" in page and 'id="download-library"' in page
    assert "already.mp4" in page or "Already" in page
    assert success_at in page or "last checked" in page.lower()
    # Playback of the local file is unchanged.
    assert client.get("/videos/file/already.mp4").status_code == 200
    body = client.get("/api/videos/library").get_json()
    assert body["last_success"] == success_at
    assert body["placeholder"] is False


def test_button_route_and_download_job(hosted, tmp_path, monkeypatch):
    lib = make_library(tmp_path)
    client = _client(tmp_path, lib, monkeypatch, manifest_url=hosted["base"])
    page = client.get("/videos").get_data(as_text=True)
    assert "Download library" in page and 'id="download-library"' in page
    assert "placeholder" not in page.lower() or "library-placeholder" not in page
    assert "The remote manifest has not been checked yet." in page

    checked = client.post("/api/videos/library/check")
    assert checked.status_code == 200
    payload = checked.get_json()
    assert payload["ok"] is True
    assert payload["comparison"]["counts"] == {"new": 2, "updated": 0, "current": 0}
    assert payload["comparison"]["download_bytes"] == len(hosted["clip_a"]) + len(hosted["clip_b"])
    ids = [v["id"] for v in payload["comparison"]["videos"]]
    assert ids == ["clip-a", "clip-b"]

    started = client.post("/api/videos/library/download", json={"ids": ["clip-a"]})
    assert started.status_code == 200 and started.get_json()["ok"] is True
    job = current_app_wait(client)
    assert job["state"] == "done" and job["files_done"] == 1
    dest = lib.videos_dir / "power/clip-a.mp4"
    assert dest.read_bytes() == hosted["clip_a"]

    # The list page and watch page pick up the new file; no player change.
    listing = client.get("/videos").get_data(as_text=True)
    assert "Clip A" in listing or "clip-a" in listing.lower() or "Clip-A" in listing
    watch = client.get("/videos/watch/power/clip-a.mp4")
    assert watch.status_code == 200
    html = watch.get_data(as_text=True)
    assert "Credit:" in html and "CC BY 4.0" in html
    assert client.get("/videos/file/power/clip-a.mp4").data == hosted["clip_a"]

    # Second check now shows clip-a current and clip-b still new.
    again = client.post("/api/videos/library/check").get_json()
    by_id = {v["id"]: v for v in again["comparison"]["videos"]}
    assert by_id["clip-a"]["status"] == "current"
    assert by_id["clip-b"]["status"] == "new"

    all_pending = client.post("/api/videos/library/download", json={"all": True})
    assert all_pending.status_code == 200
    job = current_app_wait(client)
    assert job["state"] == "done"
    assert (lib.videos_dir / "hydro/clip-b.mp4").read_bytes() == hosted["clip_b"]


def current_app_wait(client, timeout=8.0):
    """Poll the job status until it leaves the downloading state."""
    import time
    end = time.time() + timeout
    job = client.get("/api/videos/library/status").get_json()
    while job["state"] in {"downloading", "checking", "queued"} and time.time() < end:
        time.sleep(0.05)
        job = client.get("/api/videos/library/status").get_json()
    return job


def test_updated_file_is_replaced_when_hash_changes(hosted, tmp_path):
    lib = make_library(tmp_path)
    entry = dict(hosted["manifest"]["videos"][0])
    dest = lib.videos_dir / entry["file"]
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"old-version-xxxx")
    assert remote_library.file_status(entry, lib.videos_dir) == "updated"
    remote_library.download_entry(entry, lib.videos_dir)
    assert dest.read_bytes() == hosted["clip_a"]


def test_build_remote_manifest_from_sidecars_and_catalog(tmp_path):
    lib = make_library(tmp_path)
    folder = lib.videos_dir
    # A First Book original with sidecar + captions, plus a catalog-style file.
    orig = folder / "originals" / "001-head-and-flow.mp4"
    orig.parent.mkdir()
    orig.write_bytes(b"original-bytes")
    (orig.with_name(orig.name + ".json")).write_text(json.dumps({
        "file": "originals/001-head-and-flow.mp4",
        "title": "Head and flow: how much power is in your stream?",
        "topic": "power",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "credit": "by The First Book / galasfoundry",
        "attribution": "by The First Book / galasfoundry",
        "creator": "The First Book / galasfoundry",
        "duration_s": 119.1,
        "subtitles": "001-head-and-flow.srt",
        "fact_check": "Fact-checked by Murder, 19 Nov 2026. PASS WITH FIXES applied.",
        "episode": "001",
    }), encoding="utf-8")
    orig.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")

    from firstbook.library import VIDEOS_CATALOG_PATH
    catalog_item = videos.load_catalog(VIDEOS_CATALOG_PATH)["videos"][0]
    cat_path = folder / catalog_item["file"]
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    cat_path.write_bytes(b"catalog-bytes")

    out = tmp_path / "manifest.json"
    assert build_video_manifest.main([
        str(folder), "--remote",
        "--base-url", "https://example.invalid/library/",
        "--url-mode", "path",
        "-o", str(out),
    ]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == SCHEMA
    by_id = {v["id"]: v for v in data["videos"]}
    head = by_id["001"] if "001" in by_id else by_id["001-head-and-flow"]
    # id comes from sidecar episode "001" (preferred) or the stem.
    assert head["file"] == "originals/001-head-and-flow.mp4"
    assert head["size"] == len(b"original-bytes")
    assert head["sha256"] == _sha(b"original-bytes")
    assert head["url"].endswith("originals/001-head-and-flow.mp4")
    assert head["subtitles_url"].endswith("originals/001-head-and-flow.srt")
    assert head["license"] == "CC BY 4.0"
    assert "Murder" in head["sidecar"]["fact_check"]
    # Catalog metadata is pulled in for the third-party file.
    other = next(v for v in data["videos"] if v["file"] == catalog_item["file"])
    assert other["credit"] or other["sidecar"].get("attribution")
    assert catalog_item["creator"] in (other["sidecar"].get("creator") or "") + (other.get("credit") or "")
    # --url-mode flat uses only the filename in the URL.
    flat = remote_library.build_remote_manifest(
        folder, base_url="https://example.invalid/r/", url_mode="flat",
    )
    flat_head = next(v for v in flat["videos"] if "head-and-flow" in v["file"])
    assert flat_head["url"] == "https://example.invalid/r/001-head-and-flow.mp4"
    assert flat_head["file"] == "originals/001-head-and-flow.mp4"


def test_placeholder_check_mentions_placeholder(tmp_path, monkeypatch):
    lib = make_library(tmp_path)
    monkeypatch.delenv("FIRSTBOOK_LIBRARY_MANIFEST", raising=False)
    monkeypatch.setenv("FIRSTBOOK_LIBRARY_URL", remote_library.PLACEHOLDER_LIBRARY_URL)
    # Don't actually hit GitHub: point at a closed port but keep the placeholder flag
    # by using the default URL on the settings and a closed port via manifest override? 
    # Use a clearly-placeholder URL that cannot be reached.
    monkeypatch.setenv(
        "FIRSTBOOK_LIBRARY_URL",
        "http://127.0.0.1:1/library-placeholder/",
    )
    result = remote_library.check_library(lib.videos_dir)
    assert result["ok"] is False and result["placeholder"] is True
    assert "placeholder" in (result["error"] or "").lower()
    client = _client(tmp_path, lib, monkeypatch, manifest_url="http://127.0.0.1:1/library-placeholder/")
    page = client.get("/videos").get_data(as_text=True)
    assert "Download library" in page
    assert "placeholder" in page.lower()
