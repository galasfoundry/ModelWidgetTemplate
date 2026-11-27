"""Branch lanes: manifest fields, grouping, and streaming from a mock library."""

from __future__ import annotations

import hashlib
import http.server
import json
import subprocess
import threading
from pathlib import Path

import numpy as np

from app import create_app
from firstbook import branch_audio, branches, remote_library
from paths import PROJECT_ROOT
from tests.firstbook_helpers import image_settings, make_library

SCHEMA = remote_library.SCHEMA
EPISODES_URL = "https://github.com/galasfoundry/ModelWidgetTemplate/releases/download/episodes-v1/"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_parse_manifest_keeps_branch_and_parent():
    data = {
        "schema": SCHEMA,
        "base_url": "https://example.invalid/episodes/",
        "videos": [{
            "id": "004-fire-triangle",
            "file": "originals/004-fire-triangle.mp4",
            "title": "The fire triangle",
            "topic": "fire",
            "license": "CC BY 4.0",
            "url": "https://example.invalid/004-fire-triangle.mp4",
            "size": 12,
            "sha256": "ab" * 32,
            "duration_s": 114.4,
            "subtitles_url": "https://example.invalid/004-fire-triangle.srt",
            "branch": "Fundamentals",
            "parent": "002",
            "episode": "004",
        }, {
            "id": "001-head-and-flow",
            "file": "originals/001-head-and-flow.mp4",
            "title": "Head and flow",
            "url": "https://example.invalid/001-head-and-flow.mp4",
            "size": 4,
            "sha256": "cd" * 32,
            "parent": None,
            "episode": "001",
            "branch": "fundamentals",
        }],
    }
    parsed = remote_library.parse_manifest(data)
    fire, head = parsed["videos"]
    assert fire["branch"] == "fundamentals"
    assert fire["parent"] == "002"
    assert fire["episode"] == "004"
    assert fire["subtitles"].endswith("004-fire-triangle.srt")
    assert head["parent"] is None and head["branch"] == "fundamentals"
    # Older manifests without branch fields still parse.
    bare = remote_library.parse_manifest({"schema": SCHEMA, "videos": [{
        "id": "clip", "file": "power/clip.mp4", "url": "https://example.invalid/clip.mp4",
        "size": 1, "sha256": "a" * 64,
    }]})
    assert bare["videos"][0]["branch"] == "" and bare["videos"][0]["parent"] is None


def test_build_remote_manifest_includes_branch_fields(tmp_path):
    folder = tmp_path / "videos"
    orig = folder / "originals" / "004-fire-triangle.mp4"
    orig.parent.mkdir(parents=True)
    payload = b"fire-bytes"
    orig.write_bytes(payload)
    orig.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHeat\n", encoding="utf-8")
    (orig.with_name(orig.name + ".json")).write_text(json.dumps({
        "id": "004-fire-triangle",
        "title": "The fire triangle",
        "topic": "fire",
        "episode": "004",
        "branch": "fundamentals",
        "parent": "002",
        "duration_s": 10.5,
        "subtitles": "004-fire-triangle.srt",
        "license": "CC BY 4.0",
    }), encoding="utf-8")
    manifest = remote_library.build_remote_manifest(
        folder, base_url="https://example.invalid/episodes-v1/", url_mode="flat",
    )
    entry = manifest["videos"][0]
    assert entry["id"] == "004-fire-triangle"
    assert entry["branch"] == "fundamentals" and entry["parent"] == "002" and entry["episode"] == "004"
    assert entry["sha256"] == _sha(payload) and entry["size"] == len(payload)
    assert entry["url"] == "https://example.invalid/episodes-v1/004-fire-triangle.mp4"
    assert entry["subtitles_url"] == "https://example.invalid/episodes-v1/004-fire-triangle.srt"
    again = remote_library.parse_manifest(manifest)["videos"][0]
    assert again["branch"] == "fundamentals" and again["parent"] == "002"


def test_nest_branches_groups_fundamentals_and_planned_forks():
    episodes = [
        {"episode": "002", "branch": "fundamentals", "parent": "001", "title": "Watts", "file": "originals/002.mp4"},
        {"episode": "004", "branch": "fundamentals", "parent": "002", "title": "Fire", "file": "originals/004.mp4"},
        {"episode": "001", "branch": "fundamentals", "parent": None, "title": "Head", "file": "originals/001.mp4"},
        {"episode": "003", "branch": "fundamentals", "parent": "002", "title": "Volts", "file": "originals/003.mp4"},
    ]
    tree = branches.nest_branches(episodes, branches.load_branch_defs(), keep_empty=True)
    by_id = {lane["id"]: lane for lane in tree}
    assert "micro-hydro" not in by_id and "fire" not in by_id
    fund = by_id["fundamentals"]
    assert [ep["episode"] for ep in fund["episodes"]] == ["001", "002", "003", "004"]
    forks = {ep["episode"]: [fork["id"] for fork in ep["forks"]] for ep in fund["episodes"]}
    assert forks["001"] == ["micro-hydro"]
    assert forks["003"] == ["batteries-inverters"]
    assert forks["004"] == ["fire"]
    assert forks["002"] == []
    micro = fund["episodes"][0]["forks"][0]
    assert micro["parent_episode"] == "001" and "001" in micro["parent_label"]
    assert micro["episodes"] == []
    water = by_id["water-treatment"]
    assert water["episodes"] == [] and "F6" in water["parent_label"]
    # A search drops empty planned lanes.
    narrowed = branches.nest_branches(
        [ep for ep in episodes if ep["episode"] == "004"],
        branches.load_branch_defs(),
        keep_empty=False,
    )
    assert [lane["id"] for lane in narrowed] == ["fundamentals"]
    assert [ep["episode"] for ep in narrowed[0]["episodes"]] == ["004"]
    assert narrowed[0]["episodes"][0]["forks"] == []


def test_active_manifest_follows_the_configured_host(tmp_path, monkeypatch):
    shipped = tmp_path / "shipped.json"
    entry = {
        "id": "001-head-and-flow",
        "file": "originals/001-head-and-flow.mp4",
        "title": "Head and flow",
        "url": EPISODES_URL + "001-head-and-flow.mp4",
        "size": 8,
        "sha256": "a" * 64,
        "branch": "fundamentals",
        "parent": None,
        "episode": "001",
    }
    shipped.write_text(json.dumps({
        "schema": SCHEMA, "base_url": EPISODES_URL, "videos": [entry],
    }), encoding="utf-8")
    monkeypatch.setattr(remote_library, "SHIPPED_MANIFEST_PATH", shipped)
    monkeypatch.delenv("FIRSTBOOK_LIBRARY_MANIFEST", raising=False)
    monkeypatch.setenv("FIRSTBOOK_LIBRARY_URL", "http://127.0.0.1:9/other/")
    assert remote_library.active_manifest(tmp_path / "videos") is None
    monkeypatch.setenv("FIRSTBOOK_LIBRARY_URL", EPISODES_URL)
    loaded = remote_library.active_manifest(tmp_path / "videos")
    assert loaded["videos"][0]["episode"] == "001"
    # A cache for the configured host wins over the shipped file.
    cache_dir = tmp_path / "videos"
    cache_dir.mkdir()
    cached_entry = dict(entry, title="From cache", url="http://127.0.0.1:9/other/001.mp4")
    (cache_dir / remote_library.CACHED_MANIFEST_NAME).write_text(json.dumps({
        "schema": SCHEMA, "base_url": "http://127.0.0.1:9/other/", "videos": [cached_entry],
    }), encoding="utf-8")
    monkeypatch.setenv("FIRSTBOOK_LIBRARY_URL", "http://127.0.0.1:9/other/")
    assert remote_library.active_manifest(cache_dir)["videos"][0]["title"] == "From cache"


def test_shipped_manifest_lists_four_episodes_with_hashes():
    path = PROJECT_ROOT / "config" / "library_manifest.json"
    parsed = remote_library.parse_manifest(json.loads(path.read_text(encoding="utf-8")))
    assert parsed["base_url"].rstrip("/") + "/" == EPISODES_URL
    assert len(parsed["videos"]) == 4
    episodes = []
    for entry in parsed["videos"]:
        assert entry["branch"] == "fundamentals"
        assert entry["sha256"] and len(entry["sha256"]) == 64
        assert entry["size"] > 1000
        assert entry["duration_s"] and entry["duration_s"] > 60
        assert entry["url"].startswith(EPISODES_URL) and entry["url"].endswith(".mp4")
        assert entry["subtitles_url"].startswith(EPISODES_URL) and entry["subtitles_url"].endswith(".srt")
        assert "github.com" in entry["url"] and "/originals/" not in entry["url"]
        episodes.append({
            "episode": entry["episode"],
            "branch": entry["branch"],
            "parent": entry["parent"],
            "title": entry["title"],
            "file": entry["file"],
        })
    by_ep = {ep["episode"]: ep for ep in episodes}
    assert set(by_ep) == {"001", "002", "003", "004"}
    assert by_ep["001"]["parent"] is None
    assert by_ep["002"]["parent"] == "001"
    assert by_ep["003"]["parent"] == "002"
    assert by_ep["004"]["parent"] == "002"
    tree = branches.nest_branches(episodes, branches.load_branch_defs(), keep_empty=True)
    fund = next(lane for lane in tree if lane["id"] == "fundamentals")
    assert [ep["episode"] for ep in fund["episodes"]] == ["001", "002", "003", "004"]


def test_branch_themes_cover_every_lane_and_differ():
    themes = branch_audio.load_themes()
    ids = [row["id"] for row in branches.load_branch_defs()]
    assert set(themes["branches"]) == set(ids)
    base = themes["base"]
    assert base["chords"][0][0] == -24 and base["bar_s"] == 4.0
    fund = themes["branches"]["fundamentals"]
    assert fund["transpose"] == 0 and fund["waveform"] == "sine" and fund["seed"] == 7
    assert themes["branches"]["micro-hydro"]["transpose"] == -5
    assert themes["branches"]["fire"]["waveform"] == "soft-saw"
    assert 5 <= fund["duration_s"] <= 10
    left = branch_audio.synthesize(themes, "fundamentals", duration_s=0.5)
    right = branch_audio.synthesize(themes, "fire", duration_s=0.5)
    assert left.dtype == np.float32
    assert len(left) == int(0.5 * themes["sample_rate"])
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert float(np.max(np.abs(left - right))) > 0.02
    folder = PROJECT_ROOT / "static" / "audio" / "branches"
    for ident in ids:
        ogg = folder / f"{ident}.ogg"
        mp3 = folder / f"{ident}.mp3"
        assert ogg.read_bytes()[:4] == b"OggS"
        assert mp3.read_bytes()[:3] == b"ID3"
        probed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(ogg)],
            capture_output=True, text=True, check=False,
        )
        assert probed.returncode == 0
        assert 5 <= float(probed.stdout.strip()) <= 10


def _serve(root: Path):
    """Tiny file server that honours a single Range, which the stdlib handler does not."""

    class RangeHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            path = (root / self.path.split("?", 1)[0].lstrip("/")).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError:
                self.send_error(404)
                return
            if not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            spec = self.headers.get("Range") or ""
            if spec.startswith("bytes=") and "-" in spec[6:]:
                start_s, end_s = spec[6:].split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else len(data) - 1
                end = min(end, len(data) - 1)
                chunk = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/"
    return server, thread, base


def test_branch_page_streams_from_mock_library_and_plays_offline_after_download(tmp_path, monkeypatch):
    root = tmp_path / "host"
    payload = b"STREAM" * 3000
    rel = "originals/001-head-and-flow.mp4"
    (root / rel).parent.mkdir(parents=True)
    (root / rel).write_bytes(payload)
    srt = "1\n00:00:00,000 --> 00:00:01,000\nHello branch\n"
    (root / "originals/001-head-and-flow.srt").write_text(srt, encoding="utf-8")
    server, thread, base = _serve(root)
    try:
        entry = {
            "id": "001-head-and-flow",
            "file": rel,
            "title": "Head and flow: how much power is in your stream?",
            "topic": "power",
            "license": "CC BY 4.0",
            "credit": "“Head and flow” by The First Book / galasfoundry, CC BY 4.0",
            "source_url": "The First Book original",
            "url": base + rel,
            "size": len(payload),
            "sha256": _sha(payload),
            "duration_s": 12,
            "subtitles_url": base + "originals/001-head-and-flow.srt",
            "subtitles": "originals/001-head-and-flow.srt",
            "branch": "fundamentals",
            "parent": None,
            "episode": "001",
            "sidecar": {"creator": "The First Book / galasfoundry", "description": "A stream."},
        }
        (root / "manifest.json").write_text(json.dumps({
            "schema": SCHEMA, "base_url": base, "videos": [entry],
        }), encoding="utf-8")
        lib = make_library(tmp_path)
        monkeypatch.setenv("FIRSTBOOK_LIBRARY_URL", base)
        monkeypatch.delenv("FIRSTBOOK_LIBRARY_MANIFEST", raising=False)
        client = create_app(image_settings(tmp_path), library=lib).test_client()
        checked = client.post("/api/videos/library/check")
        assert checked.status_code == 200 and checked.get_json()["ok"] is True
        page = client.get("/videos").get_data(as_text=True)
        assert 'id="branch-tree"' in page
        assert "Fundamentals" in page and "Micro-hydro" in page
        assert "Branches from 001" in page
        assert "Head and flow" in page and "Not downloaded" in page
        assert 'data-branch="micro-hydro"' in page
        assert "Sound off" in page
        watch = client.get("/videos/watch/" + rel)
        assert watch.status_code == 200
        html = watch.get_data(as_text=True)
        assert "/videos/stream/" + rel in html
        assert base + rel not in html
        assert "Streaming from the library" in html
        assert "needs a network" in html
        part = client.get("/videos/stream/" + rel, headers={"Range": "bytes=0-5"})
        assert part.status_code == 206 and part.data == payload[:6]
        assert client.get("/videos/stream/" + rel).data == payload
        assert client.get("/videos/stream/originals/missing.mp4").status_code == 404
        started = client.post("/api/videos/library/download", json={"ids": ["001-head-and-flow"]})
        assert started.status_code == 200 and started.get_json()["ok"] is True
        import time
        end = time.time() + 8
        job = client.get("/api/videos/library/status").get_json()
        while job["state"] in {"downloading", "checking"} and time.time() < end:
            time.sleep(0.05)
            job = client.get("/api/videos/library/status").get_json()
        assert job["state"] == "done"
        saved = lib.videos_dir / rel
        assert saved.read_bytes() == payload
        offline = client.get("/videos/watch/" + rel).get_data(as_text=True)
        assert "/videos/file/" + rel in offline
        assert "/videos/stream/" not in offline
        assert "Saved on this computer" in offline
        assert "keeps playing with no network" in offline
        assert base + rel not in offline
        assert client.get("/videos/file/" + rel).data == payload
        listed = client.get("/videos").get_data(as_text=True)
        assert "Downloaded" in listed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
