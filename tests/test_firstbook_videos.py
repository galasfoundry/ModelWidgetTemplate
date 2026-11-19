"""Video library: manifest + sidecar metadata, search, Range streaming, manifest script."""

from __future__ import annotations

import json

from app import create_app
from firstbook import videos
from scripts import build_video_manifest
from tests.firstbook_helpers import image_settings, make_library


def _setup(tmp_path):
    lib = make_library(tmp_path)
    (lib.videos_dir / "electronics").mkdir()
    (lib.videos_dir / "electronics" / "soldering_basics.mp4").write_bytes(bytes(range(256)) * 4)
    (lib.videos_dir / "fix-a-puncture.webm").write_bytes(b"webm" * 10)
    (lib.videos_dir / ".hidden").mkdir()
    (lib.videos_dir / ".hidden" / "skip.mp4").write_bytes(b"x")
    (lib.videos_dir / "readme.txt").write_text("not a video")
    manifest = {"version": 1, "videos": [{
        "file": "electronics/soldering_basics.mp4", "title": "Soldering basics",
        "topic": "electronics", "tags": ["Soldering", "repair"], "duration_s": 754}]}
    (lib.videos_dir / "videos.json").write_text(json.dumps(manifest), encoding="utf-8")
    (lib.videos_dir / "fix-a-puncture.webm.json").write_text(
        json.dumps({"topic": "repair", "tags": ["bike"]}), encoding="utf-8")
    return lib


def test_index_merges_manifest_sidecar_and_bare_files(tmp_path):
    lib = _setup(tmp_path)
    items = {v["id"]: v for v in videos.index(lib.videos_dir)}
    assert set(items) == {"electronics/soldering_basics.mp4", "fix-a-puncture.webm"}
    solder = items["electronics/soldering_basics.mp4"]
    assert solder["title"] == "Soldering basics" and solder["tags"] == ["soldering", "repair"]
    assert solder["duration_s"] == 754 and solder["mimetype"] == "video/mp4"
    puncture = items["fix-a-puncture.webm"]
    assert puncture["title"] == "Fix a puncture" and puncture["topic"] == "repair"
    assert puncture["duration_s"] is None


def test_search_by_words_topic_and_tag(tmp_path):
    items = videos.index(_setup(tmp_path).videos_dir)
    assert [v["id"] for v in videos.search(items, "solder")] == ["electronics/soldering_basics.mp4"]
    assert len(videos.search(items, "", tag="repair")) == 1
    assert [v["topic"] for v in videos.search(items, "", topic="repair")] == ["repair"]
    assert videos.search(items, "solder bike") == []
    assert "electronics" in videos.TOPICS and "3d-printing" in videos.TOPICS


def test_api_and_range_streaming(tmp_path):
    lib = _setup(tmp_path)
    client = create_app(image_settings(tmp_path), library=lib).test_client()
    body = client.get("/api/videos?q=soldering").get_json()
    assert body["count"] == 1 and body["total"] == 2
    url = body["videos"][0]["url"]
    full = client.get(url)
    assert full.status_code == 200 and full.headers["Content-Type"] == "video/mp4"
    assert full.headers.get("Accept-Ranges") == "bytes"
    part = client.get(url, headers={"Range": "bytes=0-9"})
    assert part.status_code == 206 and part.data == bytes(range(10))
    assert client.get("/videos/file/../videos.json").status_code == 404
    assert client.get("/videos/file/readme.txt").status_code == 404
    assert client.get("/videos/file/nothing.mp4").status_code == 404
    assert "Soldering basics" in client.get("/videos").get_data(as_text=True)
    assert client.get("/videos/watch/electronics/soldering_basics.mp4").status_code == 200
    assert client.get("/videos/watch/nope.mp4").status_code == 404


def test_manifest_script_keeps_existing_entries(tmp_path, capsys):
    lib = _setup(tmp_path)
    (lib.videos_dir / "new_clip.mkv").write_bytes(b"mkv")
    assert build_video_manifest.main([str(lib.videos_dir), "--topic", "radio"]) == 0
    data = json.loads((lib.videos_dir / "videos.json").read_text(encoding="utf-8"))
    by_file = {v["file"]: v for v in data["videos"]}
    assert by_file["electronics/soldering_basics.mp4"]["title"] == "Soldering basics"
    assert by_file["new_clip.mkv"]["topic"] == "radio"
    assert by_file["new_clip.mkv"]["title"] == "New clip"
    assert build_video_manifest.main([str(tmp_path / "nowhere")]) == 1
