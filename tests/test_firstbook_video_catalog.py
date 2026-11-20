"""Creative Commons video catalog: parsing, licence filter, manifest credits. No network."""

from __future__ import annotations

import json
from pathlib import Path

from app import create_app
from firstbook import videos
from firstbook.library import VIDEOS_CATALOG_PATH
from scripts import fetch_videos
from tests.firstbook_helpers import image_settings, make_library

REAL_CATALOG = videos.load_catalog(VIDEOS_CATALOG_PATH)


def _item(**over):
    item = {
        "id": "demo", "url": "https://www.youtube.com/watch?v=abc", "file": "hydro/demo.mp4",
        "title": "Demo turbine", "creator": "Someone", "creator_url": "https://www.youtube.com/@someone",
        "license": "CC BY 3.0", "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "attribution": "“Demo turbine” by Someone, CC BY 3.0", "topic": "Hydro", "tags": ["hydro"],
        "duration_s": 61, "description": "A demo.",
    }
    item.update(over)
    return item


def test_shipped_catalog_is_the_seven_cc_by_videos():
    items = REAL_CATALOG["videos"]
    assert len(items) == 7
    raw = json.loads(Path(VIDEOS_CATALOG_PATH).read_text(encoding="utf-8"))
    assert len(raw["videos"]) == 7, "every shipped entry must pass the catalog filter"
    by_topic = {}
    for item in items:
        by_topic.setdefault(item["topic"], []).append(item["id"])
        assert item["license"] == "CC BY 3.0" and videos.is_cc_by(item["license"])
        assert item["url"].startswith("https://www.youtube.com/watch?v=")
        assert item["file"] == f"{item['topic']}/{item['id']}.mp4"
        assert item["creator"] in item["attribution"] and item["url"] in item["attribution"]
        assert "CC BY 3.0" in item["attribution"] and item["creator_url"].startswith("https://")
        assert isinstance(item["duration_s"], int) and item["duration_s"] > 0
    assert {t: len(ids) for t, ids in by_topic.items()} == {"hydro": 3, "batteries": 3, "starlink-off-grid": 1}
    assert set(REAL_CATALOG["topics"]) == set(by_topic)
    assert len({i["id"] for i in items}) == 7 and len({i["url"] for i in items}) == 7


def test_catalog_drops_non_cc_by_and_unsafe_entries(tmp_path):
    bad = [
        _item(id="std", license="Standard YouTube Licence"),
        _item(id="nc", license="CC BY-NC 4.0"),
        _item(id="sa", license="CC BY-SA 4.0"),
        _item(id="escape", file="../../etc/x.mp4"),
        _item(id="abs", file="/tmp/x.mp4"),
        _item(id="txt", file="hydro/x.txt"),
        _item(id="noattr", attribution=""),
        "not a dict",
    ]
    path = tmp_path / "cat.json"
    path.write_text(json.dumps({"version": 1, "videos": [_item(), *bad]}), encoding="utf-8")
    items = videos.load_catalog(path)["videos"]
    assert [i["id"] for i in items] == ["demo"] and items[0]["topic"] == "hydro"
    assert videos.load_catalog(tmp_path / "missing.json")["videos"] == []
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert videos.load_catalog(tmp_path / "broken.json")["videos"] == []


def test_youtube_licence_wording():
    assert videos.is_cc_by("Creative Commons Attribution license (reuse allowed)")
    assert not videos.is_cc_by("")
    assert not videos.is_cc_by("NA")
    assert not videos.is_cc_by("Creative Commons Attribution-NonCommercial")


def test_manifest_carries_credits_and_watch_page_shows_them(tmp_path):
    lib = make_library(tmp_path)
    item = videos.load_catalog(VIDEOS_CATALOG_PATH)["videos"][0]
    target = lib.videos_dir / item["file"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\x00" * 2048)
    (lib.videos_dir / "other.mp4").write_bytes(b"x")  # a user file stays untouched
    manifest = fetch_videos.write_manifest(lib.videos_dir, REAL_CATALOG["videos"])
    files = [e["file"] for e in manifest["videos"]]
    assert sorted(files) == sorted([item["file"], "other.mp4"]), "only files on disk are listed"
    entry = next(e for e in manifest["videos"] if e["file"] == item["file"])
    assert entry["attribution"] == item["attribution"] and entry["source"] == item["url"]
    assert entry["creator"] == item["creator"] and entry["topic"] == "hydro"

    indexed = {v["id"]: v for v in videos.index(lib.videos_dir)}[item["file"]]
    assert indexed["creator"] == item["creator"] and indexed["license_url"].startswith("https://creativecommons.org/")

    client = create_app(image_settings(tmp_path), library=lib).test_client()
    page = client.get("/videos").get_data(as_text=True)
    assert "DIY off-grid hydro" in page and f"by {item['creator']}" in page
    watch = client.get(f"/videos/watch/{item['file']}").get_data(as_text=True)
    assert "Credit:" in watch and item["creator_url"] in watch and "creativecommons.org/licenses/by/3.0" in watch
    assert videos.search(videos.index(lib.videos_dir), item["creator"].lower().split()[0])

    # Re-running keeps user edits to other fields and does not duplicate entries.
    data = json.loads((lib.videos_dir / "videos.json").read_text(encoding="utf-8"))
    for e in data["videos"]:
        if e["file"] == item["file"]:
            e["notes"] = "mine"
    (lib.videos_dir / "videos.json").write_text(json.dumps(data), encoding="utf-8")
    again = fetch_videos.write_manifest(lib.videos_dir, REAL_CATALOG["videos"])
    assert len(again["videos"]) == 2
    assert next(e for e in again["videos"] if e["file"] == item["file"])["notes"] == "mine"


def test_format_selector_is_modest_mp4():
    sel = fetch_videos.format_selector(480, True)
    assert "height<=480" in sel and "avc1" in sel and "m4a" in sel
    assert "+" not in fetch_videos.format_selector(360, False)


def test_cli_list_and_unknown_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VIDEOS_DIR", str(tmp_path / "v"))
    assert fetch_videos.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "7 videos" in out and "Starlink off-grid" in out and "missing" in out
    assert fetch_videos.main(["nope"]) == 2


def test_cli_skips_a_video_whose_licence_changed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VIDEOS_DIR", str(tmp_path / "v"))
    monkeypatch.setattr(fetch_videos, "find_ytdlp", lambda: ["yt-dlp"])
    seen = []

    def fake_ytdlp(cmd, capture):  # the match filter rejects it: exit 0, no file
        seen.append(cmd)
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(fetch_videos, "_run", fake_ytdlp)
    assert fetch_videos.main(["pump-as-turbine-micro-hydro"]) == 1
    assert not (tmp_path / "v" / "hydro" / "pump-as-turbine-micro-hydro.mp4").exists()
    cmd = seen[0]
    assert cmd[cmd.index("--match-filters") + 1] == fetch_videos.LICENSE_FILTER
    assert "creative commons attribution" in fetch_videos.LICENSE_FILTER
    assert "skipped: the licence" in capsys.readouterr().err


def test_dry_run_reports_licence_without_downloading(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VIDEOS_DIR", str(tmp_path / "v"))
    monkeypatch.setattr(fetch_videos, "find_ytdlp", lambda: ["yt-dlp"])
    monkeypatch.setattr(fetch_videos, "check_license", lambda y, url: (False, "Standard YouTube License"))
    monkeypatch.setattr(fetch_videos, "_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")))
    assert fetch_videos.main(["--dry-run", "pump-as-turbine-micro-hydro"]) == 1
    assert "would skip" in capsys.readouterr().out


def test_cli_force_keeps_old_copy_when_download_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VIDEOS_DIR", str(tmp_path / "v"))
    old = tmp_path / "v" / "hydro" / "pump-as-turbine-micro-hydro.mp4"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old copy")
    monkeypatch.setattr(fetch_videos, "find_ytdlp", lambda: ["yt-dlp"])
    monkeypatch.setattr(fetch_videos, "check_license", lambda y, url: (True, "Creative Commons Attribution license"))
    monkeypatch.setattr(fetch_videos, "_run", lambda cmd, capture: type("P", (), {"returncode": 1})())
    assert fetch_videos.main(["--force", "pump-as-turbine-micro-hydro"]) == 1
    assert old.read_bytes() == b"old copy"
    assert "kept the existing copy" in capsys.readouterr().err

def test_original_episode_sidecar_credits_and_subtitles(tmp_path):
    """Shipped sidecar + srt describe episode 001; the mp4 is supplied at setup."""
    from paths import PROJECT_ROOT

    sidecar_path = PROJECT_ROOT / "library" / "videos" / "originals" / "001-head-and-flow.mp4.json"
    srt_path = PROJECT_ROOT / "library" / "videos" / "originals" / "001-head-and-flow.srt"
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert meta["creator"] == "The First Book / galasfoundry"
    assert meta["license"] == "CC BY 4.0" and videos.is_cc_by(meta["license"])
    assert meta["topic"] == "power" and meta["file"] == "originals/001-head-and-flow.mp4"
    assert "Murder" in meta["fact_check"]
    assert "as long as it rains" in srt_path.read_text(encoding="utf-8")

    lib = make_library(tmp_path)
    dest = lib.videos_dir / "originals"
    dest.mkdir()
    (dest / "001-head-and-flow.mp4").write_bytes(b"\x00" * 2048)
    (dest / "001-head-and-flow.mp4.json").write_text(sidecar_path.read_text(encoding="utf-8"), encoding="utf-8")
    (dest / "001-head-and-flow.srt").write_text(srt_path.read_text(encoding="utf-8"), encoding="utf-8")

    items = {v["id"]: v for v in videos.index(lib.videos_dir)}
    assert "originals/001-head-and-flow.mp4" in items
    item = items["originals/001-head-and-flow.mp4"]
    assert item["title"].startswith("Head and flow")
    assert item["creator"] == "The First Book / galasfoundry"
    assert item["subtitles"] == "originals/001-head-and-flow.srt"
    assert item["subtitles_url"] == "/videos/subtitles/originals/001-head-and-flow.srt"

    client = create_app(image_settings(tmp_path), library=lib).test_client()
    page = client.get("/videos").get_data(as_text=True)
    assert "Head and flow" in page and "The First Book / galasfoundry" in page
    assert "Power" in page and "CC BY 4.0" in page
    watch = client.get("/videos/watch/originals/001-head-and-flow.mp4").get_data(as_text=True)
    assert "Credit:" in watch and "creativecommons.org/licenses/by/4.0" in watch
    assert "Fact-checked by Murder" in watch
    assert "Stored at reduced resolution" not in watch
    assert 'kind="subtitles"' in watch and item["subtitles_url"] in watch
    sub = client.get(item["subtitles_url"])
    assert sub.status_code == 200 and sub.mimetype == "text/vtt"
    body = sub.get_data(as_text=True)
    assert body.startswith("WEBVTT") and "as long as it rains" in body
    assert client.get("/videos/subtitles/../README.md").status_code == 404
    assert client.get("/videos/subtitles/originals/nope.srt").status_code == 404


def test_srt_to_vtt_rewrites_timestamps():
    vtt = videos.srt_to_vtt("1\n00:00:00,500 --> 00:00:03,784\nHello\n")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.500 --> 00:00:03.784" in vtt
    already = "WEBVTT\n\n00:00:00.500 --> 00:00:01.000\nHi\n"
    assert videos.srt_to_vtt(already).startswith("WEBVTT")
