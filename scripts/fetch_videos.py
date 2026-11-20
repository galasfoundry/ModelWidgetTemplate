#!/usr/bin/env python3
"""Fetch the Creative Commons videos in config/videos_catalog.json for offline use.

Run this while online (it is a setup tool; the app never goes online for
videos). It needs yt-dlp (``pip install yt-dlp``, see requirements-optional.txt)
and works best with ffmpeg on PATH, which joins YouTube's separate video and
audio streams into one MP4.

    python scripts/fetch_videos.py --list            # catalog, sizes on disk
    python scripts/fetch_videos.py                   # fetch everything missing (360p MP4, ~290 MB)
    python scripts/fetch_videos.py --max-height 480  # sharper, ~500 MB
    python scripts/fetch_videos.py --topic hydro
    python scripts/fetch_videos.py 50-dollar-water-turbine-results
    python scripts/fetch_videos.py --dry-run         # check licences, download nothing

Each download re-reads the video's licence on YouTube in the same request and
refuses anything that is no longer Creative Commons Attribution (CC BY). Files land in
the library videos folder (VIDEOS_DIR / LIBRARY_DIR, default library/videos/)
as <topic>/<id>.mp4, and videos.json is rebuilt with each creator's attribution
line, which the watch page shows. Half-finished downloads wait in videos/.downloading/ (hidden from the
page) and resume on the next run. YouTube sometimes refuses with "HTTP 403"
or "confirm you're not a bot"; the script waits and retries, and running it
again later picks up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402
from firstbook import videos  # noqa: E402
from firstbook.library import VIDEOS_CATALOG_PATH, Library  # noqa: E402
from scripts import build_video_manifest  # noqa: E402

# Half-finished streams wait here; the library skips dot-folders, so the
# page never lists a partial file. A rerun resumes from them.
PARTIAL_DIR = ".downloading"
# 360p H.264 is about 0.4-0.5 Mbit/s: roughly 3.5 MB a minute, readable on a
# laptop screen, and the seven starter videos fit in about 290 MB (vs ~500 MB
# at 480p). Pass --max-height 480 for sharper copies.
DEFAULT_MAX_HEIGHT = 360
RETRIES = 4
# yt-dlp match filter: only Creative Commons Attribution, never -NC/-ND/-SA.
LICENSE_FILTER = "license~='(?i)^creative commons attribution' & license!~='(?i)noncommercial|noderiv|sharealike'"  # YouTube sometimes answers 403 or "confirm you're not a bot"; waiting helps


def format_selector(max_height: int, have_ffmpeg: bool) -> str:
    """H.264 + AAC in MP4 plays in every browser and decodes cheaply on old laptops."""
    single = f"b[ext=mp4][height<={max_height}]/w[ext=mp4]"
    if not have_ffmpeg:
        return single
    return (
        f"bv*[height<={max_height}][vcodec^=avc1][ext=mp4]+ba[ext=m4a]"
        f"/bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/{single}"
    )


def find_ytdlp() -> list[str] | None:
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "yt_dlp"]


def _run(cmd: list[str], capture: bool) -> subprocess.CompletedProcess:
    last = None
    for attempt in range(1, RETRIES + 1):
        last = subprocess.run(cmd, capture_output=capture, text=True, check=False)
        if last.returncode == 0:
            return last
        err = (last.stderr or "") if capture else ""
        if attempt < RETRIES:
            wait = 30 * attempt
            hint = " (YouTube rate limit)" if "not a bot" in err else ""
            print(f"  yt-dlp failed{hint}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return last


def check_license(ytdlp: list[str], url: str) -> tuple[bool, str]:
    proc = _run([*ytdlp, "--no-warnings", "--skip-download", "--print", "%(license)s", url], capture=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        return False, f"could not read licence: {tail[0]}"
    text = proc.stdout.strip()
    return videos.is_cc_by(text), text or "no licence shown"


def write_manifest(videos_dir: Path, items: list[dict]) -> dict:
    """Merge catalog metadata for files on disk into videos.json, then rebuild it."""
    path = videos_dir / videos.MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    entries = {e["file"]: e for e in data.get("videos", []) if isinstance(e, dict) and isinstance(e.get("file"), str)}
    for item in items:
        if (videos_dir / item["file"]).is_file():
            entries[item["file"]] = {**entries.get(item["file"], {}), **videos.manifest_entry(item)}
    path.write_text(json.dumps({"version": 1, "videos": list(entries.values())}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    manifest = build_video_manifest.build(videos_dir)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def print_list(items: list[dict], videos_dir: Path, topics: dict) -> None:
    total = 0.0
    for item in items:
        file = videos_dir / item["file"]
        size = file.stat().st_size / 1024**2 if file.is_file() else 0.0
        total += size
        mins = f"{int(item.get('duration_s') or 0) // 60}:{int(item.get('duration_s') or 0) % 60:02d}"
        state = f"{size:6.1f} MB" if size else " missing"
        print(f"{item['id']:<44} {topics.get(item['topic'], item['topic']):<20} {mins:>6} {state}  {item['creator']}")
    print(f"{len(items)} videos, {total:.1f} MB on disk in {videos_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="catalog ids (default: all)")
    parser.add_argument("--topic", help="only this topic, e.g. hydro, batteries, starlink-off-grid")
    parser.add_argument("--list", action="store_true", help="show the catalog and what is on disk")
    parser.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT, help=f"largest video height to fetch (default {DEFAULT_MAX_HEIGHT})")
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    parser.add_argument("--dry-run", action="store_true", help="check licences only; download nothing")
    parser.add_argument("--catalog", type=Path, help="catalog file (default config/videos_catalog.json)")
    args = parser.parse_args(argv)

    catalog = videos.load_catalog(args.catalog or paths.resolve_path("VIDEOS_CATALOG_PATH", VIDEOS_CATALOG_PATH))
    items = catalog["videos"]
    known = {i["id"] for i in items}
    unknown = [i for i in args.ids if i not in known]
    if unknown:
        print(f"Unknown id(s): {', '.join(unknown)}. Try --list.", file=sys.stderr)
        return 2
    if args.ids:
        items = [i for i in items if i["id"] in args.ids]
    if args.topic:
        items = [i for i in items if i["topic"] == args.topic.strip().lower()]
    videos_dir = Library.from_env().videos_dir
    if args.list:
        print_list(items, videos_dir, catalog["topics"])
        return 0

    ytdlp = find_ytdlp()
    if ytdlp is None:
        print("yt-dlp is not installed: pip install yt-dlp", file=sys.stderr)
        return 1
    have_ffmpeg = shutil.which("ffmpeg") is not None
    if not have_ffmpeg:
        print("ffmpeg not found: falling back to single-file MP4 streams (often 360p).", file=sys.stderr)
    videos_dir.mkdir(parents=True, exist_ok=True)
    failed, fetched = [], 0
    for item in items:
        out = videos_dir / item["file"]
        print(f"\n{item['id']}: {item['title']} by {item['creator']}", flush=True)
        if out.is_file() and not args.force:
            print(f"  on disk ({out.stat().st_size / 1024**2:.1f} MB); skipping")
            continue
        if args.dry_run:
            ok, licence = check_license(ytdlp, item["url"])
            print(f"  licence on YouTube: {licence}{'' if ok else '  (not CC BY: would skip)'}", flush=True)
            if not ok:
                failed.append(item["id"])
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        (videos_dir / PARTIAL_DIR).mkdir(parents=True, exist_ok=True)
        licence_file = videos_dir / PARTIAL_DIR / f"{out.stem}.licence.txt"
        licence_file.unlink(missing_ok=True)
        start = time.monotonic()
        # One request per video: yt-dlp reads the page once, records the licence,
        # and its match filter refuses the download unless it is CC BY.
        cmd = [*ytdlp, "--no-warnings", "--no-playlist", "--continue", "--newline", "--progress",
               "--retries", "10", "--fragment-retries", "10", "--http-chunk-size", "10M",
               "--print-to-file", "%(license)s", str(licence_file),
               "--match-filters", LICENSE_FILTER,
               "-f", format_selector(args.max_height, have_ffmpeg), "--merge-output-format", "mp4",
               "--remux-video", "mp4", "-P", f"home:{out.parent}", "-P", f"temp:{videos_dir / PARTIAL_DIR}",
               "-o", f"{out.stem}.%(ext)s", item["url"]]
        # With --force the old copy is kept aside until the new one is complete.
        backup = videos_dir / PARTIAL_DIR / f"{out.name}.old"
        if out.is_file():
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.replace(out, backup)
        proc = _run(cmd, capture=False)
        licence = licence_file.read_text(encoding="utf-8").strip() if licence_file.is_file() else ""
        licence_file.unlink(missing_ok=True)
        if licence:
            print(f"  licence on YouTube: {licence}", flush=True)
        if proc.returncode == 0 and not out.is_file():
            # yt-dlp exits 0 when its match filter rejects the video.
            print("  skipped: the licence on YouTube is not CC BY any more", file=sys.stderr)
        if proc.returncode != 0 or not out.is_file():
            if backup.is_file():
                os.replace(backup, out)
                print("  FAILED; kept the existing copy", file=sys.stderr)
            else:
                print("  FAILED", file=sys.stderr)
            failed.append(item["id"])
            continue
        backup.unlink(missing_ok=True)
        fetched += 1
        for leftover in (videos_dir / PARTIAL_DIR).glob(f"{out.stem}.*"):
            leftover.unlink(missing_ok=True)  # streams from an older, abandoned attempt
        print(f"  saved {out} ({out.stat().st_size / 1024**2:.1f} MB, {time.monotonic() - start:.0f}s)")
        time.sleep(15)  # be gentle with YouTube between videos

    if not args.dry_run:
        manifest = write_manifest(videos_dir, catalog["videos"])
        print(f"\nvideos.json now lists {len(manifest['videos'])} videos; {fetched} fetched this run.")
        print("Open http://127.0.0.1:5000/videos after `python app.py`.")
    if failed:
        print(f"Not fetched: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
