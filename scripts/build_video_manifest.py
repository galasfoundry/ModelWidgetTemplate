#!/usr/bin/env python3
"""Scan a videos folder and write a local or remote video manifest.

Usage:
    python scripts/build_video_manifest.py [FOLDER] [--topic TOPIC] [--dry-run]
    python scripts/build_video_manifest.py --remote [FOLDER] --base-url URL [--url-mode path|flat] [-o FILE]

FOLDER defaults to the library videos folder (VIDEOS_DIR / LIBRARY_DIR).

Without --remote this refreshes the local videos.json used by the app.
Existing entries keep their title, tags and notes. New files get a title
from the filename and ``--topic`` (default "other"). Entries for files that
have gone are dropped. ``duration_s`` is filled with ffprobe when it is on
PATH; without it the field stays null. Nothing touches the network.

With --remote it builds a versioned firstbook.library.v1 manifest for
publishing: sha256 and size of each file, plus metadata from sidecars,
videos.json and config/videos_catalog.json. File URLs are based on
--base-url. --url-mode flat is for hosts such as GitHub Releases that
cannot have folders; path keeps topic/name.mp4. Publishing a new episode
is then: upload the files and this manifest. No code change.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firstbook import remote_library, videos  # noqa: E402
from firstbook.library import Library  # noqa: E402


def probe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout.strip()
        return round(float(out), 1)
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def build(folder: Path, topic: str = "other") -> dict:
    existing = videos.load_manifest(folder)
    entries = []
    for file in videos.scan_files(folder):
        rel = file.relative_to(folder).as_posix()
        entry = dict(existing.get(rel) or {})
        entry.setdefault("file", rel)
        entry.setdefault("title", videos.title_from_filename(file.name))
        entry.setdefault("topic", topic)
        entry.setdefault("tags", [])
        entry.setdefault("description", "")
        entry.setdefault("source", "")
        entry.setdefault("license", "")
        if entry.get("duration_s") is None:
            entry["duration_s"] = probe_duration(file)
        entries.append(entry)
    return {"version": 1, "videos": entries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", type=Path)
    parser.add_argument("--topic", default="other", help=f"topic for new files, e.g. {', '.join(videos.TOPICS[:5])}")
    parser.add_argument("--dry-run", action="store_true", help="print the manifest instead of writing it")
    parser.add_argument("--remote", action="store_true", help="build a firstbook.library.v1 manifest for publishing")
    parser.add_argument(
        "--base-url",
        default=remote_library.PLACEHOLDER_LIBRARY_URL,
        help="base URL prepended to file paths (or filenames with --url-mode flat)",
    )
    parser.add_argument(
        "--url-mode",
        choices=("path", "flat"),
        default="path",
        help="path keeps folder/file.mp4 in the URL; flat uses the filename only (GitHub Releases)",
    )
    parser.add_argument("-o", "--output", type=Path, help="with --remote, write here (default: stdout)")
    args = parser.parse_args(argv)
    folder = (args.folder or Library.from_env().videos_dir).resolve()
    if not folder.is_dir():
        print(f"No such folder: {folder}", file=sys.stderr)
        return 1
    if args.remote:
        manifest = remote_library.build_remote_manifest(folder, base_url=args.base_url, url_mode=args.url_mode)
        text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        if args.dry_run or args.output is None:
            print(text, end="")
        if args.output is not None and not args.dry_run:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(f"Wrote {len(manifest['videos'])} remote entries to {args.output}", file=sys.stderr)
        return 0
    manifest = build(folder, args.topic.strip().lower())
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.dry_run:
        print(text, end="")
    else:
        (folder / videos.MANIFEST_NAME).write_text(text, encoding="utf-8")
        print(f"Wrote {len(manifest['videos'])} entries to {folder / videos.MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
