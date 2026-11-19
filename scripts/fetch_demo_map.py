#!/usr/bin/env python3
"""Fetch UK map packs for The First Book from the Protomaps daily build.

Run this ONCE while online (it is a setup tool; the app itself never goes
online for maps). It downloads the go-pmtiles CLI into tools/ if it is not on
PATH, finds the newest Protomaps planet build, and cuts small extracts by
HTTP range requests, so the 100+ GB planet file is never downloaded.

    python scripts/fetch_demo_map.py                 # demo: uk-overview (~7 MB) + bristol (~23 MB)
    python scripts/fetch_demo_map.py --uk 12         # + whole UK to zoom 12 (~350 MB)
    python scripts/fetch_demo_map.py --uk 14         # + whole UK to zoom 14 (~1.5 GB)
    python scripts/fetch_demo_map.py --only uk --uk 10
    python scripts/fetch_demo_map.py --name leeds --bbox=-1.65,53.75,-1.45,53.85 --maxzoom 15
    python scripts/fetch_demo_map.py --dry-run       # print sizes, write nothing

Sizes are from the 18 Nov 2026 build. Files land in the library maps folder
(MAPS_DIR / LIBRARY_DIR, default library/maps/). Map data © OpenStreetMap
contributors (ODbL); show the attribution, which the /maps viewer does.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402
from firstbook.library import Library  # noqa: E402

PMTILES_VERSION = "1.31.2"
RELEASE_URL = "https://github.com/protomaps/go-pmtiles/releases/download/v{v}/{asset}"
BUILDS_URL = "https://build-metadata.protomaps.dev/builds.json"
BUILD_URL = "https://build.protomaps.com/{key}"
UK_BBOX = "-8.65,49.86,1.77,60.86"

# name -> (bbox, maxzoom). Bristol & Bath is the demo city; change freely.
DEMO_PACKS = {
    "uk-overview": (UK_BBOX, 8),
    "bristol": ("-2.75,51.38,-2.45,51.56", 15),
}


def _get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "the-first-book-setup/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _asset_name() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    if system == "Linux":
        return f"go-pmtiles_{PMTILES_VERSION}_Linux_{arch}.tar.gz", "pmtiles"
    if system == "Darwin":
        return f"go-pmtiles-{PMTILES_VERSION}_Darwin_{arch}.zip", "pmtiles"
    if system == "Windows":
        return f"go-pmtiles_{PMTILES_VERSION}_Windows_{arch}.zip", "pmtiles.exe"
    raise SystemExit(f"No go-pmtiles build for {system}; install it from https://github.com/protomaps/go-pmtiles")


def ensure_cli() -> str:
    found = shutil.which("pmtiles")
    if found:
        return found
    asset, exe = _asset_name()
    target = paths.PROJECT_ROOT / "tools" / exe
    if target.is_file():
        return str(target)
    url = RELEASE_URL.format(v=PMTILES_VERSION, asset=asset)
    print(f"Downloading the pmtiles CLI: {url}")
    data = _get(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            member = next(n for n in zf.namelist() if Path(n).name == exe)
            target.write_bytes(zf.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            member = next(m for m in tf.getmembers() if Path(m.name).name == exe)
            target.write_bytes(tf.extractfile(member).read())
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(target)


def latest_build() -> str:
    builds = json.loads(_get(BUILDS_URL, timeout=60))
    key = sorted(b["key"] for b in builds if str(b.get("key", "")).endswith(".pmtiles"))[-1]
    return BUILD_URL.format(key=key)


def extract(cli: str, source: str, out: Path, bbox: str, maxzoom: int, dry_run: bool) -> int:
    cmd = [cli, "extract", source, str(out), f"--bbox={bbox}", f"--maxzoom={maxzoom}"]
    if dry_run:
        cmd.append("--dry-run")
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uk", type=int, metavar="MAXZOOM", help="also extract the whole UK to this zoom (10-15)")
    parser.add_argument("--only", choices=["demo", "uk", "custom"], help="skip the other packs")
    parser.add_argument("--name", help="custom pack name (needs --bbox)")
    parser.add_argument("--bbox", help="custom west,south,east,north")
    parser.add_argument("--maxzoom", type=int, default=15, help="custom pack max zoom (default 15)")
    parser.add_argument("--build", help="a specific build URL or file instead of the newest daily build")
    parser.add_argument("--force", action="store_true", help="overwrite packs that already exist")
    parser.add_argument("--dry-run", action="store_true", help="report sizes only")
    args = parser.parse_args(argv)

    jobs: dict[str, tuple[str, int]] = {}
    if args.only in (None, "demo"):
        jobs.update(DEMO_PACKS)
    if args.uk is not None and args.only in (None, "uk"):
        jobs[f"uk-z{args.uk}"] = (UK_BBOX, args.uk)
    if args.name or args.bbox:
        if not (args.name and args.bbox):
            parser.error("--name and --bbox go together")
        jobs[args.name] = (args.bbox, args.maxzoom)
    if not jobs:
        parser.error("nothing to fetch")

    maps_dir = Library.from_env().maps_dir
    maps_dir.mkdir(parents=True, exist_ok=True)
    cli = ensure_cli()
    source = args.build or latest_build()
    print(f"Source build: {source}\nMaps folder: {maps_dir}")
    failed = 0
    for name, (bbox, maxzoom) in jobs.items():
        out = maps_dir / f"{name}.pmtiles"
        if out.exists() and not args.force and not args.dry_run:
            print(f"{out.name} already exists; skipping (use --force to replace)")
            continue
        tmp = out.with_suffix(".pmtiles.part")
        code = extract(cli, source, tmp, bbox, maxzoom, args.dry_run)
        if code != 0:
            failed += 1
            tmp.unlink(missing_ok=True)
        elif not args.dry_run:
            os.replace(tmp, out)
            print(f"Wrote {out} ({out.stat().st_size / 1024**2:.1f} MB)")
    if not args.dry_run:
        print("Open http://127.0.0.1:5000/maps after `python app.py`.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
