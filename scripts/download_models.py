#!/usr/bin/env python3
"""Download catalog models into weights/ from the command line.

The easiest way to fill a laptop while it is online; the web app then uses
them with no internet. Downloads resume if interrupted (just run it again),
and models already on disk are skipped without touching the network.

    python scripts/download_models.py --list                 # what's in the catalog, what's on disk
    python scripts/download_models.py mobilenetv3-small gemma-3-270m-it-gguf
    python scripts/download_models.py --all                  # everything that fits this machine
    python scripts/download_models.py --all --dry-run        # sizes only, nothing downloaded

GGUF chat models download one quantised file; using them in the app needs
``pip install llama-cpp-python``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelwidget import catalog, downloads  # noqa: E402
from modelwidget.capabilities import detect_capabilities  # noqa: E402
from modelwidget.settings import Settings  # noqa: E402

_MB = 1024**2


def _rows(weights_dir: Path, catalog_path: Path | None) -> tuple[list[dict], list[dict]]:
    data = catalog.load_catalog(catalog_path)
    caps = detect_capabilities(weights_dir)
    rows = [catalog.evaluate(entry, caps, weights_dir) for entry in data["models"]]
    for row in rows:
        row["disk_mb"] = downloads.disk_size_mb(row["snapshot"]) if row["installed"] else None
    return rows, data["planned"]


def print_table(rows: list[dict], planned: list[dict], weights_dir: Path, out=None) -> None:
    out = out or sys.stdout
    print(f"Weights folder: {weights_dir}", file=out)
    print(f"{'id':32} {'category':8} {'size':>9} {'on disk':>9}  status", file=out)
    for row in rows:
        size = f"~{row['approx_size_mb']:g} MB" if row.get("approx_size_mb") else "?"
        disk = f"{row['disk_mb']} MB" if row["installed"] else "-"
        if not row["downloadable"]:
            note = "cannot download here: " + "; ".join(row["reasons"])
        elif row["installed"]:
            note = "installed" + ("" if row["compatible"] else " (to use: " + "; ".join(row["reasons"]) + ")")
        else:
            note = "not downloaded"
        if row["warnings"]:
            note += " [warning: " + "; ".join(row["warnings"]) + "]"
        print(f"{row['id']:32} {row['category']:8} {size:>9} {disk:>9}  {note}", file=out)
    for entry in planned:
        print(f"{entry['id']:32} {catalog.category_of(entry):8} {'':>9} {'-':>9}  planned (not downloadable yet)", file=out)
    used = sum(r["disk_mb"] or 0 for r in rows)
    print(f"Total on disk: {used:.1f} MB", file=out)


class _Bar:
    """A one-line progress counter on stderr."""

    def __init__(self, name: str):
        self.name = name
        self.total = None
        self.done = 0
        self.last = 0.0

    def plan(self, files):
        self.total = sum(s for _, s in files if s) or None
        size = f"{self.total / _MB:.1f} MB" if self.total else "unknown size"
        print(f"  {len(files)} file(s), {size}", file=sys.stderr)

    def bytes(self, so_far):
        self.done = so_far
        now = time.time()
        if now - self.last >= 0.5:
            self.last = now
            pct = f" {100 * self.done / self.total:5.1f}%" if self.total else ""
            print(f"\r  {self.done / _MB:8.1f} MB{pct}", end="", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="catalog ids to download")
    parser.add_argument("--all", action="store_true", help="every downloadable model that fits this machine")
    parser.add_argument("--list", action="store_true", help="show the catalog and what is on disk")
    parser.add_argument("--dry-run", action="store_true", help="show what would download, fetch nothing")
    parser.add_argument("--weights-dir", type=Path, help="defaults to WEIGHTS_DIR or weights/")
    parser.add_argument("--catalog", type=Path, help="defaults to config/models_catalog.json")
    args = parser.parse_args(argv)

    weights_dir = (args.weights_dir or Settings.from_env().weights_dir).resolve()
    try:
        rows, planned = _rows(weights_dir, args.catalog)
    except catalog.SelectionError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.list or not (args.all or args.ids):
        print_table(rows, planned, weights_dir)
        if not args.list:
            print("\nPass model ids or --all to download.", file=sys.stderr)
        return 0

    by_id = {row["id"]: row for row in rows}
    planned_ids = {p["id"] for p in planned}
    if args.all:
        chosen = [row for row in rows if row["downloadable"]]
    else:
        unknown = [i for i in args.ids if i not in by_id]
        for model_id in unknown:
            why = "is planned, not downloadable yet" if model_id in planned_ids else "is not in the catalog"
            print(f"{model_id} {why}. Try --list.", file=sys.stderr)
        if unknown:
            return 2
        chosen = [by_id[i] for i in args.ids]

    failures = 0
    for row in chosen:
        label = f"{row['id']} ({row['hf_id']}{', ' + row['gguf_file'] if row.get('gguf_file') else ''})"
        if row["installed"]:
            print(f"{label}: already on disk, {row['disk_mb']} MB at {row['snapshot']}")
            continue
        if not row["downloadable"]:
            print(f"{label}: skipped, {'; '.join(row['reasons'])}", file=sys.stderr)
            failures += 1
            continue
        for warning in row["warnings"]:
            print(f"{label}: warning: {warning}", file=sys.stderr)
        if args.dry_run:
            print(f"{label}: would download ~{row.get('approx_size_mb', '?')} MB into {weights_dir}")
            continue
        print(f"{label}: downloading into {weights_dir}", file=sys.stderr)
        bar = _Bar(row["id"])
        started = time.time()
        try:
            snapshot, _ = downloads.download_entry(row, weights_dir, on_plan=bar.plan, on_bytes=bar.bytes)
        except downloads.DownloadError as exc:
            print(f"\n{label}: FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue
        took = time.time() - started
        print(f"\r{label}: done, {downloads.disk_size_mb(snapshot)} MB in {took:.1f} s at {snapshot}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
