#!/usr/bin/env python3
"""Search Hugging Face for GGUF chat models that fit this machine.

    python scripts/discover_models.py --q "llama 3.2 1b instruct"
    python scripts/discover_models.py --q "qwen2.5 1.5b" --add 1
    python scripts/discover_models.py --remove user-llama-3-2-1b-instruct-q4-k-m

Results are cached next to the catalog, so a previous search still prints
when Hugging Face cannot be reached. --add writes config/models_user.json
(never the curated catalog). Download the pick afterwards with
scripts/download_models.py <id>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelwidget import catalog, discover  # noqa: E402
from modelwidget.capabilities import detect_capabilities  # noqa: E402
from modelwidget.settings import Settings  # noqa: E402


def _age(seconds: int | None) -> str:
    if seconds is None:
        return "unknown age"
    if seconds < 60:
        return "less than a minute ago"
    if seconds < 3600:
        minutes = round(seconds / 60)
        return f"{minutes} minute(s) ago"
    hours = round(seconds / 3600)
    if hours < 48:
        return f"{hours} hour(s) ago"
    return f"{round(hours / 24)} day(s) ago"


def print_results(payload: dict, out=None) -> None:
    out = out or sys.stdout
    query = payload.get("query") or ""
    if payload.get("online"):
        print(f"Search: {query!r} (live from Hugging Face)", file=out)
    else:
        error = payload.get("error") or "Hugging Face could not be reached."
        if payload.get("cached"):
            print(
                f"{error}\nCached search {query!r}, {_age(payload.get('cache_age_seconds'))}.",
                file=out,
            )
        else:
            print(error, file=out)
    results = payload.get("results") or []
    if not results:
        print("No GGUF chat models to show.", file=out)
        return
    print(
        f"{'#':>3}  {'fit':9}  {'quant':8}  {'size':>8}  {'RAM est':>8}  downloads  repo",
        file=out,
    )
    for index, row in enumerate(results, start=1):
        in_cat = f"  [already {row['in_catalog']}]" if row.get("in_catalog") else ""
        print(
            f"{index:3}  {row.get('fit') or '?':9}  {row.get('quant') or '?':8}  "
            f"{row.get('approx_size_mb') or 0:7.0f}M  "
            f"{row.get('estimated_ram_mb') or 0:7.0f}M  "
            f"{row.get('downloads') or 0:9}  {row.get('hf_id')}  {row.get('name')}"
            f"{in_cat}",
            file=out,
        )
        print(f"     {row.get('reason') or ''}  licence {row.get('license') or 'unknown'}"
              f"{'  gated' if row.get('gated') else ''}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", default="", help="Search text (default: a small instruct grab)")
    parser.add_argument("--task", default="chat", help="Currently only 'chat' is used")
    parser.add_argument("--add", type=int, metavar="N", help="Add cached/live result number N")
    parser.add_argument("--remove", metavar="ID", help="Remove a user-added catalog id")
    parser.add_argument("--catalog", type=Path, default=None, help="Curated catalog path")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    caps = detect_capabilities(settings.weights_dir)
    catalog_file = args.catalog
    cache_file = discover.cache_path(catalog_file)

    if args.remove:
        try:
            removed = discover.remove_user_model(args.remove, catalog_file=catalog_file)
        except discover.DiscoveryError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Removed {removed['id']} ({removed.get('name')}). Weights on disk were left alone.")
        return 0

    payload = discover.discover(
        args.q,
        caps,
        task=args.task,
        catalog_file=catalog_file,
        cache_file=cache_file,
    )
    print_results(payload)
    if args.add is None:
        return 0 if payload.get("results") or payload.get("online") else 2

    try:
        candidate = discover.candidate_from_cache(args.add, cache_file, catalog_file)
        entry = discover.add_candidate(candidate, catalog_file)
    except discover.DiscoveryError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"\nAdded {entry['id']} ({entry['name']}) to {discover.user_catalog_path(catalog_file)}."
        f"\nDownload with: python scripts/download_models.py {entry['id']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
