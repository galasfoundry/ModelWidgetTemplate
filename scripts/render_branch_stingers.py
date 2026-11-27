#!/usr/bin/env python3
"""Regenerate the short branch theme stingers from config/branch_themes.json.

The synthesis is the original music bed (sine pad + soft bell, no samples)
with each branch's transpose, bar length, waveform and bell settings.
Writes ``static/audio/branches/<id>.ogg`` and ``.mp3``.

    python scripts/render_branch_stingers.py
    python scripts/render_branch_stingers.py --branch fire
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firstbook.branch_audio import STINGER_DIR, render_stingers  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--branch", help="only this branch id")
    parser.add_argument("-o", "--output", type=Path, default=STINGER_DIR, help="folder for the ogg and mp3 files")
    args = parser.parse_args(argv)
    written = render_stingers(args.output, branch_id=args.branch)
    for path in written:
        print(f"{path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
