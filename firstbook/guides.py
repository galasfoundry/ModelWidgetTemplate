"""Offline guides: plain Markdown or text files in the guides folder.

The title is the first ``# heading`` (or the first non-empty line). Search is
a stdlib keyword match that ranks by how often the words appear, with title
hits weighted higher. Pages show the raw text; a Markdown renderer can be
vendored later without changing this module.
"""

from __future__ import annotations

import re
from pathlib import Path

GUIDE_SUFFIXES = (".md", ".txt")
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_WORD_RE = re.compile(r"[a-z0-9']+")


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.lstrip("#").strip() or fallback
    return fallback


def list_guides(guides_dir: Path) -> list[dict]:
    root = Path(guides_dir)
    if not root.is_dir():
        return []
    out = []
    for file in sorted(root.iterdir()):
        if not file.is_file() or file.suffix.lower() not in GUIDE_SUFFIXES:
            continue
        if not _SLUG_RE.match(file.stem) or file.name.lower() == "readme.md":
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out.append(
            {
                "slug": file.stem,
                "file": file.name,
                "title": _title(text, file.stem),
                "placeholder": "placeholder" in text[:400].lower(),
                "words": len(_WORD_RE.findall(text.lower())),
                "text": text,
            }
        )
    return out


def get_guide(guides_dir: Path, slug: str) -> dict | None:
    if not _SLUG_RE.match(slug or ""):
        return None
    return next((g for g in list_guides(guides_dir) if g["slug"] == slug), None)


def search(guides: list[dict], query: str) -> list[dict]:
    """Guides containing every query word, best first, each with a short snippet."""
    words = _WORD_RE.findall(query.lower())
    if not words:
        return [_summary(g) for g in guides]
    scored = []
    for guide in guides:
        body = guide["text"].lower()
        title = guide["title"].lower()
        if not all(w in body or w in title for w in words):
            continue
        score = sum(body.count(w) + 5 * title.count(w) for w in words)
        scored.append((score, guide))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["slug"]))
    return [dict(_summary(g), score=s, snippet=_snippet(g["text"], words[0])) for s, g in scored]


def _summary(guide: dict) -> dict:
    return {k: guide[k] for k in ("slug", "file", "title", "placeholder", "words")}


def _snippet(text: str, word: str, width: int = 80) -> str:
    flat = " ".join(text.split())
    at = flat.lower().find(word)
    if at < 0:
        return flat[: width * 2]
    start = max(0, at - width)
    return ("…" if start else "") + flat[start : at + width] + ("…" if at + width < len(flat) else "")
