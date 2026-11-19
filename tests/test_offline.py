"""The First Book must run with no network: no CDN or remote URLs in pages,
and no HTTP client imported by the content modules."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL_RE = re.compile(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[/\"' ]", re.IGNORECASE)


def _web_files():
    for folder in ("templates", "static"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix.lower() in {".html", ".js", ".css", ".json", ".svg"}:
                yield path


def test_templates_and_static_have_no_remote_urls():
    offenders = []
    for path in _web_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in URL_RE.finditer(text):
            offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
    assert not offenders, "Remote URLs break offline use:\n" + "\n".join(offenders)


def test_content_modules_do_not_import_http_clients():
    banned = re.compile(r"^\s*(?:import|from)\s+(requests|httpx|urllib\.request|http\.client|aiohttp|socket)\b", re.M)
    for path in (ROOT / "firstbook").rglob("*.py"):
        assert not banned.search(path.read_text(encoding="utf-8")), path
