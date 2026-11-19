"""The First Book must run with no network: no CDN or remote URLs in pages,
and no HTTP client imported by the content modules.

Third-party libraries under static/vendor/ are minified upstream builds whose
source mentions URLs in comments and error messages; they only fetch what our
code points them at. They are checked separately: every vendored file is
loaded from this server and each library ships its licence."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "static" / "vendor"
URL_RE = re.compile(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[/\"' ]", re.IGNORECASE)


def _web_files():
    for folder in ("templates", "static"):
        for path in (ROOT / folder).rglob("*"):
            if VENDOR in path.parents:
                continue
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


def test_vendored_libraries_are_local_and_licensed():
    expected = {
        "maplibre-gl": ("maplibre-gl.js", "maplibre-gl.css", "LICENSE.txt"),
        "pmtiles": ("pmtiles.js", "LICENSE"),
        "protomaps-basemaps": ("basemaps.js", "LICENSE.md"),
        "basemaps-assets": ("fonts/OFL.txt", "fonts/Noto Sans Regular/0-255.pbf",
                            "sprites/v4/light.json", "sprites/v4/light.png",
                            "sprites/LICENSE-tangrams-icons.md"),
    }
    for folder, files in expected.items():
        for name in files:
            assert (VENDOR / folder / name).is_file(), f"missing static/vendor/{folder}/{name}"


def test_templates_load_scripts_and_styles_only_from_this_server():
    tag = re.compile(r"<(?:script|link)\b[^>]*(?:src|href)=\"([^\"]+)\"", re.IGNORECASE)
    for path in (ROOT / "templates").rglob("*.html"):
        for ref in tag.findall(path.read_text(encoding="utf-8")):
            assert ref.startswith("{{ url_for("), f"{path.name} loads {ref} from outside static/"
