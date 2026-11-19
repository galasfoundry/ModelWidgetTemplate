"""Guides: listing, keyword search, pages."""

from __future__ import annotations

from app import create_app
from firstbook import guides
from firstbook.library import DEFAULT_LIBRARY_DIR
from tests.firstbook_helpers import image_settings, make_library


def _lib(tmp_path):
    lib = make_library(tmp_path)
    (lib.guides_dir / "radio.md").write_text("# Radio basics\n\nAntenna antenna radio.\n", encoding="utf-8")
    (lib.guides_dir / "notes.txt").write_text("Plain notes\n\nSomething about an antenna.\n", encoding="utf-8")
    (lib.guides_dir / "README.md").write_text("# skipped\n", encoding="utf-8")
    (lib.guides_dir / "bad name.md").write_text("# skipped\n", encoding="utf-8")
    return lib


def test_list_and_search(tmp_path):
    lib = _lib(tmp_path)
    listed = guides.list_guides(lib.guides_dir)
    assert [g["slug"] for g in listed] == ["notes", "radio"]
    assert listed[1]["title"] == "Radio basics"
    hits = guides.search(listed, "antenna")
    assert [h["slug"] for h in hits] == ["radio", "notes"]
    assert "antenna" in hits[0]["snippet"].lower()
    assert guides.search(listed, "antenna radio")[0]["slug"] == "radio"
    assert guides.search(listed, "nothing-here") == []
    assert len(guides.search(listed, "")) == 2


def test_shipped_guides_are_marked_placeholders():
    shipped = guides.list_guides(DEFAULT_LIBRARY_DIR / "guides")
    assert shipped, "expected placeholder guides in library/guides"
    assert all(g["placeholder"] for g in shipped)


def test_guide_routes(tmp_path):
    client = create_app(image_settings(tmp_path), library=_lib(tmp_path)).test_client()
    assert client.get("/api/guides?q=antenna").get_json()["count"] == 2
    assert client.get("/api/guides/radio").get_json()["title"] == "Radio basics"
    assert client.get("/api/guides/missing").status_code == 404
    assert client.get("/api/guides/..%2Fsecret").status_code == 404
    assert "Radio basics" in client.get("/guides").get_data(as_text=True)
    assert "Antenna" in client.get("/guides/radio").get_data(as_text=True)
    assert client.get("/guides/missing").status_code == 404
