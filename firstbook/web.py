"""Flask routes for The First Book sections: home, maps, videos, guides, status.

The model widget routes stay in ``app.py``; this blueprint only adds to them.
Pages are deliberately bare (UI work is pending, see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

from typing import Callable

from flask import Blueprint, Response, abort, current_app, jsonify, render_template, request, send_file

from firstbook import guides, maps, remote_library, videos
from firstbook.library import Library, safe_child
from modelwidget import catalog
from modelwidget.capabilities import detect_capabilities


def create_blueprint(library: Library, model_status: Callable[[dict], dict]) -> Blueprint:
    bp = Blueprint("firstbook", __name__)

    # ---- pages -------------------------------------------------------------
    @bp.get("/home")
    def home():
        return render_template("home.html", status=_status())

    @bp.get("/maps")
    def maps_page():
        return render_template("maps.html", packs=_packs(), catalog=_maps_catalog())

    @bp.get("/videos")
    def videos_page():
        items = videos.index(library.videos_dir)
        q = request.args.get("q", "")
        topic = request.args.get("topic", "")
        results = sorted(
            videos.search(items, q, topic),
            key=lambda v: (v["topic"], v["title"].lower(), v["file"]),
        )
        return render_template(
            "videos.html",
            videos=results,
            topics=videos.topics_present(items),
            topic_names=videos.TOPIC_NAMES,
            q=q,
            topic=topic,
            library=_library_status(),
        )

    @bp.get("/videos/watch/<path:rel>")
    def video_watch(rel: str):
        item = next((v for v in videos.index(library.videos_dir) if v["id"] == rel), None)
        if item is None:
            abort(404)
        return render_template("video_watch.html", video=item)

    @bp.get("/guides")
    def guides_page():
        q = request.args.get("q", "")
        return render_template("guides.html", guides=guides.search(guides.list_guides(library.guides_dir), q), q=q)

    @bp.get("/guides/<slug>")
    def guide_page(slug: str):
        guide = guides.get_guide(library.guides_dir, slug)
        if guide is None:
            abort(404)
        return render_template("guide.html", guide=guide)

    # ---- maps --------------------------------------------------------------
    @bp.get("/api/maps")
    def api_maps():
        cat = _maps_catalog()
        return jsonify({"packs": _packs(cat), "catalog": cat, "maps_dir": str(library.maps_dir)})

    @bp.get("/tiles/<name>/<int:z>/<int:x>/<y>")
    def tile(name: str, z: int, x: int, y: str):
        # Accept "12" or "12.pbf" / "12.png" so either URL style works in a viewer.
        y_num = y.split(".", 1)[0]
        if not y_num.isdigit():
            abort(404)
        path = maps.find_pack(library.maps_dir, name, "mbtiles")
        if path is None:
            return jsonify({"error": f"No MBTiles pack named {name!r}."}), 404
        found = maps.read_tile(path, z, x, int(y_num))
        if found is None:
            # 204 lets map viewers draw an empty tile instead of logging errors.
            return "", 204
        data, headers = found
        return data, 200, headers

    @bp.get("/maps/files/<filename>")
    def map_file(filename: str):
        # Whole-file access with Range support, for PMTiles read by byte range.
        stem, _, suffix = filename.rpartition(".")
        path = maps.find_pack(library.maps_dir, stem, suffix) if suffix in ("pmtiles",) else None
        if path is None:
            abort(404)
        return send_file(path, mimetype="application/octet-stream", conditional=True)

    # ---- videos ------------------------------------------------------------
    @bp.get("/api/videos")
    def api_videos():
        items = videos.index(library.videos_dir)
        results = videos.search(
            items,
            request.args.get("q", ""),
            request.args.get("topic", ""),
            request.args.get("tag", ""),
        )
        return jsonify(
            {
                "videos": results,
                "count": len(results),
                "total": len(items),
                "topics": videos.topics_present(items),
                "suggested_topics": list(videos.TOPICS),
            }
        )

    @bp.get("/videos/file/<path:rel>")
    def video_file(rel: str):
        path = safe_child(library.videos_dir, rel)
        if path is None or path.suffix.lower() not in videos.VIDEO_SUFFIXES:
            abort(404)
        mimetype = videos.MIMETYPES.get(path.suffix.lower(), "application/octet-stream")
        return send_file(path, mimetype=mimetype, conditional=True)

    @bp.get("/videos/subtitles/<path:rel>")
    def video_subtitles(rel: str):
        path = safe_child(library.videos_dir, rel)
        if path is None or not path.is_file() or path.suffix.lower() not in {".srt", ".vtt"}:
            abort(404)
        if path.suffix.lower() == ".vtt":
            return send_file(path, mimetype="text/vtt; charset=utf-8")
        return Response(
            videos.srt_to_vtt(path.read_text(encoding="utf-8")),
            mimetype="text/vtt; charset=utf-8",
        )

    @bp.get("/api/videos/library")
    def api_videos_library():
        # Offline-safe: last check time and whatever is already on disk.
        body = _library_status()
        cached = remote_library.load_cached_manifest(library.videos_dir)
        body["comparison"] = (
            remote_library.compare_manifest(cached, library.videos_dir) if cached else None
        )
        return jsonify(body)

    @bp.post("/api/videos/library/check")
    def api_videos_library_check():
        result = remote_library.check_library(library.videos_dir)
        body = _library_status()
        body.update({
            "ok": result["ok"],
            "offline": result["offline"],
            "error": result["error"],
            "comparison": result["comparison"],
        })
        return jsonify(body), (200 if result["ok"] else 503)

    @bp.post("/api/videos/library/download")
    def api_videos_library_download():
        payload = request.get_json(silent=True) or {}
        manifest = remote_library.load_cached_manifest(library.videos_dir)
        if manifest is None:
            checked = remote_library.check_library(library.videos_dir)
            if not checked["ok"]:
                return jsonify({"ok": False, "error": checked["error"], "job": _library_downloads().snapshot()}), 503
            manifest = checked["manifest"]
        ids = payload.get("ids")
        if isinstance(ids, list):
            ids = [str(i) for i in ids]
        else:
            ids = None
        all_pending = bool(payload.get("all")) or ids is None
        started = _library_downloads().start(manifest, ids, all_pending=all_pending)
        code = 200 if started["ok"] else 409
        return jsonify(started), code

    @bp.get("/api/videos/library/status")
    def api_videos_library_status():
        return jsonify(_library_downloads().snapshot())

    # ---- guides ------------------------------------------------------------
    @bp.get("/api/guides")
    def api_guides():
        results = guides.search(guides.list_guides(library.guides_dir), request.args.get("q", ""))
        return jsonify({"guides": results, "count": len(results)})

    @bp.get("/api/guides/<slug>")
    def api_guide(slug: str):
        guide = guides.get_guide(library.guides_dir, slug)
        if guide is None:
            return jsonify({"error": f"No guide {slug!r}."}), 404
        return jsonify(guide)

    # ---- status ------------------------------------------------------------
    @bp.get("/api/status")
    def api_status():
        return jsonify(_status())

    # ---- helpers -----------------------------------------------------------
    def _maps_catalog() -> dict:
        return maps.load_catalog(library.maps_catalog)

    def _packs(cat: dict | None = None) -> list[dict]:
        return maps.list_packs(library.maps_dir, cat if cat is not None else _maps_catalog())

    def _library_downloads() -> remote_library.LibraryDownloads:
        mgr = current_app.config.get("VIDEO_LIBRARY")
        if mgr is None:
            mgr = remote_library.LibraryDownloads(library.videos_dir)
            current_app.config["VIDEO_LIBRARY"] = mgr
        return mgr

    def _library_status() -> dict:
        return remote_library.public_status(library.videos_dir, _library_downloads())

    def _status() -> dict:
        caps = detect_capabilities(library.weights_dir)
        video_items = videos.index(library.videos_dir)
        guide_items = guides.list_guides(library.guides_dir)
        packs = _packs()
        return {
            "app": "The First Book",
            "offline": True,
            "library": library.describe(),
            "maps": {
                "count": len(packs),
                "packs": [{k: p[k] for k in ("name", "kind", "size_mb", "title")} for p in packs],
            },
            "videos": {
                "count": len(video_items),
                "topics": videos.topics_present(video_items),
                "library": _library_status(),
            },
            "guides": {
                "count": len(guide_items),
                "placeholders": sum(1 for g in guide_items if g["placeholder"]),
            },
            "models": model_status(caps),
            "capabilities": caps,
        }

    return bp


def model_summary(load_catalog: Callable[[], dict], settings, loaded: bool, caps: dict) -> dict:
    """Counts for /api/status, built from the same catalog the picker uses."""
    try:
        data = load_catalog()
    except catalog.SelectionError as exc:
        return {"error": str(exc)}
    rows = [catalog.evaluate(entry, caps, settings.weights_dir) for entry in data["models"]]
    by_category: dict[str, int] = {}
    for row in rows:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
    return {
        "catalog_count": len(rows),
        "compatible": sum(1 for r in rows if r["compatible"]),
        "installed": [r["id"] for r in rows if r["installed"]],
        "planned": [p["id"] for p in data["planned"]],
        "by_category": dict(sorted(by_category.items())),
        "active": {"model_id": settings.model_id, "task": settings.model_task, "loaded": loaded},
    }
