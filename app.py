"""Flask widget that serves one configured Hugging Face model.

Templates and static files are pinned to the project root, so the process
working directory does not matter. The model loads on the first prediction
unless ``LAZY_LOAD=0``.

``/api/models`` lists the offline catalog filtered for this machine, and
``POST /api/models/select`` installs (only if missing), loads and switches to
one of those models, then records it in ``config/.env``.
``GET /api/models/discover`` searches Hugging Face for GGUF chat models that
fit this machine; picks land in ``config/models_user.json``, never the
curated catalog.
"""

from __future__ import annotations

import gc
import io
import os
import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from firstbook.library import Library
from firstbook.web import create_blueprint, model_summary
from modelwidget import catalog, discover
from modelwidget.capabilities import detect_capabilities
from modelwidget.downloads import DownloadManager
from modelwidget.service import ModelService
from modelwidget.settings import Settings
from paths import PROJECT_ROOT


def create_app(
    settings: Settings | None = None,
    service: ModelService | None = None,
    *,
    override_env_path: Path | None = None,
    catalog_file: Path | None = None,
    library: Library | None = None,
    downloads: DownloadManager | None = None,
) -> Flask:
    settings = settings or Settings.from_env()
    service = service or ModelService(settings)
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    app.config["SERVICE"] = service
    app.config["OVERRIDE_ENV_PATH"] = override_env_path or catalog.OVERRIDE_ENV_PATH
    app.config["CATALOG_PATH"] = catalog_file
    select_lock = threading.Lock()
    app.config["DOWNLOADS"] = downloads or DownloadManager(settings.weights_dir)

    def current() -> ModelService:
        # The model picker swaps this object, so routes look it up per request.
        return app.config["SERVICE"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/info")
    def info():
        return jsonify(current().info())

    @app.post("/api/predict")
    def predict():
        service = current()
        try:
            if service.input_kind == "image":
                upload = request.files.get("image")
                if upload is None or not upload.filename:
                    return jsonify({"error": "No image provided."}), 400
                try:
                    image = Image.open(io.BytesIO(upload.read())).convert("RGB")
                except Exception:
                    return jsonify({"error": "Could not read that image."}), 400
                payload = service.predict(image=image, text=None)
            else:
                if request.is_json:
                    data = request.get_json(silent=True) or {}
                    text = data.get("text", "")
                else:
                    text = request.form.get("text", "")
                if not isinstance(text, str):
                    return jsonify({"error": "Text must be a string."}), 400
                payload = service.predict(image=None, text=text)
            return jsonify(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

    @app.get("/api/models")
    def list_models():
        try:
            data = catalog.load_catalog(app.config["CATALOG_PATH"])
        except catalog.SelectionError as exc:
            return jsonify({"error": str(exc)}), exc.status
        active = current().settings
        caps = detect_capabilities(active.weights_dir)
        compatible: list[dict] = []
        excluded: list[dict] = []
        for entry in data["models"]:
            row = catalog.evaluate(entry, caps, active.weights_dir)
            row["active"] = catalog.is_active(entry, active)
            row["download"] = app.config["DOWNLOADS"].get(entry["id"])
            (compatible if row["compatible"] else excluded).append(row)
        active_entry = next((row["id"] for row in compatible + excluded if row["active"]), None)
        return jsonify(
            {
                "catalog": {
                    "version": data["version"],
                    "description": data["description"],
                    "path": data["path"],
                    "invalid": data["invalid"],
                },
                "capabilities": caps,
                "active": {
                    "id": active_entry,
                    "model_id": active.model_id,
                    "revision": active.model_revision,
                    "task": active.model_task,
                    "backend": active.model_backend,
                    "loaded": current().loaded,
                },
                "models": compatible,
                "excluded": excluded,
                # Roadmap entries (e.g. GGUF for llama.cpp). Never installable yet.
                "planned": [dict(item, category=catalog.category_of(item)) for item in data["planned"]],
                "installed": [row["id"] for row in compatible + excluded if row["installed"]],
                "weights_override": bool(os.environ.get("MODEL_WEIGHTS", "").strip()),
            }
        )

    @app.get("/api/models/discover")
    def discover_models():
        """Search Hugging Face for GGUF chat models that fit this machine."""
        query = request.args.get("q", "")
        task = request.args.get("task", "chat") or "chat"
        caps = detect_capabilities(current().settings.weights_dir)
        catalog_file = app.config["CATALOG_PATH"]
        payload = discover.discover(
            query,
            caps,
            task=task,
            catalog_file=catalog_file,
            cache_file=discover.cache_path(catalog_file),
        )
        return jsonify(payload)

    @app.post("/api/models/discover/add")
    def discover_add():
        """Write a discovery pick into the user catalog, not the curated one."""
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "Send JSON describing the model to add."}), 400
        catalog_file = app.config["CATALOG_PATH"]
        try:
            if "index" in data and data.get("hf_id") is None:
                try:
                    index = int(data["index"])
                except (TypeError, ValueError) as exc:
                    raise discover.DiscoveryError("index must be a number.", 400) from exc
                candidate = discover.candidate_from_cache(
                    index, discover.cache_path(catalog_file), catalog_file
                )
            else:
                candidate = data.get("candidate") if isinstance(data.get("candidate"), dict) else data
            entry = discover.add_candidate(candidate, catalog_file)
        except discover.DiscoveryError as exc:
            return jsonify({"error": str(exc)}), exc.status
        return jsonify({"added": entry}), 201

    @app.post("/api/models/discover/remove")
    def discover_remove():
        """Drop a user-added model from the user catalog. Weights stay on disk."""
        data = request.get_json(silent=True) or {}
        model_key = data.get("id") if isinstance(data, dict) else None
        if not isinstance(model_key, str) or not model_key.strip():
            return jsonify({"error": "Send JSON {\"id\": \"<catalog id>\"}."}), 400
        active = current().settings
        catalog_file = app.config["CATALOG_PATH"]
        try:
            data = catalog.load_catalog(catalog_file)
        except catalog.SelectionError as exc:
            return jsonify({"error": str(exc)}), exc.status
        entry = next((item for item in data["models"] if item["id"] == model_key.strip()), None)
        active_id = entry["id"] if entry is not None and catalog.is_active(entry, active) else None
        try:
            removed = discover.remove_user_model(
                model_key.strip(),
                catalog_file=catalog_file,
                active_id=active_id,
            )
        except discover.DiscoveryError as exc:
            return jsonify({"error": str(exc)}), exc.status
        return jsonify({"removed": removed["id"]})

    @app.post("/api/models/select")
    def select_model():

        data = request.get_json(silent=True) or {}
        model_key = data.get("id") if isinstance(data, dict) else None
        if not isinstance(model_key, str) or not model_key.strip():
            return jsonify({"error": "Send JSON {\"id\": \"<catalog id>\"}."}), 400
        if not select_lock.acquire(blocking=False):
            return jsonify({"error": "Another model switch is already running."}), 409
        try:
            return _select(model_key.strip())
        except catalog.SelectionError as exc:
            return jsonify({"error": str(exc)}), exc.status
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Model switch failed: {exc}"}), 500
        finally:
            select_lock.release()

    def _select(model_key: str):
        if os.environ.get("MODEL_WEIGHTS", "").strip():
            raise catalog.SelectionError(
                "MODEL_WEIGHTS is set, so every model would load that same checkpoint. "
                "Unset MODEL_WEIGHTS to use the model picker.",
                409,
            )
        data = catalog.load_catalog(app.config["CATALOG_PATH"])
        entry = next((item for item in data["models"] if item["id"] == model_key), None)
        if entry is None:
            raise catalog.SelectionError(f"No catalog model with id {model_key!r}.", 404)
        if app.config["DOWNLOADS"].busy(entry["id"]):
            raise catalog.SelectionError(
                f"{entry['name']} is still downloading. Use it when the download finishes.", 409
            )
        old = current()
        base = old.settings
        caps = detect_capabilities(base.weights_dir)
        row = catalog.evaluate(entry, caps, base.weights_dir)
        if not row["compatible"]:
            raise catalog.SelectionError(
                f"{entry['name']} cannot run here: {'; '.join(row['reasons'])}.", 409
            )

        snapshot, downloaded = catalog.install(entry, base.weights_dir)
        new_settings = catalog.settings_for(entry, base)
        new_service = ModelService(new_settings)
        try:
            new_service.ensure_loaded()
        except Exception as exc:
            traceback.print_exc()
            raise catalog.SelectionError(
                f"{entry['name']} was installed at {snapshot} but would not load: {exc}", 500
            ) from exc

        values = {key: str(entry.get(field) or "") for key, field in catalog.SETTING_KEYS.items()}
        env_path = catalog.write_override(values, app.config["OVERRIDE_ENV_PATH"])
        # Keep this process consistent with the file for anything that rereads it.
        os.environ.update(values)
        app.config["SERVICE"] = new_service
        del old
        gc.collect()

        payload = {
            "selected": entry["id"],
            "downloaded": downloaded,
            "snapshot": str(snapshot),
            "config_written": str(env_path),
            "info": new_service.info(),
        }
        shadowed = catalog.shell_overrides()
        if shadowed:
            payload["warning"] = (
                f"{', '.join(shadowed)} exported in the shell will override "
                f"{env_path} the next time the app starts."
            )
        return jsonify(payload)

    # The First Book sections (maps, videos, guides, status). Local files only.
    library = library or Library.from_env(weights_dir=settings.weights_dir)
    app.config["LIBRARY"] = library
    app.register_blueprint(
        create_blueprint(
            library,
            lambda caps: model_summary(
                lambda: catalog.load_catalog(app.config["CATALOG_PATH"]),
                current().settings,
                current().loaded,
                caps,
            ),
        )
    )

    # ---- downloads (separate from activating) -------------------------------
    def _catalog_entry(model_key: str) -> dict:
        data = catalog.load_catalog(app.config["CATALOG_PATH"])
        entry = next((item for item in data["models"] if item["id"] == model_key), None)
        if entry is None:
            planned = any(item["id"] == model_key for item in data["planned"])
            raise catalog.SelectionError(
                f"{model_key!r} is planned and cannot be downloaded yet."
                if planned
                else f"No catalog model with id {model_key!r}.",
                409 if planned else 404,
            )
        return entry

    @app.post("/api/models/<model_key>/download")
    def download_model(model_key: str):
        """Start a background download into weights/. Poll the status route."""
        try:
            entry = _catalog_entry(model_key)
        except catalog.SelectionError as exc:
            return jsonify({"error": str(exc)}), exc.status
        weights_dir = current().settings.weights_dir
        row = catalog.evaluate(entry, detect_capabilities(weights_dir), weights_dir)
        if row["installed"]:
            return jsonify({"id": entry["id"], "state": "done", "installed": True,
                            "snapshot": row["snapshot"], "message": "Already on disk."}), 200
        if not row["downloadable"]:
            return jsonify({"error": f"{entry['name']} cannot be downloaded here: "
                            f"{'; '.join(row['reasons'])}."}), 409
        job = app.config["DOWNLOADS"].submit(entry)
        return jsonify(dict(job, warnings=row["warnings"])), 202

    @app.get("/api/models/<model_key>/status")
    def model_status(model_key: str):
        try:
            entry = _catalog_entry(model_key)
        except catalog.SelectionError as exc:
            return jsonify({"error": str(exc)}), exc.status
        job = app.config["DOWNLOADS"].get(entry["id"])
        snapshot = catalog.entry_snapshot(entry, current().settings.weights_dir)
        payload = job or {"id": entry["id"], "state": "idle", "message": "Not downloaded yet."}
        payload["installed"] = snapshot is not None
        payload["snapshot"] = None if snapshot is None else str(snapshot)
        if snapshot is not None and payload["state"] == "idle":
            payload.update(state="done", message="Already on disk.")
        return jsonify(payload)

    @app.get("/api/models/downloads")
    def list_downloads():
        jobs = app.config["DOWNLOADS"].all()
        active = [j for j in jobs if j["state"] in ("queued", "downloading")]
        return jsonify({"jobs": jobs, "active": len(active)})

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify({"error": f"Upload is larger than {settings.max_upload_mb} MB."}), 413

    return app


def main() -> None:
    settings = Settings.from_env()
    application = create_app(settings)
    service: ModelService = application.config["SERVICE"]
    if not settings.lazy_load:
        service.ensure_loaded()
    print(f"Model widget at http://{settings.host}:{settings.port}/", flush=True)
    print(f"The First Book at http://{settings.host}:{settings.port}/home", flush=True)
    application.run(host=settings.host, port=settings.port, threaded=True)


if __name__ == "__main__":
    main()
