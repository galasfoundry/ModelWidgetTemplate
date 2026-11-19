"""Flask widget that serves one configured Hugging Face model.

Templates and static files are pinned to the project root, so the process
working directory does not matter. The model loads on the first prediction
unless ``LAZY_LOAD=0``.

``/api/models`` lists the offline catalog filtered for this machine, and
``POST /api/models/select`` installs (only if missing), loads and switches to
one of those models, then records it in ``config/.env``.
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
from modelwidget import catalog
from modelwidget.capabilities import detect_capabilities
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

        values = {key: entry[field] for key, field in catalog.SETTING_KEYS.items()}
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
