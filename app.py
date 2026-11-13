"""Flask widget that serves one configured Hugging Face model.

Templates and static files are pinned to the project root, so the process
working directory does not matter. The model loads on the first prediction
unless ``LAZY_LOAD=0``.
"""

from __future__ import annotations

import io
import traceback

from flask import Flask, jsonify, render_template, request
from PIL import Image

from modelwidget.service import ModelService
from modelwidget.settings import Settings
from paths import PROJECT_ROOT


def create_app(settings: Settings | None = None, service: ModelService | None = None) -> Flask:
    settings = settings or Settings.from_env()
    service = service or ModelService(settings)
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    app.config["SERVICE"] = service

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/info")
    def info():
        return jsonify(service.info())

    @app.post("/api/predict")
    def predict():
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
    application.run(host=settings.host, port=settings.port, threaded=True)


if __name__ == "__main__":
    main()
