"""Download the configured model into the project weights folder.

Usage (from any working directory):

    python /path/to/this/project/install_weights.py

The repo id and revision come from ``config/.env`` (then a project-root
``.env``), or from the built-in tiny default when those files are absent.
If a snapshot is already available under ``weights/``, in the Hugging Face
cache, or via ``MODEL_WEIGHTS``, that path is printed and nothing is downloaded.
"""

from __future__ import annotations

from modelwidget.settings import Settings
from paths import ensure_weights


def main() -> int:
    settings = Settings.from_env()
    path = ensure_weights(settings.model_id, settings.model_revision, settings.weights_dir)
    print(f"Model weights ready: {path}", flush=True)
    print(f"Model: {settings.model_id}", flush=True)
    print(f"Task: {settings.model_task} ({settings.model_backend})", flush=True)
    if path.name:
        print(f"Snapshot revision: {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
