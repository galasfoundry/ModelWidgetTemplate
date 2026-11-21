"""Offline model catalog and the model picker's select / install / switch path.

The catalog is a small JSON file (``config/models_catalog.json`` by default,
``MODELS_CATALOG_PATH`` to override). Each entry maps onto the four existing
settings: ``hf_id`` -> ``MODEL_ID``, ``revision`` -> ``MODEL_REVISION``,
``task`` -> ``MODEL_TASK`` and ``backend`` -> ``MODEL_BACKEND``.

Selecting a model:

1. checks the entry against this machine (RAM, disk, backend, adapter),
2. reuses the exact pinned snapshot when it is already on disk, otherwise
   downloads it with the same ``paths.download_weights`` call that
   ``install_weights.py`` uses,
3. loads a new :class:`ModelService` from that snapshot,
4. writes the four ``MODEL_*`` keys to ``config/.env`` so the next start
   uses the same model. Other lines in that file are kept.

Nothing is written unless the install and the load both succeed, so a bad
catalog entry leaves the running model alone.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path

import paths
from modelwidget.adapters.base import get_adapter_class
from modelwidget.settings import Settings

CATALOG_PATH = paths.PROJECT_ROOT / "config" / "models_catalog.json"
OVERRIDE_ENV_PATH = paths.PROJECT_ROOT / "config" / ".env"

REQUIRED_FIELDS = ("id", "name", "hf_id", "revision", "task", "backend")
# "available": download and activate from the picker.
# "downloadable": download any time; activate only when its backend is
#   installed (GGUF models need the optional llama-cpp-python).
# "planned": listed for the roadmap only; never downloaded or activated.
STATUSES = ("available", "downloadable", "planned")
CATEGORIES = ("vision", "text", "llm", "audio", "other")
SETTING_KEYS = {
    "MODEL_ID": "hf_id",
    "MODEL_REVISION": "revision",
    "MODEL_TASK": "task",
    "MODEL_BACKEND": "backend",
    # Single-file models (GGUF). Written empty for whole-snapshot models.
    "MODEL_FILE": "gguf_file",
}
# On a machine with this much RAM or less, big language models get a warning.
SMALL_MACHINE_GB = 4.5
BIG_LLM_MB = 700
# A machine sold as "2 GB" reports a little under 2 GiB, so allow 10%.
RAM_TOLERANCE = 0.9
# Leave room for the download, the cache metadata, and anything else.
DISK_HEADROOM = 2.0


class SelectionError(Exception):
    """A selection that could not be applied. ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def catalog_path() -> Path:
    return paths.resolve_path("MODELS_CATALOG_PATH", CATALOG_PATH)


def load_catalog(path: Path | None = None) -> dict:
    """Read the catalog. Malformed entries go to ``invalid`` instead of failing."""
    path = Path(path) if path is not None else catalog_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SelectionError(f"Model catalog not found at {path}.", 500) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"Could not read the model catalog at {path}: {exc}", 500) from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise SelectionError(f"Model catalog at {path} has no 'models' list.", 500)
    models: list[dict] = []
    planned: list[dict] = []
    invalid: list[dict] = []
    seen: set[str] = set()
    for index, entry in enumerate(data["models"]):
        missing = [
            key
            for key in REQUIRED_FIELDS
            if not isinstance(entry, dict) or not str(entry.get(key) or "").strip()
        ]
        if missing:
            invalid.append({"index": index, "error": f"missing {', '.join(missing)}"})
            continue
        if entry["id"] in seen:
            invalid.append({"index": index, "error": f"duplicate id {entry['id']!r}"})
            continue
        status = entry.get("status", "available")
        if status not in STATUSES:
            invalid.append({"index": index, "error": f"unknown status {status!r}"})
            continue
        seen.add(entry["id"])
        (planned if status == "planned" else models).append(entry)
    # User-added picks live beside the curated catalog (models_user.json) and
    # are never written back into this file.
    user_file = path.with_name("models_user.json")
    from modelwidget.discover import load_user_models

    for entry in load_user_models(user_file):
        if entry["id"] in seen:
            invalid.append({"id": entry["id"], "error": "user id collides with the curated catalog"})
            continue
        seen.add(entry["id"])
        (planned if entry.get("status") == "planned" else models).append(entry)
    return {
        "version": data.get("version"),
        "description": data.get("description", ""),
        "path": str(path),
        "user_path": str(user_file),
        "models": models,
        "planned": planned,
        "invalid": invalid,
    }


def installed_snapshot(
    repo_id: str, revision: str, weights_dir: Path, filename: str | None = None
) -> Path | None:
    """The exact pinned snapshot on disk, or None. Never downloads.

    ``weights/`` is checked first, then the Hugging Face hub cache, the same
    pools ``paths.find_local_snapshot`` searches. Unlike that function, a
    different revision of the same repo does not count. With ``filename``
    (a GGUF file) the snapshot only counts when that file is in it.
    """
    for root in (Path(weights_dir), paths.huggingface_hub_dir()):
        names = {revision}
        ref = paths._read_ref(root, repo_id, revision)
        if ref:
            names.add(ref)
        if filename:
            for name in names:
                snapshot = root / paths.model_dir_name(repo_id) / "snapshots" / name
                if (snapshot / filename).is_file():
                    return snapshot.resolve()
            continue
        for snapshot in paths._snapshot_dirs(root, repo_id):
            if snapshot.name in names:
                return snapshot.resolve()
    return None


def entry_snapshot(entry: dict, weights_dir: Path) -> Path | None:
    return installed_snapshot(
        entry["hf_id"], entry["revision"], weights_dir, filename=entry.get("gguf_file") or None
    )


def evaluate(entry: dict, caps: dict, weights_dir: Path) -> dict:
    """Return the entry plus ``compatible``, ``reasons`` and ``installed``."""
    reasons: list[str] = []
    warnings: list[str] = []
    snapshot = entry_snapshot(entry, weights_dir)

    min_ram = _number(entry.get("min_ram_gb"))
    total_ram = caps.get("ram_total_gb")
    if min_ram is not None and total_ram is not None and total_ram < min_ram * RAM_TOLERANCE:
        reasons.append(f"needs {min_ram:g} GB RAM, this machine has {total_ram:.1f} GB")

    size_mb = _number(entry.get("approx_size_mb"))
    free_mb = caps.get("disk_free_mb")
    if snapshot is None and size_mb is not None and free_mb is not None:
        if free_mb < size_mb * DISK_HEADROOM:
            reasons.append(f"needs about {size_mb:g} MB free disk, {free_mb:.0f} MB free")

    # RAM and disk limits stop downloads too; a missing backend only stops activation.
    download_blockers = list(reasons)
    status = entry.get("status", "available")
    if status == "planned":
        download_blockers.append("planned: not downloadable yet")
    gguf_name = str(entry.get("gguf_file") or "")
    if gguf_name:
        # Lazy import: discover imports this module.
        from modelwidget.discover import is_sidecar_gguf

        if is_sidecar_gguf(gguf_name):
            note = (
                "this file is a drafter or projector beside the model, not a chat model "
                "the app can run"
            )
            reasons.append(note)
            download_blockers.append(note)

    backend = entry["backend"]
    if caps.get("backends", {}).get(backend) is False:
        package = PIP_NAMES.get(backend, backend)
        reasons.append(f"backend {backend!r} is not installed (pip install {package}); downloading still works")
    try:
        get_adapter_class(entry["task"], backend)
    except ValueError:
        reasons.append(f"no adapter for {entry['task']} via {backend}")
    if status == "planned":
        reasons.append("planned: not supported yet")

    if (
        category_of(entry) == "llm"
        and size_mb is not None
        and size_mb >= BIG_LLM_MB
        and total_ram is not None
        and total_ram <= SMALL_MACHINE_GB
    ):
        warnings.append(
            f"about {size_mb:g} MB on a {total_ram:.1f} GB machine: close other apps before using it"
        )

    result = dict(entry)
    result["category"] = category_of(entry)
    result["downloadable"] = not download_blockers
    result["warnings"] = warnings
    result["compatible"] = not reasons
    result["reasons"] = reasons
    result["installed"] = snapshot is not None
    result["snapshot"] = None if snapshot is None else str(snapshot)
    return result


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


PIP_NAMES = {"llama_cpp": "llama-cpp-python"}


def category_of(entry: dict) -> str:
    """The entry's category, or one inferred from its task for older catalogs."""
    category = entry.get("category")
    if category in CATEGORIES:
        return category
    task = str(entry.get("task", ""))
    if task.startswith("image"):
        return "vision"
    if task == "text-generation":
        return "llm"
    if task.startswith("text"):
        return "text"
    return "other"


def is_active(entry: dict, settings: Settings) -> bool:
    return (
        entry["hf_id"] == settings.model_id
        and entry["revision"] == settings.model_revision
        and entry["task"] == settings.model_task
        and entry["backend"] == settings.model_backend
        and (entry.get("gguf_file") or "") == (getattr(settings, "model_file", "") or "")
    )


def settings_for(entry: dict, base: Settings) -> Settings:
    return replace(
        base,
        model_id=entry["hf_id"],
        model_revision=entry["revision"],
        model_task=entry["task"],
        model_backend=entry["backend"],
        model_file=entry.get("gguf_file") or "",
    )


def install(entry: dict, weights_dir: Path) -> tuple[Path, bool]:
    """Return (snapshot, downloaded). Downloads only when the pinned snapshot is missing."""
    found = entry_snapshot(entry, weights_dir)
    if found is not None:
        return found, False
    try:
        if entry.get("gguf_file"):
            from modelwidget.downloads import download_entry

            snapshot, _ = download_entry(entry, weights_dir)
        else:
            snapshot = paths.download_weights(entry["hf_id"], entry["revision"], weights_dir)
    except Exception as exc:
        raise SelectionError(
            f"{entry['name']} is not on this machine yet and could not be downloaded. "
            f"Connect to the internet once, or copy its snapshot into weights/. ({exc})",
            502,
        ) from exc
    return snapshot, True


def write_override(values: dict[str, str], env_path: Path | None = None) -> Path:
    """Set keys in ``config/.env``, keeping every other line as it was."""
    path = Path(env_path) if env_path is not None else OVERRIDE_ENV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    else:
        lines = [
            "# Written by the model picker. Edit freely; other keys are kept.",
            "# Shell exports still win over this file.",
        ]
    pending = dict(values)
    pattern = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
    out: list[str] = []
    for line in lines:
        match = pattern.match(line)
        if match and match.group(1) in values:
            key = match.group(1)
            if key in pending:
                out.append(f"{key}={pending.pop(key)}")
            # A later duplicate of the same key is dropped so the file has one value.
            continue
        out.append(line)
    for key, value in pending.items():
        out.append(f"{key}={value}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def shell_overrides() -> list[str]:
    """MODEL_* keys exported in the shell, which beat config/.env on restart."""
    return sorted(key for key in SETTING_KEYS if key in paths.SHELL_ENV_KEYS)
