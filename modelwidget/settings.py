"""Runtime settings read from the environment.

Copy ``config/.env.example`` to ``config/.env`` to override the defaults.
Shell variables win over the file. Paths are relative to the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from paths import PROJECT_ROOT, load_project_env, resolve_path

DEFAULT_MODEL_ID = "timm/mobilenetv3_small_100.lamb_in1k"
DEFAULT_MODEL_REVISION = "76b303ec28c466e9df37f4745dfb7aef20ab87ae"
DEFAULT_TASK = "image-classification"


def infer_backend(task: str, model_id: str) -> str:
    """Pick a backend when MODEL_BACKEND is ``auto`` or empty.

    Repo ids under the ``timm/`` org use the timm image adapter. Every other
    combination uses transformers.
    """
    if task == "image-classification" and model_id.startswith("timm/"):
        return "timm"
    return "transformers"


def _raw(name: str) -> str:
    return os.environ.get(name, "").strip()


def env_bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = _raw(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


@dataclass(frozen=True)
class Settings:
    model_id: str
    model_revision: str
    model_task: str
    model_backend: str
    device: str
    torch_dtype: str
    low_cpu_mem_usage: bool
    lazy_load: bool
    allow_cpu_half: bool
    top_k: int
    max_new_tokens: int
    max_input_chars: int
    host: str
    port: int
    max_upload_mb: int
    torch_num_threads: int | None
    weights_dir: Path
    label_map_path: Path | None

    @classmethod
    def from_env(cls) -> Settings:
        load_project_env()
        model_id = _raw("MODEL_ID") or DEFAULT_MODEL_ID
        revision = _raw("MODEL_REVISION")
        if not revision and model_id == DEFAULT_MODEL_ID:
            revision = DEFAULT_MODEL_REVISION
        task = _raw("MODEL_TASK") or DEFAULT_TASK
        backend = _raw("MODEL_BACKEND").lower() or "auto"
        if backend == "auto":
            backend = infer_backend(task, model_id)
        label_raw = _raw("LABEL_MAP_PATH")
        threads_raw = _raw("TORCH_NUM_THREADS")
        threads = int(threads_raw) if threads_raw else None
        if threads is not None and threads < 1:
            raise ValueError("TORCH_NUM_THREADS must be 1 or greater.")
        return cls(
            model_id=model_id,
            model_revision=revision,
            model_task=task,
            model_backend=backend,
            device=_raw("DEVICE").lower() or "auto",
            torch_dtype=_raw("TORCH_DTYPE").lower() or "float32",
            low_cpu_mem_usage=env_bool("LOW_CPU_MEM_USAGE", True),
            lazy_load=env_bool("LAZY_LOAD", True),
            allow_cpu_half=env_bool("ALLOW_CPU_HALF", False),
            top_k=max(1, env_int("TOP_K", 5)),
            max_new_tokens=max(1, env_int("MAX_NEW_TOKENS", 40)),
            max_input_chars=max(1, env_int("MAX_INPUT_CHARS", 4000)),
            host=_raw("HOST") or "127.0.0.1",
            port=env_int("PORT", 5000),
            max_upload_mb=max(1, env_int("MAX_UPLOAD_MB", 8)),
            torch_num_threads=threads,
            weights_dir=resolve_path("WEIGHTS_DIR", PROJECT_ROOT / "weights"),
            label_map_path=resolve_path("LABEL_MAP_PATH", Path()) if label_raw else None,
        )
