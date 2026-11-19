"""Detect what this machine can run: RAM, free disk, GPU, and backends.

``psutil`` is used when it is installed. Without it, Linux reads
``/proc/meminfo`` and every platform falls back to ``shutil.disk_usage``.
Nothing here imports torch at module load, and nothing touches the network.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

_GB = 1024**3
_MB = 1024**2


def _memory() -> tuple[float | None, float | None, str]:
    """Return (total_gb, available_gb, source)."""
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.total / _GB, vm.available / _GB, "psutil"
    except Exception:
        pass
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    values[key.strip()] = int(parts[0]) * 1024
        total = values.get("MemTotal")
        available = values.get("MemAvailable", values.get("MemFree"))
        if total:
            return total / _GB, (available / _GB if available else None), "proc"
    except OSError:
        pass
    return None, None, "unknown"


def _existing_parent(path: Path) -> Path:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _disk_free_mb(path: Path) -> float | None:
    target = _existing_parent(path)
    try:
        import psutil  # type: ignore

        return psutil.disk_usage(str(target)).free / _MB
    except Exception:
        pass
    try:
        return shutil.disk_usage(str(target)).free / _MB
    except OSError:
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _backend_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_capabilities(weights_dir: Path) -> dict:
    """Return a JSON-ready summary of this machine's limits."""
    total_gb, available_gb, source = _memory()
    return {
        "ram_total_gb": None if total_gb is None else round(total_gb, 2),
        "ram_available_gb": None if available_gb is None else round(available_gb, 2),
        "disk_free_mb": _round(_disk_free_mb(weights_dir)),
        "weights_dir": str(weights_dir),
        "cuda_available": _cuda_available(),
        "backends": {name: _backend_available(name) for name in ("timm", "transformers")},
        # Not used by the picker yet; reported so the planned GGUF path can be gated on it.
        "optional_backends": {"llama_cpp": _backend_available("llama_cpp")},
        "psutil": _backend_available("psutil"),
        "source": source,
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)
