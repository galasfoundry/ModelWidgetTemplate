"""Device and dtype selection.

``DEVICE=auto`` uses CUDA, then Apple MPS, then CPU. Half precision stays on
float32 when the device is CPU, because float16 convolutions there are often
unsupported or slower. Set ``ALLOW_CPU_HALF=1`` to keep the requested dtype.
"""

from __future__ import annotations

import torch

_DTYPE_NAMES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}
_DTYPE_LABELS = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


def resolve_device(requested: str) -> str:
    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if choice not in {"cpu", "cuda", "mps"}:
        raise ValueError("DEVICE must be auto, cpu, cuda, or mps.")
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("DEVICE=cuda but CUDA is not available. Use DEVICE=auto or DEVICE=cpu.")
    if choice == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("DEVICE=mps but MPS is not available. Use DEVICE=auto or DEVICE=cpu.")
    return choice


def resolve_dtype(name: str, device: str, allow_cpu_half: bool) -> tuple[torch.dtype, str]:
    key = (name or "float32").strip().lower()
    if key not in _DTYPE_NAMES:
        raise ValueError("TORCH_DTYPE must be float32, float16, or bfloat16.")
    dtype = _DTYPE_NAMES[key]
    if device == "cpu" and dtype is not torch.float32 and not allow_cpu_half:
        print(
            f"TORCH_DTYPE={key} on CPU is kept at float32. "
            "Set ALLOW_CPU_HALF=1 to force the requested dtype.",
            flush=True,
        )
        return torch.float32, "float32"
    return dtype, _DTYPE_LABELS[dtype]
