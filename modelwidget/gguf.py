"""Planned llama.cpp backend for small quantised GGUF language models. STUB.

Catalog entries with ``"backend": "llama_cpp"`` and ``"status": "planned"``
pin one ``gguf_file`` at a Hugging Face commit. They are listed by
``GET /api/models`` under ``planned`` and are never installed by the picker.

Wiring this in later means:

1. ``pip install llama-cpp-python`` (see requirements-optional.txt; it builds
   llama.cpp, so it is optional and CPU-only by default).
2. Download just the one GGUF file with ``hf_hub_download(repo_id, filename=
   gguf_file, revision=..., cache_dir=weights/)`` instead of a whole snapshot.
3. Register a text-generation adapter for backend ``llama_cpp`` built on
   :class:`GGUFTextGenerator`, then flip the entries to ``"available"``.

On a 4 GB laptop keep ``n_ctx`` small (1024-2048) and ``n_threads`` at 2-4;
a Q4 1B model needs roughly 1 GB of RAM while running.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def available() -> bool:
    """True when llama-cpp-python can be imported. Never imports it."""
    try:
        return importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        return False


class GGUFTextGenerator:
    """Thin wrapper around ``llama_cpp.Llama``. Not registered as an adapter yet."""

    def __init__(self, model_path: Path, n_ctx: int = 2048, n_threads: int = 2):
        if not available():
            raise RuntimeError(
                "The GGUF backend needs llama-cpp-python: pip install llama-cpp-python"
            )
        from llama_cpp import Llama  # type: ignore

        self.model_path = Path(model_path)
        self._llm = Llama(
            model_path=str(self.model_path), n_ctx=n_ctx, n_threads=n_threads, verbose=False
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        out = self._llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return out["choices"][0]["message"]["content"]
