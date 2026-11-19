"""llama.cpp backend for small quantised GGUF language models.

Catalog entries with ``"backend": "llama_cpp"`` and ``"status":
"downloadable"`` pin one ``gguf_file`` at a Hugging Face commit. The picker
and ``scripts/download_models.py`` download just that file into
``weights/``. Using the model needs ``pip install llama-cpp-python`` (see
requirements-optional.txt); the adapter in
``modelwidget/adapters/llama_cpp.py`` then loads it through
:class:`GGUFTextGenerator`.

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
