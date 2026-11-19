"""Text generation from one GGUF file with llama.cpp (optional llama-cpp-python).

Registered for ``("text-generation", "llama_cpp")``. Importing this module
never imports llama_cpp; that happens in :meth:`load`, so the app starts on
machines without it and the picker reports the missing package instead.
"""

from __future__ import annotations

import os
from pathlib import Path

from modelwidget.adapters.base import ModelAdapter, register
from modelwidget.gguf import GGUFTextGenerator


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


@register("text-generation", "llama_cpp")
class LlamaCppTextGenerator(ModelAdapter):
    input_kind = "text"

    def __init__(self, settings, snapshot: Path):
        self.settings = settings
        self.snapshot = Path(snapshot)
        self.generator: GGUFTextGenerator | None = None

    def model_path(self) -> Path:
        name = (self.settings.model_file or "").strip()
        if name:
            return self.snapshot / name
        found = sorted(self.snapshot.glob("*.gguf"))
        if len(found) != 1:
            raise RuntimeError(
                f"Set MODEL_FILE to the .gguf file to load; found {len(found)} in {self.snapshot}."
            )
        return found[0]

    def load(self, *, device: str, dtype=None) -> None:
        threads = self.settings.torch_num_threads or min(4, os.cpu_count() or 2)
        self.generator = GGUFTextGenerator(
            self.model_path(), n_ctx=_env_int("LLAMA_N_CTX", 2048), n_threads=threads
        )

    def predict(self, *, image=None, text: str | None = None) -> dict:
        if self.generator is None:
            raise RuntimeError("The GGUF model is not loaded.")
        prompt = (text or "").strip()[: self.settings.max_input_chars]
        if not prompt:
            raise ValueError("This model expects text.")
        reply = self.generator.generate(prompt, max_tokens=self.settings.max_new_tokens)
        return {
            "results": [{"label": "completion", "text": reply, "score": None, "rank": 1}],
            "text": reply,
        }
