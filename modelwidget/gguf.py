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
import threading
from pathlib import Path

def available() -> bool:
    """True when llama-cpp-python can be imported. Never imports it."""
    try:
        return importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ValueError):
        return False


class StopGeneration(Exception):
    """Raised inside a logits processor when the chat stop button is pressed."""


class GGUFTextGenerator:
    """Thin wrapper around ``llama_cpp.Llama``.

    ``generate()`` is the one-shot path used by ``/api/predict``. ``stream_chat()``
    is the chat page: it applies the model's GGUF chat template, streams tokens,
    and honours a stop event.
    """

    def __init__(self, model_path: Path, n_ctx: int = 2048, n_threads: int = 2):
        if not available():
            raise RuntimeError(
                "The GGUF backend needs llama-cpp-python: pip install llama-cpp-python"
            )
        from llama_cpp import Llama  # type: ignore

        self.model_path = Path(model_path)
        self.n_ctx = int(n_ctx)
        self._llm = Llama(
            model_path=str(self.model_path), n_ctx=self.n_ctx, n_threads=n_threads, verbose=False
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        out = self._llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return out["choices"][0]["message"]["content"]

    def count_tokens(self, messages: list[dict]) -> int:
        """Token length of ``messages`` after the model's chat template is applied."""
        prompt = self._format_prompt(messages)
        raw = prompt.encode("utf-8")
        return len(self._llm.tokenize(raw, special=True))

    def _format_prompt(self, messages: list[dict]) -> str:
        llm = self._llm
        formatter = None
        if getattr(llm, "chat_handler", None) is None:
            formatter = (getattr(llm, "_chat_handlers", {}) or {}).get(getattr(llm, "chat_format", None))
        if formatter is not None and callable(formatter) and not callable(getattr(llm, "chat_handler", None)):
            try:
                result = formatter(messages=messages)
                if hasattr(result, "prompt"):
                    return result.prompt
                if isinstance(result, str):
                    return result
            except TypeError:
                try:
                    result = formatter(messages)
                    if hasattr(result, "prompt"):
                        return result.prompt
                    if isinstance(result, str):
                        return result
                except Exception:
                    pass
            except Exception:
                pass
        # Fallback: a conservative stand-in when the formatter is a completion handler.
        parts = []
        for msg in messages:
            parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
        parts.append("assistant:")
        return "\n".join(parts)

    def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = 384,
        temperature: float = 0.7,
        stop_event: threading.Event | None = None,
    ):
        """Yield text deltas for one chat reply. Applies the GGUF chat template."""
        from llama_cpp import LogitsProcessorList  # type: ignore

        prepared = messages
        processors = None
        if stop_event is not None:
            def _stop(input_ids, scores):  # noqa: ARG001
                if stop_event.is_set():
                    raise StopGeneration()
                return scores
            processors = LogitsProcessorList([_stop])
        try:
            stream = self._llm.create_chat_completion(
                messages=prepared,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                logits_processor=processors,
            )
            for chunk in stream:
                if stop_event is not None and stop_event.is_set():
                    break
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    yield delta
        except StopGeneration:
            return
