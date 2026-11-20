"""Load one adapter and run prediction.

The model stays unloaded until the first prediction when ``LAZY_LOAD`` is on
(the default). A lock keeps a second request from loading a second copy on a
small machine.
"""

from __future__ import annotations

import threading

import torch

import modelwidget.adapters  # noqa: F401  (registers adapter classes)
from modelwidget.adapters.base import create_adapter, get_adapter_class
from modelwidget.devices import resolve_device, resolve_dtype
from modelwidget.settings import Settings
from paths import ensure_file, ensure_weights


class ModelBusy(RuntimeError):
    """The one loaded model is already answering a request."""


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.adapter_cls = get_adapter_class(settings.model_task, settings.model_backend)
        self.input_kind = self.adapter_cls.input_kind
        self.device = resolve_device(settings.device)
        self.dtype, self.dtype_name = resolve_dtype(
            settings.torch_dtype, self.device, settings.allow_cpu_half
        )
        self.snapshot = None
        self._adapter = None
        self._lock = threading.Lock()
        if settings.torch_num_threads is not None:
            torch.set_num_threads(settings.torch_num_threads)

    @property
    def loaded(self) -> bool:
        return self._adapter is not None

    def info(self) -> dict:
        return {
            "model_id": self.settings.model_id,
            "revision": self.settings.model_revision,
            "resolved_revision": None if self.snapshot is None else self.snapshot.name,
            "task": self.settings.model_task,
            "backend": self.settings.model_backend,
            "input_kind": self.input_kind,
            "device": self.device,
            "dtype": self.dtype_name,
            "lazy_load": self.settings.lazy_load,
            "loaded": self.loaded,
            "top_k": self.settings.top_k,
            "max_new_tokens": self.settings.max_new_tokens,
        }

    def ensure_loaded(self) -> None:
        with self._lock:
            self._load_unlocked()

    def predict(self, *, image=None, text: str | None = None) -> dict:
        if self.input_kind == "image" and image is None:
            raise ValueError("This model expects an image upload in the 'image' field.")
        if self.input_kind == "text" and not (text or "").strip():
            raise ValueError("This model expects text.")
        with self._lock:
            self._load_unlocked()
            assert self._adapter is not None
            body = self._adapter.predict(image=image, text=text)
        payload = self.info()
        payload.update(body)
        return payload

    def can_chat(self) -> bool:
        return self.settings.model_backend == "llama_cpp" and self.settings.model_task == "text-generation"

    def token_counter(self):
        """The loaded adapter's token counter. Call ensure_loaded first."""
        if self._adapter is None or not hasattr(self._adapter, "count_tokens"):
            raise RuntimeError("The GGUF model is not loaded.")
        return self._adapter.count_tokens

    def stream_chat(self, messages: list[dict], stop_event=None, max_tokens: int | None = None):
        """Yield reply text deltas. One chat at a time; shares the load lock with predict."""
        if not self.can_chat():
            raise ValueError("The active model is not a GGUF chat model.")
        if not self._lock.acquire(blocking=False):
            raise ModelBusy("The model is already answering another request.")
        try:
            self._load_unlocked()
            assert self._adapter is not None
            if not hasattr(self._adapter, "stream_chat"):
                raise ValueError("The active model is not a GGUF chat model.")
            tokens = max_tokens if max_tokens is not None else int(
                getattr(self.settings, "chat_max_tokens", 384)
            )
            yield from self._adapter.stream_chat(messages, max_tokens=tokens, stop_event=stop_event)
        finally:
            self._lock.release()

    def _load_unlocked(self) -> None:
        if self._adapter is not None:
            return
        if self.settings.model_file:
            # Single-file models (GGUF): fetch only that file, never the whole repo.
            snapshot = ensure_file(
                self.settings.model_id,
                self.settings.model_revision,
                self.settings.model_file,
                self.settings.weights_dir,
            )
        else:
            snapshot = ensure_weights(
                self.settings.model_id,
                self.settings.model_revision,
                self.settings.weights_dir,
            )
        print(
            f"Loading {self.settings.model_id} ({self.settings.model_task}, "
            f"{self.settings.model_backend}) on {self.device} {self.dtype_name} from {snapshot}",
            flush=True,
        )
        adapter = create_adapter(self.settings, snapshot)
        try:
            adapter.load(device=self.device, dtype=self.dtype)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {self.settings.model_id} from {snapshot}: {exc}"
            ) from exc
        self.snapshot = snapshot
        self._adapter = adapter
        print("Model loaded.", flush=True)
