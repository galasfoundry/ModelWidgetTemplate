"""Offline tests for The First Book chat page: status, trim, SSE, stop, predict cap."""

from __future__ import annotations

from dataclasses import replace

from app import create_app
from firstbook import chat as chatlib
from modelwidget.settings import Settings
from tests.firstbook_helpers import image_settings, make_library


def test_settings_keep_predict_cap_and_add_chat_length():
    s = Settings.from_env()
    assert s.max_new_tokens == 40
    assert s.chat_max_tokens == 384
    assert s.llama_n_ctx == 2048


def test_normalize_and_build_messages_ignore_client_system():
    raw = [
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": "  Hello  "},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": ""},
        {"role": "tool", "content": "nope"},
        "bad",
    ]
    history = chatlib.normalize_history(raw, 4000)
    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    messages = chatlib.build_messages(history)
    assert messages[0]["role"] == "system"
    assert "The First Book" in messages[0]["content"]
    assert "electrical" in messages[0]["content"]
    assert messages[1:] == history


def test_trim_messages_drops_old_turns_and_keeps_latest_user():
    def count(msgs):
        return sum(len(m["content"]) for m in msgs)

    messages = [
        {"role": "system", "content": "SYS"},  # 3
        {"role": "user", "content": "aaaaaa"},  # 6
        {"role": "assistant", "content": "bbbbbb"},  # 6
        {"role": "user", "content": "cccccc"},  # 6
    ]
    # budget = n_ctx - max_tokens - reserve. 18-4-1=13: system+last two turns is 15,
    # so the first user and that assistant drop, leaving system + latest user (9).
    trimmed = chatlib.trim_messages(messages, count, n_ctx=18, max_tokens=4, reserve=1)
    assert [m["role"] for m in trimmed] == ["system", "user"]
    assert trimmed[-1]["content"] == "cccccc"
    assert trimmed[0]["content"] == "SYS"

    huge = [{"role": "system", "content": "S"}, {"role": "user", "content": "x" * 80}]
    cut = chatlib.trim_messages(huge, count, n_ctx=20, max_tokens=4, reserve=1)
    # budget=15, system 1, last user must shrink
    assert cut[0]["role"] == "system"
    assert cut[-1]["role"] == "user"
    assert 1 <= len(cut[-1]["content"]) < 80


def test_status_missing_backend_includes_install_command():
    class S:
        model_id = "timm/x"
        model_revision = "abc"
        model_task = "image-classification"
        model_backend = "timm"

    st = chatlib.status(
        backend_available=False,
        settings=S(),
        loaded=False,
        catalog_data={"models": []},
        installed_ids=set(),
    )
    assert st["ready"] is False and st["reason"] == "missing_backend"
    assert "prefer-binary" in st["install_command"]
    assert "abetlen.github.io/llama-cpp-python/whl/cpu" in st["install_command"]
    assert st["install_command"] == chatlib.INSTALL_COMMAND


def test_status_needs_a_selected_installed_chat_model():
    class S:
        model_id = "timm/x"
        model_revision = "abc"
        model_task = "image-classification"
        model_backend = "timm"

    gemma = {
        "id": "gemma-3-270m-it-gguf",
        "name": "Gemma",
        "hf_id": "unsloth/gemma-3-270m-it-GGUF",
        "revision": "c90975dbd40c0c7b275fefaae758c3415c906238",
        "task": "text-generation",
        "backend": "llama_cpp",
        "status": "downloadable",
    }
    empty = chatlib.status(
        backend_available=True, settings=S(), loaded=False,
        catalog_data={"models": [gemma]}, installed_ids=set(),
    )
    assert empty["reason"] == "no_chat_model" and empty["ready"] is False

    need = chatlib.status(
        backend_available=True, settings=S(), loaded=False,
        catalog_data={"models": [gemma]}, installed_ids={"gemma-3-270m-it-gguf"},
    )
    assert need["reason"] == "need_chat_model" and need["ready"] is False

    class G:
        model_id = gemma["hf_id"]
        model_revision = gemma["revision"]
        model_task = "text-generation"
        model_backend = "llama_cpp"

    ok = chatlib.status(
        backend_available=True, settings=G(), loaded=True,
        catalog_data={"models": [gemma]}, installed_ids={"gemma-3-270m-it-gguf"},
    )
    assert ok["ready"] is True and ok["active_chat_model"] == "gemma-3-270m-it-gguf"


class _FakeChatService:
    def __init__(self, settings, chunks="Hello from the fake model."):
        self.settings = settings
        self.loaded = True
        self.chunks = chunks
        self.stopped = False

    def can_chat(self):
        return True

    def ensure_loaded(self):
        self.loaded = True

    def token_counter(self):
        return lambda messages: sum(len(m.get("content") or "") for m in messages)

    def stream_chat(self, messages, stop_event=None, max_tokens=None):
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        for ch in self.chunks:
            if stop_event is not None and stop_event.is_set():
                self.stopped = True
                return
            yield ch


def _chat_client(tmp_path, monkeypatch, service=None):
    lib = make_library(tmp_path)
    settings = image_settings(tmp_path)
    if service is not None:
        settings = service.settings
    app = create_app(settings, library=lib, override_env_path=tmp_path / ".env")
    monkeypatch.setattr("modelwidget.gguf.available", lambda: True)

    def fake_evaluate(entry, caps, weights_dir):
        return {"installed": entry.get("id") == "gemma-3-270m-it-gguf", "compatible": True}

    monkeypatch.setattr("modelwidget.catalog.evaluate", fake_evaluate)
    if service is not None:
        app.config["SERVICE"] = service
    return app, app.test_client()


def _gemma_settings(tmp_path):
    return replace(
        image_settings(tmp_path),
        model_id="unsloth/gemma-3-270m-it-GGUF",
        model_revision="c90975dbd40c0c7b275fefaae758c3415c906238",
        model_task="text-generation",
        model_backend="llama_cpp",
        model_file="gemma-3-270m-it-Q4_K_M.gguf",
        chat_max_tokens=384,
        llama_n_ctx=2048,
        max_new_tokens=40,
    )


def test_chat_page_and_nav_when_backend_missing(tmp_path, monkeypatch):
    lib = make_library(tmp_path)
    app = create_app(image_settings(tmp_path), library=lib, override_env_path=tmp_path / ".env")
    monkeypatch.setattr("modelwidget.gguf.available", lambda: False)
    client = app.test_client()
    page = client.get("/chat").get_data(as_text=True)
    assert ">Chat<" in page  # nav + heading
    assert chatlib.INSTALL_COMMAND in page
    assert 'href="/"' in page or "model page" in page.lower()
    assert "/static/js/chat.js" in page
    home = client.get("/home").get_data(as_text=True)
    assert 'href="/chat"' in home
    status = client.get("/api/chat/status").get_json()
    assert status["ready"] is False and status["reason"] == "missing_backend"
    refused = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert refused.status_code == 409
    assert "llama-cpp-python" in refused.get_json()["error"]


def test_chat_streams_and_stop_endpoint(tmp_path, monkeypatch):
    settings = _gemma_settings(tmp_path)
    service = _FakeChatService(settings, chunks="Hello from the fake model.")
    app, client = _chat_client(tmp_path, monkeypatch, service)
    page = client.get("/chat").get_data(as_text=True)
    assert "data-ready=\"true\"" in page
    assert "gemma-3-270m-it-gguf" in page
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Hi there"}]})
    assert res.status_code == 200
    assert res.mimetype == "text/event-stream"
    import json as _json
    body = res.get_data(as_text=True)
    tokens = []
    done = False
    for block in body.split("\n\n"):
        line = next((ln for ln in block.split("\n") if ln.startswith("data: ")), "")
        if not line:
            continue
        data = _json.loads(line[6:])
        if data.get("token"):
            tokens.append(data["token"])
        if data.get("done"):
            done = True
    assert "".join(tokens) == "Hello from the fake model."
    assert done
    stop = client.post("/api/chat/stop")
    assert stop.status_code == 200 and stop.get_json()["stopped"] is True
    assert app.config["CHAT_STOP"].is_set()

    # A stream already seeing the stop event yields nothing further.
    service2 = _FakeChatService(settings, chunks="ABCDE")
    app.config["SERVICE"] = service2
    app.config["CHAT_STOP"].set()
    # The route clears the event at the start of a new request, so set it from
    # inside stream_chat after the first token by using a wrapper:

    class MidStop(service2.__class__):
        def stream_chat(self, messages, stop_event=None, max_tokens=None):
            yield "A"
            if stop_event is not None:
                stop_event.set()
            yield from super().stream_chat(messages, stop_event=stop_event, max_tokens=max_tokens)

    app.config["SERVICE"] = MidStop(settings, chunks="BCDE")
    body = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Go"}]}).get_data(as_text=True)
    assert "A" in body
    assert "E" not in body
    assert '"stopped": true' in body or '"stopped":true' in body


def test_api_predict_still_uses_max_new_tokens(tmp_path):
    """Chat length must not change the one-shot predict cap."""
    settings = replace(image_settings(tmp_path), max_new_tokens=40, chat_max_tokens=384)
    # The image adapter does not read chat_max_tokens; a text GGUF generate()
    # is still called with settings.max_new_tokens (see the llama_cpp adapter).
    import inspect
    from modelwidget.adapters.llama_cpp import LlamaCppTextGenerator

    src = inspect.getsource(LlamaCppTextGenerator.predict)
    assert "max_new_tokens" in src
    assert "chat_max_tokens" not in src
    assert settings.max_new_tokens == 40 and settings.chat_max_tokens == 384
