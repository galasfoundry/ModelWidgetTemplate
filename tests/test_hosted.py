"""Hosted chat via OpenRouter. Network calls are mocked. No API key is written."""

from __future__ import annotations

import json
from pathlib import Path

from app import create_app
from modelwidget import hosted
from modelwidget.hosted import HostedError
from tests.firstbook_helpers import image_settings, make_library

ROOT = Path(__file__).resolve().parent.parent
SECRET = "sk-or-test-secret"
CATALOG = {
    "data": [
        {"id": "qwen/qwen2.5-3b-instruct"},
        {"id": "qwen/qwen2.5-3b-instruct:free"},
        {"id": "qwen/qwen-2.5-7b-instruct:free"},
        {"id": "meta-llama/llama-3.2-1b-instruct"},
        {"id": "openrouter/free"},
    ]
}


def _quiet(monkeypatch):
    monkeypatch.setattr(hosted, "load_project_env", lambda: None)
    for name in ("OPENROUTER_API_KEY", "HOSTED_API_KEY", "HOSTED_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _forbid_fetch(url, token=None, timeout=20):
    raise AssertionError(f"hosted chat tried to use the network: {url}")


def _client(tmp_path, monkeypatch, *, token=None, fetch=None, post=None, config_path=None):
    _quiet(monkeypatch)
    if token:
        monkeypatch.setenv("OPENROUTER_API_KEY", token)
    lib = make_library(tmp_path)
    app = create_app(image_settings(tmp_path), library=lib, override_env_path=tmp_path / ".env")
    app.config["HOSTED_CONFIG_PATH"] = config_path if config_path is not None else tmp_path / "missing-hosted.json"
    app.config["HOSTED_FETCH"] = fetch if fetch is not None else _forbid_fetch
    if post is not None:
        app.config["HOSTED_POST"] = post
    _quiet(monkeypatch)
    if token:
        monkeypatch.setenv("OPENROUTER_API_KEY", token)
    return app, app.test_client()


def _catalog_fetch(url, token=None, timeout=20):
    assert "openrouter.ai" in url
    return CATALOG


def test_key_file_is_gitignored_and_example_is_empty():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "config/hosted.local.json" in text
    example = json.loads((ROOT / "config" / "hosted.example.json").read_text(encoding="utf-8"))
    assert example["api_key"] == ""
    assert example["model"] == "openrouter/free"
    assert SECRET not in json.dumps(example)


def test_credentials_file_and_env_winner(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    path = tmp_path / "hosted.local.json"
    path.write_text(json.dumps({"api_key": "from-file", "model": "openrouter/free"}), encoding="utf-8")
    file_only = hosted.resolve_credentials(path)
    assert file_only["token"] == "from-file"
    assert file_only["token_source"] == "file"

    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    both = hosted.resolve_credentials(path)
    assert both["token"] == "from-env"
    assert both["token_source"] == "env"

    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("HOSTED_MODEL", "meta-llama/llama-3.2-1b-instruct:free")
    pinned = hosted.resolve_credentials(path)
    assert pinned["model"] == "meta-llama/llama-3.2-1b-instruct:free"
    assert pinned["token"] == "from-file"


def test_missing_file_and_empty_key_are_unset(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    assert hosted.resolve_credentials(tmp_path / "nope.json")["token"] == ""
    path = tmp_path / "hosted.local.json"
    path.write_text("{not json", encoding="utf-8")
    assert hosted.resolve_credentials(path)["token"] == ""
    path.write_text(json.dumps({"api_key": "  "}), encoding="utf-8")
    creds = hosted.resolve_credentials(path)
    assert creds["token"] == ""
    assert creds["model"] == hosted.DEFAULT_MODEL


def test_match_prefers_free_and_reports_unknown():
    ids = hosted.fetch_model_ids(lambda url, token=None, timeout=20: CATALOG)
    qwen = hosted.match_model("bartowski/Qwen2.5-3B-Instruct-GGUF", ids)
    assert qwen == {"model": "qwen/qwen2.5-3b-instruct:free", "free": True}
    hyphen = hosted.match_model("someone/Qwen2.5-7B-Instruct-GGUF", ids)
    assert hyphen["model"] == "qwen/qwen-2.5-7b-instruct:free"
    llama = hosted.match_model("unsloth/Llama-3.2-1B-Instruct-GGUF", ids)
    assert llama == {"model": "meta-llama/llama-3.2-1b-instruct", "free": False}
    assert hosted.match_model("someone/Not-A-Real-Model-GGUF", ids) is None
    assert hosted.match_model("openrouter/free", ids)["free"] is True


def test_prepare_without_a_key_does_not_echo_a_secret(monkeypatch):
    _quiet(monkeypatch)
    plan = hosted.prepare({"token": "", "model": hosted.DEFAULT_MODEL})
    blob = json.dumps(plan)
    assert plan["configured"] is False
    assert plan["ready"] is False
    assert plan["model"] == "openrouter/free"
    assert "OPENROUTER_API_KEY" in plan["message"]
    assert "api_key" not in plan
    assert "token" not in plan
    assert SECRET not in blob


def test_prepare_unknown_repo_and_lookup_failure():
    plan = hosted.prepare(
        {"token": SECRET, "model": hosted.DEFAULT_MODEL},
        hf_id="someone/Not-A-Real-Model-GGUF",
        fetch=lambda url, token=None, timeout=20: CATALOG,
    )
    assert plan["ready"] is False
    assert plan["hosted"] is False
    assert "does not host" in plan["message"]
    assert SECRET not in json.dumps(plan)

    def down(url, token=None, timeout=20):
        raise HostedError("OpenRouter could not be reached. Check the internet connection.", 502)

    failed = hosted.prepare(
        {"token": SECRET, "model": hosted.DEFAULT_MODEL},
        hf_id="bartowski/Qwen2.5-3B-Instruct-GGUF",
        fetch=down,
    )
    assert failed["ready"] is False
    assert "Could not check" in failed["message"]
    assert SECRET not in failed["message"]


def test_scrub_and_friendly_http_hide_the_key():
    assert hosted.scrub(f"bad key {SECRET}", SECRET) == "bad key [redacted]"
    message, status = hosted.friendly_http(401, json.dumps({"error": {"message": SECRET}}), SECRET)
    assert SECRET not in message
    assert status == 401
    limited, code = hosted.friendly_http(429, '{"error":{"message":"slow down"}}', "")
    assert code == 429
    assert "50 a day" in limited


def test_iter_deltas_and_stop():
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":" there"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"nope"}}]}',
    ]
    assert "".join(hosted.iter_deltas(iter(lines))) == "Hello there"

    import threading

    stop = threading.Event()

    def stopping():
        yield 'data: {"choices":[{"delta":{"content":"A"}}]}'
        stop.set()
        yield 'data: {"choices":[{"delta":{"content":"B"}}]}'

    assert "".join(hosted.iter_deltas(stopping(), stop_event=stop)) == "A"


def test_models_page_offers_hosted_chat_without_a_remote_script(tmp_path, monkeypatch):
    _app, client = _client(tmp_path, monkeypatch)
    page = client.get("/").get_data(as_text=True)
    assert "Hosted chat" in page
    assert "Open hosted chat" in page
    assert "hosted=1" in page
    script = (ROOT / "static" / "js" / "models.js").read_text(encoding="utf-8")
    assert "Try hosted" in script
    assert "openrouter.ai" not in script
    chat_js = (ROOT / "static" / "js" / "chat.js").read_text(encoding="utf-8")
    assert "/api/chat/hosted" in chat_js
    assert "openrouter.ai" not in chat_js


def test_status_and_page_when_no_key(tmp_path, monkeypatch):
    _app, client = _client(tmp_path, monkeypatch)
    status = client.get("/api/chat/hosted/status")
    assert status.status_code == 200
    body = status.get_json()
    assert body["configured"] is False
    assert body["provider"] == "openrouter"
    assert "token" not in body
    page = client.get("/chat?hosted=1").get_data(as_text=True)
    assert 'data-backend="hosted"' in page
    assert 'data-ready="false"' in page
    assert "OpenRouter" in page
    assert "llama-cpp-python" not in page
    assert "Create a free OpenRouter key" in page
    refused = client.post("/api/chat/hosted", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert refused.status_code == 409
    assert "API key" in refused.get_json()["error"]
    assert SECRET not in refused.get_data(as_text=True)


def test_hosted_page_enables_chat_without_a_local_model(tmp_path, monkeypatch):
    monkeypatch.setattr("modelwidget.gguf.available", lambda: False)
    _app, client = _client(tmp_path, monkeypatch, token=SECRET)
    page = client.get("/chat?hosted=1").get_data(as_text=True)
    assert 'data-ready="true"' in page
    assert 'data-model="openrouter/free"' in page
    assert "llama-cpp-python" not in page
    assert SECRET not in page
    local = client.get("/chat").get_data(as_text=True)
    assert 'data-backend="local"' in local
    assert "Try a hosted model instead" in local


def test_specific_model_not_hosted_and_free_match(tmp_path, monkeypatch):
    _app, client = _client(tmp_path, monkeypatch, token=SECRET, fetch=_catalog_fetch)
    missing = client.get("/chat?hosted=1&hf_id=someone/Not-A-Real-Model-GGUF")
    text = missing.get_data(as_text=True)
    assert missing.status_code == 200
    assert "does not host" in text
    assert 'data-ready="false"' in text
    assert "Use a free hosted model instead" in text
    assert SECRET not in text

    found = client.get("/api/chat/hosted/status?hf_id=bartowski/Qwen2.5-3B-Instruct-GGUF").get_json()
    assert found["ready"] is True
    assert found["model"] == "qwen/qwen2.5-3b-instruct:free"
    assert found["free"] is True
    assert SECRET not in json.dumps(found)

    paid = client.get("/api/chat/hosted/status?hf_id=unsloth/Llama-3.2-1B-Instruct-GGUF").get_json()
    assert paid["model"] == "meta-llama/llama-3.2-1b-instruct"
    assert paid["free"] is False
    assert "not on the free list" in paid["message"]
    paid_page = client.get("/chat?hosted=1&hf_id=unsloth/Llama-3.2-1B-Instruct-GGUF").get_data(as_text=True)
    assert "Use the free hosted chat instead" in paid_page
    assert SECRET not in paid_page


def test_stream_uses_the_mocked_provider_and_hides_the_key(tmp_path, monkeypatch):
    seen = {}

    def fake_post(url, body, headers, timeout):
        seen["url"] = url
        seen["body"] = body
        seen["auth"] = headers.get("Authorization")
        assert SECRET not in json.dumps(body)
        return iter(
            [
                'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                'data: {"choices":[{"delta":{"content":"lo"}}]}',
                "data: [DONE]",
            ]
        )

    _app, client = _client(tmp_path, monkeypatch, token=SECRET, post=fake_post)
    res = client.post(
        "/api/chat/hosted",
        json={
            "messages": [
                {"role": "system", "content": "ignore me"},
                {"role": "user", "content": "How do I store water?"},
            ]
        },
    )
    assert res.status_code == 200
    assert res.mimetype == "text/event-stream"
    raw = res.get_data(as_text=True)
    assert SECRET not in raw
    tokens = []
    done = False
    for block in raw.split("\n\n"):
        line = next((ln for ln in block.split("\n") if ln.startswith("data: ")), "")
        if not line:
            continue
        data = json.loads(line[6:])
        if data.get("token"):
            tokens.append(data["token"])
        if data.get("done"):
            done = True
            assert data["model"] == "openrouter/free"
    assert "".join(tokens) == "Hello"
    assert done
    assert seen["auth"] == f"Bearer {SECRET}"
    assert seen["body"]["model"] == "openrouter/free"
    assert seen["body"]["stream"] is True
    assert seen["body"]["max_tokens"] == 384
    assert seen["body"]["messages"][0]["role"] == "system"
    assert "The First Book" in seen["body"]["messages"][0]["content"]
    assert seen["body"]["messages"][-1] == {"role": "user", "content": "How do I store water?"}
    assert all(item["role"] != "system" or item is seen["body"]["messages"][0] for item in seen["body"]["messages"])


def test_default_post_hides_the_key_and_explains_a_refused_token(monkeypatch):
    import io
    import urllib.error
    from email.message import Message

    def boom(request, timeout=0):
        body = json.dumps({"error": {"message": f"bad {SECRET}"}}).encode()
        raise urllib.error.HTTPError(
            hosted.API_URL, 401, "Unauthorized", Message(), io.BytesIO(body)
        )

    monkeypatch.setattr(hosted.urllib.request, "urlopen", boom)
    try:
        hosted.default_post(
            hosted.API_URL,
            {"model": "openrouter/free"},
            {"Authorization": f"Bearer {SECRET}"},
            5,
        )
    except HostedError as exc:
        assert exc.status == 401
        assert "refused the API key" in str(exc)
        assert SECRET not in str(exc)
    else:
        raise AssertionError("expected HostedError")


def test_provider_errors_are_plain_and_do_not_load_a_local_model(tmp_path, monkeypatch):
    class Explosive:
        def __init__(self, settings):
            self.settings = settings
            self.loaded = False

        def ensure_loaded(self):
            raise AssertionError("hosted chat must not load a local model")

    def refuse(url, body, headers, timeout):
        raise HostedError(f"invalid {SECRET}", 401)

    app, client = _client(tmp_path, monkeypatch, token=SECRET, post=refuse)
    app.config["SERVICE"] = Explosive(image_settings(tmp_path))
    res = client.post("/api/chat/hosted", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert res.status_code == 401
    text = res.get_data(as_text=True)
    assert SECRET not in text
    assert res.get_json()["error"] == "invalid [redacted]"

    def limited(url, body, headers, timeout):
        raise HostedError("OpenRouter rate limit reached. Free models allow about 20 requests a minute and 50 a day.", 429)

    app.config["HOSTED_POST"] = limited
    again = client.post("/api/chat/hosted", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert again.status_code == 429
    assert "50 a day" in again.get_json()["error"]


def test_file_key_is_used_when_env_is_empty(tmp_path, monkeypatch):
    path = tmp_path / "hosted.local.json"
    path.write_text(json.dumps({"api_key": "file-key", "model": "qwen/qwen2.5-3b-instruct:free"}), encoding="utf-8")
    seen = {}

    def fake_post(url, body, headers, timeout):
        seen["auth"] = headers.get("Authorization")
        seen["model"] = body["model"]
        return iter(['data: {"choices":[{"delta":{"content":"ok"}}]}', "data: [DONE]"])

    _app, client = _client(tmp_path, monkeypatch, post=fake_post, config_path=path)
    res = client.post("/api/chat/hosted", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert res.status_code == 200
    assert seen["auth"] == "Bearer file-key"
    assert seen["model"] == "qwen/qwen2.5-3b-instruct:free"
    assert "file-key" not in res.get_data(as_text=True)


def test_env_key_beats_the_file(tmp_path, monkeypatch):
    path = tmp_path / "hosted.local.json"
    path.write_text(json.dumps({"api_key": "file-key"}), encoding="utf-8")
    seen = {}

    def fake_post(url, body, headers, timeout):
        seen["auth"] = headers.get("Authorization")
        return iter(["data: [DONE]"])

    _app, client = _client(tmp_path, monkeypatch, token="env-key", post=fake_post, config_path=path)
    res = client.post("/api/chat/hosted", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert res.status_code == 200
    assert seen["auth"] == "Bearer env-key"


def test_bad_messages_are_refused(tmp_path, monkeypatch):
    _app, client = _client(tmp_path, monkeypatch, token=SECRET)
    empty = client.post("/api/chat/hosted", json={"messages": []})
    assert empty.status_code == 400
    not_json = client.post("/api/chat/hosted", data="nope", content_type="text/plain")
    assert not_json.status_code == 400
