"""Hosted chat for models that will not fit on this laptop.

OpenRouter is the provider. A free API key and the ``openrouter/free``
router let ``/chat?hosted=1`` talk to a large model without downloading it.
Kaggle and Colab were set aside because they only run a notebook; they do
not give this app an address to send a chat message to. Hugging Face's
router bills the same providers, and a free Hugging Face account currently
includes no credits.

The key is read from ``OPENROUTER_API_KEY`` (or ``HOSTED_API_KEY``) in the
environment, which ``config/.env`` already feeds, or from
``config/hosted.local.json``. That file is gitignored. Nothing in this
module puts the key in a response.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Iterator

from paths import PROJECT_ROOT, load_project_env

PROVIDER = "openrouter"
PROVIDER_NAME = "OpenRouter"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_MODEL = "openrouter/free"
KEYS_URL = "https://openrouter.ai/keys"
LOCAL_CONFIG_NAME = "hosted.local.json"
USER_AGENT = "TheFirstBook/0.1 (hosted chat)"
CHAT_TIMEOUT = 90
MODELS_TIMEOUT = 20
MODELS_CACHE_SECONDS = 600
# A rough stand-in for a context window when trimming history. Free models
# are larger than this; the cap only stops a very long page session from
# posting every turn.
HOSTED_CONTEXT = 8192

LIMITS_NOTE = (
    "Free models allow about 20 requests a minute and 50 a day. "
    "After the account has bought at least $10 of credits, the daily cap "
    "rises to about 1,000. A failed try still counts. The free model list "
    "changes, and a busy free model can refuse a request even when your "
    "own cap is not used up."
)

SETUP_NOTE = (
    "Create a free key at https://openrouter.ai/keys (no credit card). "
    "Put OPENROUTER_API_KEY=your-key in config/.env, which is not committed, "
    "or copy config/hosted.example.json to config/hosted.local.json and set "
    "api_key there. Restart the app if it was already running when you "
    "edited config/.env."
)

# Orgs that republish someone else's weights as GGUF. The model name is the
# part after the slash; the org is not who hosts the API.
QUANT_ORGS = {
    "bartowski",
    "unsloth",
    "thebloke",
    "lmstudio-community",
    "mradermacher",
    "quantfactory",
    "second-state",
    "librai",
    "tensorblock",
    "lone-striker",
    "lonestriker",
}

# First matching prefix of the model slug picks an OpenRouter publisher.
FAMILY_ORGS = (
    ("llama", "meta-llama"),
    ("qwen", "qwen"),
    ("gemma", "google"),
    ("mistral", "mistralai"),
    ("mixtral", "mistralai"),
    ("phi-", "microsoft"),
    ("phi", "microsoft"),
    ("deepseek", "deepseek"),
    ("nemotron", "nvidia"),
    ("yi-", "01-ai"),
    ("command-r", "cohere"),
    ("smollm", "huggingface"),
    ("tinyllama", "tinyllama"),
)

_GGUF_TAIL = re.compile(r"[-_.]?gguf$", re.I)
_QUANT_TAIL = re.compile(
    r"[-_.](?:ud[-_])?(?:q[2-8](?:_[k0-9])?(?:_[smlx]{1,2})?|iq[1-4](?:_[mnsxl]{1,3})?|"
    r"f16|bf16|fp16|f32)$",
    re.I,
)
_VERSION_HYPHEN = re.compile(r"([a-z])(\d)")

HOSTED_SYSTEM_PROMPT = (
    "You are The First Book, a practical assistant for UK survival and "
    "self-sufficiency. You help with practical skills: off-grid power, water, "
    "food, shelter, repair, navigation and first aid. Be practical and concise. "
    "Be cautious about safety, especially electrical work, water, and medical "
    "advice: warn about real dangers, do not invent procedures you are unsure "
    "of, and say when a qualified person or official guidance is needed. If you "
    "do not know, say so."
)

FetchGet = Callable[..., object]
PostFn = Callable[..., Iterator[str]]

_models_cache: dict = {"at": 0.0, "ids": None}


class HostedError(Exception):
    """A hosted-chat failure the page can show. ``status`` is an HTTP code."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def local_config_path():
    return PROJECT_ROOT / "config" / LOCAL_CONFIG_NAME


def clear_model_cache() -> None:
    _models_cache["at"] = 0.0
    _models_cache["ids"] = None


def load_local_config(path=None) -> dict:
    """Read the gitignored JSON file. A missing or broken file is empty."""
    target = local_config_path() if path is None else path
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_credentials(path=None) -> dict:
    """Token and the owner's pinned model. The environment wins over the file.

    Callers must not put this dict on an HTTP response. ``token`` may be empty.
    """
    load_project_env()
    file_data = load_local_config(path)
    file_key = str(file_data.get("api_key") or file_data.get("token") or "").strip()
    env_key = ""
    for name in ("OPENROUTER_API_KEY", "HOSTED_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            env_key = value
            break
    file_model = str(file_data.get("model") or "").strip()
    model = os.environ.get("HOSTED_MODEL", "").strip() or file_model or DEFAULT_MODEL
    return {
        "token": env_key or file_key,
        "token_source": "env" if env_key else ("file" if file_key else ""),
        "model": model,
    }


def scrub(text: str, token: str) -> str:
    """Drop the key if a provider echoed it back in an error."""
    if token and text and token in text:
        return text.replace(token, "[redacted]")
    return text


def _is_free_id(model: str) -> bool:
    return model == DEFAULT_MODEL or model.endswith(":free")


def _strip_file_name(name: str) -> str:
    slug = name.strip().lower().replace("_", "-")
    changed = True
    while changed and slug:
        changed = False
        nxt = _GGUF_TAIL.sub("", slug)
        nxt = _QUANT_TAIL.sub("", nxt)
        if nxt and nxt != slug:
            slug = nxt.strip("-")
            changed = True
    return slug


def _hyphenate_version(slug: str) -> str:
    return _VERSION_HYPHEN.sub(r"\1-\2", slug)


def candidate_model_ids(hf_id: str) -> list[str]:
    """OpenRouter ids that might correspond to a Hugging Face repo.

    Quant-mirror orgs (bartowski, unsloth, and the like) are dropped. A
    ``:free`` variant is listed beside each guess. The caller keeps only ids
    the provider actually publishes.
    """
    text = (hf_id or "").strip()
    if not text:
        return []
    if text.lower() == DEFAULT_MODEL:
        return [DEFAULT_MODEL]
    ids: list[str] = []

    def add(model_id: str) -> None:
        model_id = model_id.strip().strip("-").lower()
        if not model_id or "/" not in model_id:
            return
        for item in (model_id, model_id + ":free"):
            if item not in ids:
                ids.append(item)

    if "/" not in text:
        add(text.lower())
        return [item for item in ids if item != DEFAULT_MODEL + ":free"]

    org, _, name = text.partition("/")
    org_l = org.lower()
    bare = _strip_file_name(name)
    shapes = []
    for slug in (bare, _hyphenate_version(bare)):
        if slug and slug not in shapes:
            shapes.append(slug)

    if org_l not in QUANT_ORGS:
        for slug in shapes:
            add(f"{org_l}/{slug}")

    family = None
    for prefix, publisher in FAMILY_ORGS:
        if bare.startswith(prefix):
            family = publisher
            break
    if family and family != org_l:
        for slug in shapes:
            add(f"{family}/{slug}")
    return ids


def match_model(hf_id: str, catalog_ids: set[str]) -> dict | None:
    """Pick a hosted id for ``hf_id``. A free variant wins when both exist."""
    text = (hf_id or "").strip()
    if not text:
        return None
    if text == DEFAULT_MODEL or text in catalog_ids:
        return {"model": text, "free": _is_free_id(text)}
    candidates = candidate_model_ids(text)
    free_hits = [item for item in candidates if item.endswith(":free") and item in catalog_ids]
    if free_hits:
        return {"model": free_hits[0], "free": True}
    paid = [item for item in candidates if not item.endswith(":free") and item in catalog_ids]
    if paid:
        return {"model": paid[0], "free": False}
    return None


def _base_info(*, configured: bool, hf_id: str) -> dict:
    return {
        "provider": PROVIDER,
        "provider_name": PROVIDER_NAME,
        "configured": configured,
        "ready": False,
        "hosted": False,
        "model": "",
        "hf_id": hf_id,
        "free": False,
        "message": "",
        "limits": LIMITS_NOTE,
        "setup": SETUP_NOTE,
        "keys_url": KEYS_URL,
    }


def _finish(info: dict, model: str, *, free: bool) -> dict:
    info["model"] = model
    info["free"] = free
    info["hosted"] = True
    note = ""
    if not free:
        note = (
            f"This uses {model}, which is not on the free list. "
            "OpenRouter may ask for credits. The general hosted chat stays on the free router."
        )
    if not info["configured"]:
        info["ready"] = False
        info["message"] = "Hosted chat needs an OpenRouter API key. " + SETUP_NOTE
        if note:
            info["message"] = note + " " + info["message"]
        return info
    info["ready"] = True
    info["message"] = note or f"Ready. This chat uses {model} on OpenRouter."
    return info


def prepare(creds: dict, *, hf_id: str = "", requested_model: str = "", fetch: FetchGet | None = None) -> dict:
    """Decide which hosted model to call, without revealing the API key.

    ``openrouter/free`` and any ``:free`` id are accepted without a model-list
    lookup. A Hugging Face repo is matched against that list, and a repo
    OpenRouter does not serve is reported as not hosted.
    """
    hf_id = (hf_id or "").strip()
    requested_model = (requested_model or "").strip()
    configured_model = (creds.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    info = _base_info(configured=bool(creds.get("token")), hf_id=hf_id)

    if requested_model and _is_free_id(requested_model):
        return _finish(info, requested_model, free=True)

    if not hf_id:
        model = requested_model or configured_model
        if _is_free_id(model) or model == configured_model:
            return _finish(info, model, free=_is_free_id(model))

    try:
        ids = load_model_ids(fetch, creds.get("token") or None)
    except HostedError as exc:
        info["message"] = (
            "Could not check which models OpenRouter is hosting, so this one was not started. "
            f"{exc} Open hosted chat without a specific model to use the free router."
        )
        return info

    if requested_model and (requested_model in ids or _is_free_id(requested_model)):
        return _finish(info, requested_model, free=_is_free_id(requested_model))

    match = match_model(hf_id, ids) if hf_id else None
    if match is None:
        target = hf_id or requested_model or "that model"
        info["message"] = (
            f"OpenRouter does not host {target}. A Hugging Face GGUF repo is often only a "
            "file mirror, not a model their API can run. Open hosted chat without that model "
            "to talk to a free one instead."
        )
        return info
    return _finish(info, match["model"], free=match["free"])


def build_messages(history: list[dict]) -> list[dict]:
    """System prompt first. The browser cannot replace it."""
    return [{"role": "system", "content": HOSTED_SYSTEM_PROMPT}, *history]


def rough_token_count(messages: list[dict]) -> int:
    """About four characters to a token. Used only to trim a long page history."""
    return sum(max(1, len(str(item.get("content") or "")) // 4) for item in messages)


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": USER_AGENT,
        "HTTP-Referer": "http://127.0.0.1:5000/",
        "X-Title": "The First Book",
    }


def _extract_message(body: str) -> str:
    text = (body or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:300]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or "")[:300]
        if isinstance(err, str):
            return err[:300]
        if data.get("message"):
            return str(data["message"])[:300]
    return text[:300]


def friendly_http(code: int, body: str, token: str = "") -> tuple[str, int]:
    """A short sentence for a provider HTTP error, with the key removed."""
    detail = scrub(_extract_message(body), token)
    if code == 401:
        return (
            "OpenRouter refused the API key. Check OPENROUTER_API_KEY in config/.env "
            "or the key in config/hosted.local.json.",
            401,
        )
    if code == 402:
        return (
            "OpenRouter says this model needs credits. Use the general hosted chat, "
            "which stays on the free router, or add credits on the OpenRouter site.",
            402,
        )
    if code == 404:
        return (
            "OpenRouter does not host that model. Open hosted chat without a specific "
            "model to use a free one.",
            404,
        )
    if code == 429:
        extra = f" {detail}" if detail else ""
        return (
            "OpenRouter rate limit reached. Free models allow about 20 requests a minute "
            f"and 50 a day. Wait and try again.{extra}",
            429,
        )
    if code == 400:
        return (detail or "OpenRouter rejected the request. The model may not be hosted.", 400)
    if detail:
        return (f"OpenRouter returned HTTP {code}: {detail}", 502 if code >= 500 else code)
    return (f"OpenRouter returned HTTP {code}.", 502 if code >= 500 else code)


def fetch_json(url: str, token: str | None = None, timeout: int = MODELS_TIMEOUT):
    """GET JSON. Tests pass their own function into :func:`load_model_ids`."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:500]
        message, status = friendly_http(exc.code, raw, token or "")
        raise HostedError(message, status) from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        raise HostedError(
            f"OpenRouter could not be reached. Check the internet connection. ({exc})",
            502,
        ) from exc


def fetch_model_ids(fetch: FetchGet | None = None, token: str | None = None) -> set[str]:
    getter = fetch or fetch_json
    payload = getter(MODELS_URL, token=token, timeout=MODELS_TIMEOUT)
    rows = payload.get("data") if isinstance(payload, dict) else payload
    ids: set[str] = set()
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("id"):
                ids.add(str(row["id"]))
            elif isinstance(row, str) and row:
                ids.add(row)
    ids.add(DEFAULT_MODEL)
    return ids


def load_model_ids(fetch: FetchGet | None = None, token: str | None = None) -> set[str]:
    """Model ids, cached for a few minutes when using the real network call."""
    if fetch is not None:
        return fetch_model_ids(fetch, token)
    now = time.time()
    cached = _models_cache.get("ids")
    if isinstance(cached, set) and now - float(_models_cache.get("at") or 0) < MODELS_CACHE_SECONDS:
        return cached
    ids = fetch_model_ids(None, token)
    _models_cache["at"] = now
    _models_cache["ids"] = ids
    return ids


def default_post(url: str, body: dict, headers: dict, timeout: int) -> Iterator[str]:
    """POST a streaming chat completion and yield the response lines."""
    token = ""
    auth = headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[7:]
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:500]
        message, status = friendly_http(exc.code, raw, token)
        raise HostedError(message, status) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HostedError(
            f"OpenRouter could not be reached. Check the internet connection. ({exc})",
            502,
        ) from exc

    def lines() -> Iterator[str]:
        try:
            while True:
                raw = response.readline()
                if not raw:
                    break
                yield raw.decode("utf-8", errors="replace").rstrip("\r\n")
        finally:
            response.close()

    return lines()


def open_stream(
    messages: list[dict],
    *,
    model: str,
    token: str,
    max_tokens: int,
    post: PostFn | None = None,
) -> Iterator[str]:
    """Open the provider stream. Raises HostedError before any token is read."""
    if not token:
        raise HostedError("Hosted chat needs an OpenRouter API key. " + SETUP_NOTE, 409)
    if not model:
        raise HostedError("No hosted model was chosen.", 400)
    sender = post or default_post
    return sender(
        API_URL,
        {
            "model": model,
            "messages": messages,
            "max_tokens": max(1, int(max_tokens)),
            "stream": True,
        },
        _headers(token),
        CHAT_TIMEOUT,
    )


def _delta_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta") or {}
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def iter_deltas(lines: Iterator[str], stop_event=None) -> Iterator[str]:
    """Yield text pieces from OpenAI-style server-sent events."""
    for line in lines:
        if stop_event is not None and stop_event.is_set():
            return
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("error"):
            err = payload["error"]
            text = err.get("message") if isinstance(err, dict) else str(err)
            code = err.get("code") if isinstance(err, dict) else 502
            try:
                code_int = int(code)
            except (TypeError, ValueError):
                code_int = 502
            message, status = friendly_http(code_int, json.dumps({"error": {"message": text}}))
            raise HostedError(message, status)
        if isinstance(payload, dict):
            piece = _delta_text(payload)
            if piece:
                yield piece
