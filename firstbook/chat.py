"""Offline chat for The First Book: system prompt, history trim, status.

The page and API live in ``firstbook.web``. This module has no Flask and no
llama_cpp import, so tests can exercise the prompt, the trim, and the
"not ready" messages without a model.
"""

from __future__ import annotations

from typing import Callable

# Shown on /chat and in /api/chat/status when llama-cpp-python is missing.
# Matches the prebuilt-wheel command in requirements-optional.txt.
INSTALL_COMMAND = (
    "pip install llama-cpp-python --prefer-binary \\\n"
    "  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
)

SYSTEM_PROMPT = (
    "You are The First Book, an offline assistant for UK survival and "
    "self-sufficiency. You help with practical skills: off-grid power, water, "
    "food, shelter, repair, navigation and first aid. Be practical and concise. "
    "Be cautious about safety, especially electrical work, water, and medical "
    "advice: warn about real dangers, do not invent procedures you are unsure "
    "of, and say when a qualified person or official guidance is needed. If you "
    "do not know, say so. You have no live network; do not pretend to look "
    "things up online."
)

DEFAULT_CHAT_MAX_TOKENS = 384
DEFAULT_N_CTX = 2048
# Leave a little room for chat-template special tokens around the messages.
TEMPLATE_RESERVE = 32
ROLES = {"user", "assistant"}


def normalize_history(raw, max_chars: int) -> list[dict]:
    """Keep only user/assistant turns with string content, clipped to max_chars."""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = item.get("content")
        if role not in ROLES or not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        out.append({"role": role, "content": text})
    return out


def build_messages(history: list[dict]) -> list[dict]:
    """System prompt first, then the conversation. The client cannot replace the system turn."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history]


def trim_messages(
    messages: list[dict],
    count_tokens: Callable[[list[dict]], int],
    n_ctx: int,
    max_tokens: int,
    reserve: int = TEMPLATE_RESERVE,
) -> list[dict]:
    """Drop the oldest conversation turns until the prompt plus a reply fits in n_ctx.

    The system turn is always kept. The latest user message is kept too; if it
    still does not fit, its content is cut from the start (the question's end
    is usually what matters).
    """
    budget = max(1, int(n_ctx) - int(max_tokens) - int(reserve))
    if not messages:
        return messages
    system = [m for m in messages if m.get("role") == "system"][:1]
    rest = [m for m in messages if m.get("role") != "system"]
    if not rest:
        return system

    def fits(chunk: list[dict]) -> bool:
        return count_tokens(system + chunk) <= budget

    kept = list(rest)
    while len(kept) > 1 and not fits(kept):
        kept.pop(0)
    if fits(kept):
        return system + kept

    last = dict(kept[-1])
    text = last.get("content") or ""
    # Binary shrink the last turn's text until it fits, or a short stub remains.
    lo, hi = 0, len(text)
    best = text[: max(1, min(64, len(text)))]
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[-mid:] if mid else text[:1]
        last["content"] = candidate
        if fits([last]):
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    last["content"] = best
    return system + [last]


def is_chat_entry(entry: dict) -> bool:
    return (
        entry.get("backend") == "llama_cpp"
        and entry.get("task") == "text-generation"
        and entry.get("status") in {"available", "downloadable"}
    )


def status(
    *,
    backend_available: bool,
    settings,
    loaded: bool,
    catalog_data: dict,
    installed_ids: set[str],
    chat_max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
    n_ctx: int = DEFAULT_N_CTX,
) -> dict:
    """Describe whether /chat can answer, and what to show if it cannot."""
    chat_models = []
    active_id = None
    for entry in catalog_data.get("models") or []:
        if not isinstance(entry, dict) or not is_chat_entry(entry):
            continue
        row = {
            "id": entry["id"],
            "name": entry.get("name") or entry["id"],
            "installed": entry["id"] in installed_ids,
        }
        chat_models.append(row)
        if (
            entry.get("hf_id") == getattr(settings, "model_id", None)
            and entry.get("revision") == getattr(settings, "model_revision", None)
            and entry.get("backend") == getattr(settings, "model_backend", None)
        ):
            active_id = entry["id"]
            row["active"] = True
        else:
            row["active"] = False

    ready = False
    reason = "ok"
    message = ""
    if not backend_available:
        reason = "missing_backend"
        message = (
            "Chat needs llama-cpp-python, which is not installed. "
            "Install it with this command, restart the app, then come back:"
        )
    elif not any(m["installed"] for m in chat_models):
        reason = "no_chat_model"
        message = (
            "No chat model is on disk yet. Open the model page, download a GGUF "
            "chat model (Gemma 3 270M is a good start), and select Use this model."
        )
    else:
        active = next((m for m in chat_models if m["id"] == active_id), None)
        if active is None or not active["installed"]:
            reason = "need_chat_model"
            message = (
                "The active model is not a GGUF chat model. Open the model page and "
                "select a downloaded chat model, then come back."
            )
        else:
            ready = True
            message = "Ready."

    return {
        "ready": ready,
        "reason": reason,
        "message": message,
        "install_command": INSTALL_COMMAND,
        "backend_available": backend_available,
        "active_chat_model": active_id,
        "loaded": loaded,
        "chat_models": chat_models,
        "chat_max_tokens": chat_max_tokens,
        "n_ctx": n_ctx,
        "models_url": "/",
    }
