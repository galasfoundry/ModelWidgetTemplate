"""Search Hugging Face for GGUF chat models that this machine can run.

A 'hot apocalypse' grab alongside the curated catalog: the public Hub API
is queried for llama.cpp-compatible GGUF files, each candidate is labelled
fits / tight / won't fit from this machine's RAM and free disk, and a pick
is written to ``config/models_user.json`` (gitignored), never the curated
catalog. Search results are cached on disk so a previous search still
shows when Hugging Face cannot be reached.

No token is required for public repos. ``HF_TOKEN`` (or
``HUGGING_FACE_HUB_TOKEN``) is used only for gated repos.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from modelwidget import catalog

HF_API = "https://huggingface.co/api/models"
USER_AGENT = "TheFirstBook/0.1 (offline model discovery)"
DEFAULT_QUERY = "1b instruct"
DEFAULT_LIMIT = 25
RESULT_LIMIT = 15
TREE_TIMEOUT = 20
SEARCH_TIMEOUT = 20

# Estimated process RSS, calibrated against measured chat loads on this app
# (not just the GGUF mapping): Gemma 3 270M Q4_K_M, 241 MB file, 672–728 MB
# RSS; Qwen 2.5 1.5B Q4_K_M, 940 MB file, 2091 MB RSS (2381 MB peak while
# switching). size × 1.5 + 700 MB covers both and stays slightly above the
# 1.5B measurement. The switch peak is two models overlapping, so it is not
# the target for a single load.
RAM_FACTOR = 1.5
RAM_OVERHEAD_MB = 700
RAM_FIT_FRACTION = 0.70  # must stay under this share of total RAM
RAM_COMFORT_FRACTION = 0.50  # below this is 'fits'; between this and 70% is 'tight'
DISK_MARGIN = 1.15

PREFERRED_QUANTS = (
    "Q4_K_M",
    "Q4_K_S",
    "Q4_0",
    "Q3_K_M",
    "Q3_K_L",
    "Q3_K_S",
    "IQ4_XS",
    "IQ4_NL",
    "IQ3_M",
)
# Only chosen when the estimate is comfortably under half of total RAM.
SPARE_QUANTS = ("Q5_K_M", "Q5_K_S", "Q5_0", "Q6_K", "Q8_0")

# Parameter counts at or above this are skipped without a file-tree fetch.
_LARGE_PARAM_RE = re.compile(r"(?:^|[^0-9])(1[1-9]|[2-9]\d|\d{3,})\s*B\b", re.I)
_QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:UD-)?"
    r"(?:Q[2-8](?:_[K01])?(?:_[SMLX]{1,2})?"
    r"|IQ[1-4](?:_[MNSXL]{1,3})?"
    r"|F16|BF16|FP16|F32))(?:[-_.]|$)",
    re.I,
)
_SKIP_FILE_RE = re.compile(
    r"(mmproj|mm-proj|projector|embed(?:ding)?|clip|eva-?clip|"
    r"encoder|lora|imatrix|000\d+-of-000\d+)",
    re.I,
)
_CHAT_HINTS = ("instruct", "chat", "-it-", "-it_", "-it/", "it-gguf", "assistant", "conversational")
_NOT_CHAT_HINTS = (
    "embed",
    "rerank",
    "whisper",
    "tts",
    "asr",
    "clip",
    "diffusion",
    "wav2vec",
    "audio.cpp",
    "sentence-similarity",
    "feature-extraction",
    "text-to-image",
    "automatic-speech",
    "text-to-audio",
    "fill-mask",
)
_NOT_CHAT_PIPELINES = {
    "sentence-similarity",
    "feature-extraction",
    "fill-mask",
    "text-to-image",
    "automatic-speech-recognition",
    "text-to-audio",
    "text-to-speech",
    "image-to-text",
    "image-text-to-text",
    "zero-shot-classification",
}


FetchFn = Callable[..., object]


class DiscoveryError(Exception):
    """A discovery request that failed. ``status`` is an HTTP code when set."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def user_catalog_path(catalog_file: Path | None = None) -> Path:
    base = Path(catalog_file) if catalog_file is not None else catalog.catalog_path()
    return base.with_name("models_user.json")


def cache_path(catalog_file: Path | None = None) -> Path:
    base = Path(catalog_file) if catalog_file is not None else catalog.catalog_path()
    return base.with_name("models_discover_cache.json")


def hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def fetch_json(url: str, token: str | None = None, timeout: int = SEARCH_TIMEOUT):
    """GET a JSON document from the Hub. Tests inject a stub of this function."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:200]
        raise DiscoveryError(
            f"Hugging Face returned HTTP {exc.code} for {url}: {body or exc.reason}",
            502,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
        raise DiscoveryError(f"Hugging Face could not be reached: {exc}", 502) from exc


def estimated_ram_mb(size_mb: float) -> float:
    return size_mb * RAM_FACTOR + RAM_OVERHEAD_MB


def assess_fit(size_mb: float, caps: dict) -> tuple[str, float, str]:
    """Return (label, estimated_ram_mb, reason). Labels: fits, tight, won't fit."""
    estimate = estimated_ram_mb(size_mb)
    reasons: list[str] = []
    label = "fits"

    total_gb = caps.get("ram_total_gb")
    if total_gb is None:
        label = "tight"
        reasons.append(f"RAM unknown; estimated {estimate:.0f} MB")
    else:
        ceiling = total_gb * 1024 * RAM_FIT_FRACTION
        comfort = total_gb * 1024 * RAM_COMFORT_FRACTION
        if estimate > ceiling:
            label = "won't fit"
            reasons.append(
                f"needs about {estimate:.0f} MB RAM, 70% of this machine is {ceiling:.0f} MB"
            )
        elif estimate > comfort:
            label = "tight"
            reasons.append(f"about {estimate:.0f} MB RAM on a {total_gb:.1f} GB machine")
        else:
            reasons.append(f"about {estimate:.0f} MB RAM, file is {size_mb:.0f} MB")

    free_mb = caps.get("disk_free_mb")
    if free_mb is not None and free_mb < size_mb * DISK_MARGIN:
        label = "won't fit"
        reasons.append(f"needs about {size_mb:.0f} MB disk with margin, {free_mb:.0f} MB free")

    return label, estimate, "; ".join(reasons)


def _quant_of(filename: str) -> str | None:
    match = _QUANT_RE.search(filename.replace(" ", "_"))
    return None if match is None else match.group(1).upper().replace("FP16", "F16")


def _usable_gguf(path: str) -> bool:
    name = Path(path).name
    if not name.lower().endswith(".gguf"):
        return False
    if _SKIP_FILE_RE.search(name):
        return False
    return True


def _looks_chat(repo_id: str, tags: list[str], pipeline: str | None, task: str) -> bool:
    if task != "chat":
        return True
    blob = " ".join([repo_id, pipeline or "", *tags]).lower()
    if any(hint in blob for hint in _NOT_CHAT_HINTS):
        return False
    if pipeline and pipeline.lower() in _NOT_CHAT_PIPELINES:
        return False
    if any(hint in blob for hint in _CHAT_HINTS):
        return True
    # A bare 'base' or pretrain repo with no instruct/chat tag is not a chat model.
    return False


def _needs_trust_remote_code(detail: dict) -> bool:
    tags = [str(tag).lower() for tag in (detail.get("tags") or [])]
    if "trust_remote_code" in tags or "custom_code" in tags:
        return True
    card = detail.get("cardData") or {}
    if card.get("trust_remote_code") is True:
        return True
    config = detail.get("config") or {}
    if isinstance(config, dict) and config.get("trust_remote_code") is True:
        return True
    return False


def _is_gated(value) -> bool:
    return bool(value) and value not in (False, "false", "False")


def _license_of(detail: dict, tags: list[str] | None = None) -> str | None:
    card = detail.get("cardData") or {}
    license_id = card.get("license") or detail.get("license")
    if license_id:
        return str(license_id)
    for tag in tags or detail.get("tags") or []:
        text = str(tag)
        if text.lower().startswith("license:"):
            return text.split(":", 1)[1]
    return None


def _large_for_machine(repo_id: str) -> bool:
    return _LARGE_PARAM_RE.search(repo_id.replace("-", " ").replace("_", " ")) is not None


def _pick_file(files: list[dict], caps: dict) -> dict | None:
    """Choose the best quant that fits this machine from ``{path, size}`` files."""
    scored: list[tuple[int, dict, str, float, str]] = []
    for item in files:
        path = item["path"]
        if not _usable_gguf(path):
            continue
        size = item.get("size")
        if not size:
            continue
        quant = _quant_of(Path(path).name) or "UNKNOWN"
        size_mb = size / (1024 * 1024)
        label, estimate, reason = assess_fit(size_mb, caps)
        scored.append((0, {**item, "quant": quant, "size_mb": round(size_mb, 1)}, label, estimate, reason))

    if not scored:
        return None

    def take(names: tuple[str, ...], allowed: set[str]) -> dict | None:
        for preferred in names:
            for _, item, label, estimate, reason in scored:
                if item["quant"] == preferred and label in allowed:
                    return {
                        "path": item["path"],
                        "size": item["size"],
                        "size_mb": item["size_mb"],
                        "quant": item["quant"],
                        "fit": label,
                        "estimated_ram_mb": round(estimate, 1),
                        "reason": reason,
                    }
        return None

    # Prefer Q4_K_M (even if tight), then Q4_0 / Q3. Q5 / Q8 only when comfortable.
    picked = (
        take(PREFERRED_QUANTS, {"fits", "tight"})
        or take(SPARE_QUANTS, {"fits"})
        or take(PREFERRED_QUANTS, {"won't fit"})
        or take(SPARE_QUANTS, {"tight", "won't fit"})
    )
    if picked:
        return picked
    # Last resort: the smallest remaining file (F16 etc.).
    _, item, label, estimate, reason = min(scored, key=lambda row: row[1]["size"])
    return {
        "path": item["path"],
        "size": item["size"],
        "size_mb": item["size_mb"],
        "quant": item["quant"],
        "fit": label,
        "estimated_ram_mb": round(estimate, 1),
        "reason": reason,
    }


def _slug(hf_id: str, quant: str) -> str:
    name = hf_id.split("/", 1)[-1]
    raw = f"{name}-{quant}".lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    slug = re.sub(r"-gguf-", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return f"user-{slug}"[:80]


def _display_name(hf_id: str, quant: str) -> str:
    name = hf_id.split("/", 1)[-1]
    name = re.sub(r"-?GGUF$", "", name, flags=re.I)
    name = name.replace("-", " ").replace("_", " ")
    return f"{name} (GGUF {quant})"


def _search_url(query: str, limit: int) -> str:
    params = {
        "filter": "gguf",
        "search": query,
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
    }
    return f"{HF_API}?{urllib.parse.urlencode(params)}"


def _detail_url(repo_id: str) -> str:
    return f"{HF_API}/{urllib.parse.quote(repo_id, safe='/')}"


def _tree_url(repo_id: str, revision: str) -> str:
    return (
        f"{HF_API}/{urllib.parse.quote(repo_id, safe='/')}"
        f"/tree/{urllib.parse.quote(revision, safe='/')}"
        "?recursive=1"
    )


def _flatten_tree(nodes, prefix: str = "") -> list[dict]:
    files: list[dict] = []
    if not isinstance(nodes, list):
        return files
    for node in nodes:
        if not isinstance(node, dict):
            continue
        path = node.get("path") or (f"{prefix}/{node.get('name')}" if node.get("name") else "")
        kind = node.get("type") or "file"
        if kind == "file":
            files.append({"path": path, "size": node.get("size") or 0})
        elif kind == "directory" and isinstance(node.get("items"), list):
            files.extend(_flatten_tree(node["items"], path))
    return files


def _already_in_catalog(hf_id: str, gguf_file: str, catalog_file: Path | None) -> dict | None:
    try:
        data = catalog.load_catalog(catalog_file)
    except catalog.SelectionError:
        return None
    needle = Path(gguf_file).name
    for entry in data["models"]:
        if entry.get("hf_id") == hf_id and Path(entry.get("gguf_file") or "").name == needle:
            return entry
    return None


def search_models(
    query: str,
    caps: dict,
    *,
    task: str = "chat",
    token: str | None = None,
    limit: int = DEFAULT_LIMIT,
    fetch: FetchFn | None = None,
    catalog_file: Path | None = None,
) -> dict:
    """Query the Hub and return labelled candidates. Raises DiscoveryError on network failure."""
    fetch = fetch or fetch_json
    query = (query or "").strip() or DEFAULT_QUERY
    token = token if token is not None else hf_token()
    listings = fetch(_search_url(query, limit), token=token, timeout=SEARCH_TIMEOUT)
    if not isinstance(listings, list):
        raise DiscoveryError("Hugging Face returned an unexpected search response.")

    results: list[dict] = []
    skipped: list[dict] = []
    for listing in listings:
        if not isinstance(listing, dict) or not listing.get("id"):
            continue
        repo_id = listing["id"]
        tags = [str(t) for t in (listing.get("tags") or [])]
        pipeline = listing.get("pipeline_tag")
        if "gguf" not in [t.lower() for t in tags] and not repo_id.lower().endswith("-gguf"):
            skipped.append({"id": repo_id, "reason": "not a GGUF repo"})
            continue
        if not _looks_chat(repo_id, tags, pipeline, task):
            skipped.append({"id": repo_id, "reason": "not a chat or instruct model"})
            continue
        if _large_for_machine(repo_id):
            skipped.append({"id": repo_id, "reason": "parameter count is too large for this kind of machine"})
            continue

        try:
            detail = fetch(_detail_url(repo_id), token=token, timeout=TREE_TIMEOUT)
        except DiscoveryError as exc:
            skipped.append({"id": repo_id, "reason": str(exc)})
            continue
        if not isinstance(detail, dict):
            continue

        gated = _is_gated(detail.get("gated", listing.get("gated")))
        if gated and not token:
            skipped.append({"id": repo_id, "reason": "gated: set HF_TOKEN to include this repo"})
            continue
        if _needs_trust_remote_code(detail):
            skipped.append({"id": repo_id, "reason": "card requires trust_remote_code"})
            continue

        revision = detail.get("sha") or "main"
        try:
            tree = fetch(_tree_url(repo_id, revision), token=token, timeout=TREE_TIMEOUT)
        except DiscoveryError as exc:
            skipped.append({"id": repo_id, "reason": str(exc)})
            continue
        files = _flatten_tree(tree)
        picked = _pick_file(files, caps)
        if picked is None:
            skipped.append({"id": repo_id, "reason": "no usable GGUF file in the repo"})
            continue

        licence = _license_of(detail, tags)
        existing = _already_in_catalog(repo_id, picked["path"], catalog_file)
        candidate = {
            "hf_id": repo_id,
            "name": _display_name(repo_id, picked["quant"]),
            "revision": revision,
            "gguf_file": Path(picked["path"]).name,
            "gguf_path": picked["path"],
            "quant": picked["quant"],
            "approx_size_mb": picked["size_mb"],
            "estimated_ram_mb": picked["estimated_ram_mb"],
            "fit": picked["fit"],
            "reason": picked["reason"],
            "downloads": listing.get("downloads") or 0,
            "likes": listing.get("likes") or 0,
            "license": licence,
            "gated": gated,
            "pipeline_tag": pipeline,
            "suggested_id": _slug(repo_id, picked["quant"]),
            "in_catalog": None if existing is None else existing["id"],
        }
        results.append(candidate)
        if len(results) >= RESULT_LIMIT:
            break

    order = {"fits": 0, "tight": 1, "won't fit": 2}
    results.sort(key=lambda row: (order.get(row["fit"], 9), -row["downloads"], -row["likes"]))
    return {
        "query": query,
        "task": task,
        "results": results,
        "skipped": skipped,
        "online": True,
        "cached": False,
    }


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_cache(payload: dict, path: Path) -> Path:
    document = dict(payload)
    document["saved_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(path, document)
    return path


def load_cache(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return None
    return data


def cache_age_seconds(saved_at: str | None, now: datetime | None = None) -> int | None:
    if not saved_at:
        return None
    try:
        stamp = datetime.fromisoformat(saved_at)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0, int((now - stamp).total_seconds()))


def discover(
    query: str,
    caps: dict,
    *,
    task: str = "chat",
    token: str | None = None,
    fetch: FetchFn | None = None,
    catalog_file: Path | None = None,
    cache_file: Path | None = None,
) -> dict:
    """Live search with a disk cache fallback when Hugging Face is unreachable."""
    cache_file = Path(cache_file) if cache_file is not None else cache_path(catalog_file)
    requested = (query or "").strip()
    try:
        payload = search_models(
            requested,
            caps,
            task=task,
            token=token,
            fetch=fetch,
            catalog_file=catalog_file,
        )
        payload["machine"] = {
            "ram_total_gb": caps.get("ram_total_gb"),
            "disk_free_mb": caps.get("disk_free_mb"),
        }
        save_cache(payload, cache_file)
        payload["cached_at"] = datetime.now(timezone.utc).isoformat()
        payload["cache_age_seconds"] = 0
        payload["error"] = None
        return payload
    except DiscoveryError as exc:
        cached = load_cache(cache_file)
        if cached is None:
            return {
                "query": requested or DEFAULT_QUERY,
                "task": task,
                "results": [],
                "skipped": [],
                "online": False,
                "cached": False,
                "cached_at": None,
                "cache_age_seconds": None,
                "error": str(exc),
                "machine": {
                    "ram_total_gb": caps.get("ram_total_gb"),
                    "disk_free_mb": caps.get("disk_free_mb"),
                },
            }
        age = cache_age_seconds(cached.get("saved_at"))
        cached_query = cached.get("query") or DEFAULT_QUERY
        note = str(exc)
        if requested and cached_query.lower() != requested.lower():
            note = (
                f"{note} Showing cached results for {cached_query!r}, "
                f"not the new search {requested!r}."
            )
        return {
            "query": cached_query,
            "requested": requested or cached_query,
            "task": cached.get("task") or task,
            "results": cached.get("results") or [],
            "skipped": cached.get("skipped") or [],
            "online": False,
            "cached": True,
            "cached_at": cached.get("saved_at"),
            "cache_age_seconds": age,
            "error": note,
            "machine": cached.get("machine") or {
                "ram_total_gb": caps.get("ram_total_gb"),
                "disk_free_mb": caps.get("disk_free_mb"),
            },
        }


def candidate_to_entry(candidate: dict, catalog_file: Path | None = None) -> dict:
    """Turn a discovery row into a catalog entry for ``models_user.json``."""
    if candidate.get("fit") == "won't fit":
        raise DiscoveryError(
            f"{candidate.get('name') or candidate.get('hf_id')} is labelled "
            f"'won't fit' on this machine: {candidate.get('reason')}.",
            409,
        )
    hf_id = str(candidate["hf_id"]).strip()
    gguf_file = Path(str(candidate.get("gguf_file") or candidate.get("gguf_path") or "")).name
    if not hf_id or not gguf_file:
        raise DiscoveryError("A candidate needs hf_id and a GGUF filename.", 400)
    existing = _already_in_catalog(hf_id, gguf_file, catalog_file)
    if existing is not None:
        raise DiscoveryError(
            f"That file is already in the catalog as {existing['id']}.",
            409,
        )
    quant = candidate.get("quant") or _quant_of(gguf_file) or "Q4_K_M"
    size_mb = float(candidate.get("approx_size_mb") or 0)
    estimate = float(candidate.get("estimated_ram_mb") or estimated_ram_mb(size_mb))
    entry_id = str(candidate.get("suggested_id") or _slug(hf_id, quant))
    # Avoid colliding with a different user or curated id.
    try:
        data = catalog.load_catalog(catalog_file)
        taken = {item["id"] for item in data["models"]} | {
            item["id"] for item in data.get("planned") or []
        }
    except catalog.SelectionError:
        taken = set()
    if entry_id in taken:
        suffix = str(candidate.get("revision") or "x")[:8]
        entry_id = f"{entry_id}-{suffix}"[:80]
    min_ram = max(1.0, round(estimate / 1024, 1))
    licence = candidate.get("license")
    return {
        "id": entry_id,
        "name": candidate.get("name") or _display_name(hf_id, quant),
        "hf_id": hf_id,
        "revision": candidate.get("revision") or "main",
        "gguf_file": gguf_file,
        "task": "text-generation",
        "backend": "llama_cpp",
        "category": "llm",
        "status": "downloadable",
        "approx_size_mb": round(size_mb) if size_mb else None,
        "min_ram_gb": min_ram,
        "license": licence,
        "description": (
            f"Added from Hugging Face discovery. {hf_id} / {gguf_file}."
        ),
        "theme_tags": ["llm", "chat", "gguf", "discovered"],
        "source": "user",
    }


def load_user_models(path: Path) -> list[dict]:
    """Read ``models_user.json``. A missing file is empty; bad entries are skipped."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError):
        return []
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for entry in models:
        if not isinstance(entry, dict):
            continue
        missing = [key for key in catalog.REQUIRED_FIELDS if not str(entry.get(key) or "").strip()]
        if missing or entry["id"] in seen:
            continue
        seen.add(entry["id"])
        item = dict(entry)
        item["source"] = "user"
        if not item.get("status"):
            item["status"] = "downloadable"
        out.append(item)
    return out


def write_user_models(path: Path, models: list[dict]) -> Path:
    clean = []
    for entry in models:
        item = dict(entry)
        item["source"] = "user"
        clean.append(item)
    _atomic_write(
        path,
        {
            "version": 1,
            "description": "Models added through Discover. Not committed to git.",
            "models": clean,
        },
    )
    return path


def add_candidate(candidate: dict, catalog_file: Path | None = None, user_path: Path | None = None) -> dict:
    """Append a discovery pick to the user catalog. Returns the new entry."""
    user_path = Path(user_path) if user_path is not None else user_catalog_path(catalog_file)
    entry = candidate_to_entry(candidate, catalog_file)
    models = load_user_models(user_path)
    if any(item["id"] == entry["id"] for item in models):
        raise DiscoveryError(f"{entry['id']} is already in the user catalog.", 409)
    models.append(entry)
    write_user_models(user_path, models)
    return entry


def remove_user_model(
    model_id: str,
    *,
    catalog_file: Path | None = None,
    user_path: Path | None = None,
    active_id: str | None = None,
) -> dict:
    """Remove a user-added catalog entry. Weights on disk are left alone."""
    user_path = Path(user_path) if user_path is not None else user_catalog_path(catalog_file)
    models = load_user_models(user_path)
    found = next((item for item in models if item["id"] == model_id), None)
    if found is None:
        # Distinguish curated ids from unknown ids when we can.
        try:
            data = catalog.load_catalog(catalog_file)
            curated = next((item for item in data["models"] if item["id"] == model_id), None)
            if curated is not None and curated.get("source") != "user":
                raise DiscoveryError("The curated catalog cannot be edited this way.", 409)
        except catalog.SelectionError:
            pass
        raise DiscoveryError(f"No user-added model with id {model_id!r}.", 404)
    if active_id and active_id == model_id:
        raise DiscoveryError("Switch away from this model before removing it.", 409)
    remaining = [item for item in models if item["id"] != model_id]
    write_user_models(user_path, remaining)
    return found


def candidate_from_cache(index: int, cache_file: Path, catalog_file: Path | None = None) -> dict:
    """1-based index into the last cached search."""
    cached = load_cache(cache_file)
    if cached is None:
        raise DiscoveryError("No cached search to add from. Run a search first.", 404)
    results = cached.get("results") or []
    if index < 1 or index > len(results):
        raise DiscoveryError(f"No cached result number {index}.", 400)
    return results[index - 1]
