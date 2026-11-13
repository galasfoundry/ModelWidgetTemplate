"""Project-relative paths and weight lookup for the model widget.

Locations come from the directory that contains this file, not from the
current working directory and not from any one machine. Environment
variables, ``config/.env``, and an optional project-root ``.env`` can
override them.

Weights use the Hugging Face cache layout under ``weights/``:

    weights/models--<org>--<name>/snapshots/<revision>/<file>

``install_weights.py`` and the app both call :func:`ensure_weights`. An
existing snapshot under ``weights/`` or in the Hugging Face hub cache is
reused. A ``MODEL_WEIGHTS`` override is used as given and never triggers a
download.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SKIP_NAMES = {".gitattributes", "README.md", "readme.md", "LICENSE", "LICENSE.txt"}
_SKIP_SUFFIXES = (
    ".md",
    ".h5",
    ".msgpack",
    ".ot",
    ".onnx",
    ".tflite",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
)
_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def load_dotenv(dotenv_path: Path | None = None) -> None:
    """Load KEY=VALUE lines into the environment.

    Existing variables win, so a shell export overrides the file. Missing
    files are ignored. This is enough for a local .env and avoids an extra
    dependency.
    """
    path = dotenv_path if dotenv_path is not None else PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_project_env() -> None:
    """Load ``config/.env``, then the project-root ``.env``.

    The first file to set a key wins. Variables already exported in the
    shell win over both files.
    """
    load_dotenv(PROJECT_ROOT / "config" / ".env")
    load_dotenv(PROJECT_ROOT / ".env")


def project_path(value: str | Path) -> Path:
    """Resolve a path against the project root when it is not already absolute."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_path(env_var: str, default: Path) -> Path:
    """Return the path in env_var, or default. Relative values use the project root."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default.resolve()
    return project_path(raw)


def huggingface_hub_dir() -> Path:
    """Hugging Face hub cache, honoring HF_HOME and HUGGINGFACE_HUB_CACHE."""
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_dir_name(repo_id: str) -> str:
    """Cache folder name Hugging Face uses for a repo id."""
    return "models--" + repo_id.replace("/", "--")


def select_filenames(files: list[str]) -> list[str]:
    """Pick the files ``hf_hub_download`` should store for a model repo.

    Framework duplicates (TensorFlow, Flax, ONNX, Rust) are skipped. When a
    safetensors checkpoint is present, the pickle ``.bin`` weights are skipped
    too.
    """
    chosen: list[str] = []
    for name in files:
        base = name.rsplit("/", 1)[-1]
        lowered = name.lower()
        if base in _SKIP_NAMES or name.endswith(_SKIP_SUFFIXES):
            continue
        if lowered.startswith(("flax_model", "tf_model", "rust_model", "onnx/")):
            continue
        if "/onnx/" in lowered or lowered.startswith("onnx/"):
            continue
        chosen.append(name)
    has_safetensors = any(name.endswith(".safetensors") for name in chosen)
    if has_safetensors:
        chosen = [name for name in chosen if not name.endswith(".bin")]
    return chosen


def _has_weights(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    if any((directory / name).is_file() for name in _WEIGHT_FILENAMES):
        return True
    return any(path.is_file() for path in directory.glob("*.safetensors"))


def _read_ref(cache_root: Path, repo_id: str, revision: str) -> str | None:
    ref_path = cache_root / model_dir_name(repo_id) / "refs" / revision
    if not ref_path.is_file():
        return None
    try:
        text = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _snapshot_dirs(cache_root: Path, repo_id: str) -> list[Path]:
    root = cache_root / model_dir_name(repo_id) / "snapshots"
    if not root.is_dir():
        return []
    found: list[Path] = []
    for candidate in root.iterdir():
        if candidate.is_dir() and _has_weights(candidate):
            found.append(candidate)
    return found


def _rank_snapshots(snapshots: list[Path], revision: str) -> list[Path]:
    preferred_names = {revision} if revision else set()

    def sort_key(path: Path) -> tuple[int, float]:
        preferred = 0 if path.name in preferred_names else 1
        try:
            newest_first = -path.stat().st_mtime
        except OSError:
            newest_first = 0.0
        return (preferred, newest_first)

    ordered = sorted(snapshots, key=sort_key)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in ordered:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def find_local_snapshot(
    repo_id: str,
    revision: str,
    weights_dir: Path,
    hub_dir: Path | None = None,
) -> Path | None:
    """Return an on-disk snapshot for ``repo_id``, or None. This never downloads.

    Search order when ``MODEL_WEIGHTS`` is unset:

    1. The pinned revision under ``weights/``.
    2. The pinned revision in the Hugging Face hub cache.
    3. Any other snapshot of this repo under ``weights/``.
    4. Any other snapshot of this repo in the hub cache.

    A branch name such as ``main`` is matched through the cache ``refs`` file
    as well as the snapshot folder name.
    """
    hub = huggingface_hub_dir() if hub_dir is None else hub_dir
    # weights/ is searched before the hub cache. An exact revision wins
    # inside a pool; any other snapshot of this repo is the fallback.
    for root in (weights_dir, hub):
        names = {revision} if revision else set()
        if revision:
            ref = _read_ref(root, repo_id, revision)
            if ref:
                names.add(ref)
        snapshots = _snapshot_dirs(root, repo_id)
        if not snapshots:
            continue
        exact = [item for item in snapshots if revision and item.name in names]
        if exact:
            return _rank_snapshots(exact, revision)[0].resolve()
        if not revision:
            return _rank_snapshots(snapshots, revision)[0].resolve()
    for root in (weights_dir, hub):
        snapshots = _snapshot_dirs(root, repo_id)
        if not snapshots:
            continue
        chosen = _rank_snapshots(snapshots, revision)[0].resolve()
        if revision and chosen.name != revision:
            print(
                f"Revision {revision} was not found locally for {repo_id}. "
                f"Reusing snapshot {chosen.name}.",
                flush=True,
            )
        return chosen
    return None


def _snapshot_from_override(raw: str) -> Path | None:
    """Resolve MODEL_WEIGHTS to a directory that holds a checkpoint."""
    path = project_path(raw)
    if path.is_dir():
        if _has_weights(path):
            return path.resolve()
        nested = []
        snapshots = path / "snapshots"
        if snapshots.is_dir():
            nested = [item for item in snapshots.iterdir() if item.is_dir() and _has_weights(item)]
        else:
            for candidate in path.glob("models--*/snapshots/*"):
                if candidate.is_dir() and _has_weights(candidate):
                    nested.append(candidate)
        if nested:
            return _rank_snapshots(nested, "")[0].resolve()
        return None
    if path.is_file():
        parent = path.parent
        if _has_weights(parent) or (parent / "config.json").is_file():
            return parent.resolve()
        return None
    return None


def resolve_revision(repo_id: str, revision: str) -> str:
    """Return a commit SHA. An empty revision becomes the repo's current default."""
    if revision and _SHA_RE.match(revision):
        return revision
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve a model revision. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    info = HfApi().model_info(repo_id, revision=revision or None)
    sha = getattr(info, "sha", "") or ""
    if not sha:
        raise RuntimeError(f"Could not resolve a commit for {repo_id}.")
    if not revision:
        print(
            f"MODEL_REVISION is unset for {repo_id}. Using the current default "
            f"commit {sha}. Pin that value in config/.env to keep this checkout fixed.",
            flush=True,
        )
    return sha


def download_weights(repo_id: str, revision: str, weights_dir: Path | None = None) -> Path:
    """Download the model into the project weights folder with ``hf_hub_download``.

    ``cache_dir`` is the weights folder, so Hugging Face writes
    ``weights/models--<repo>/snapshots/<revision>/``. The revision stored on
    disk is a commit SHA.
    """
    destination = WEIGHTS_DIR if weights_dir is None else Path(weights_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pinned = resolve_revision(repo_id, revision)
    print(
        f"Model weights not found. Downloading {repo_id} (revision {pinned}) into {destination}",
        flush=True,
    )
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download model weights. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        filenames = select_filenames(list_repo_files(repo_id, revision=pinned))
        if not filenames:
            raise RuntimeError(f"{repo_id} at {pinned} has no model files to download.")
        downloaded = None
        for filename in filenames:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=pinned,
                cache_dir=str(destination),
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {repo_id} revision {pinned} into {destination}. "
            "Check the network connection and the repo id. Gated repos need HF_TOKEN. "
            f"Original error: {exc}"
        ) from exc

    if downloaded is None:
        raise RuntimeError(f"No files were downloaded for {repo_id}.")
    snapshot = snapshot_from_download(Path(downloaded))
    print(f"Downloaded model weights to {snapshot}", flush=True)
    return snapshot


def snapshot_from_download(downloaded: Path) -> Path:
    """Return the ``snapshots/<revision>/`` directory for an ``hf_hub_download`` path.

    The file is usually a symlink into ``blobs/``. Resolving the file first
    lands in that blob directory, which holds the bytes without ``config.json``
    or the tokenizer beside them.
    """
    return Path(downloaded).parent.resolve()


def ensure_weights(repo_id: str, revision: str, weights_dir: Path | None = None) -> Path:
    """Return the snapshot directory to load, downloading it on first run.

    A MODEL_WEIGHTS override is used as-is. A missing override is an error
    and does not trigger a download to a different folder.
    """
    destination = WEIGHTS_DIR if weights_dir is None else Path(weights_dir)
    raw = os.environ.get("MODEL_WEIGHTS", "").strip()
    if raw:
        found = _snapshot_from_override(raw)
        if found is None:
            missing = project_path(raw)
            raise FileNotFoundError(
                f"MODEL_WEIGHTS is set but no checkpoint was found at {missing}. "
                "Point MODEL_WEIGHTS at a checkpoint file, a snapshot directory, "
                "or a Hugging Face cache folder that contains one."
            )
        return found

    found = find_local_snapshot(repo_id, revision, destination)
    if found is not None:
        return found
    return download_weights(repo_id, revision, destination)
