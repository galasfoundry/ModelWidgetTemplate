"""Download catalog models into ``weights/`` in the background, with progress.

Downloading is separate from activating. ``DownloadManager.submit(entry)``
queues a job and returns at once; one worker thread downloads jobs in order
(one at a time is kinder to a 4 GB laptop and a home connection), and
``status()`` reports each job:

    {"id": ..., "state": "queued" | "downloading" | "done" | "error",
     "bytes_done": int, "bytes_total": int | None, "percent": float | None,
     "files_done": int, "files_total": int, "current_file": str | None,
     "message": str, "error": str | None, "snapshot": str | None, ...}

Files go to ``weights/models--<org>--<name>/snapshots/<revision>/`` through
``hf_hub_download(cache_dir=weights/)``, the same layout the loader reads.
An interrupted download is simply run again. With huggingface_hub 2.x most
repos transfer through Xet, which writes a uniquely named ``*.incomplete``
file and starts that file over rather than resuming it; leftover partials
older than a minute are deleted first so a laptop disk does not fill up
with them. Files already finished in a multi-file model are kept. A model already on disk finishes instantly and never
touches the network.

GGUF entries (``gguf_file`` set) download that one file only, never the
whole repo, which usually holds many quantisations.

The same :func:`download_entry` powers ``scripts/download_models.py``.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import paths

_MB = 1024**2


class DownloadError(Exception):
    """A download that could not run. The message is meant for people."""


@dataclass
class Job:
    id: str
    name: str
    hf_id: str
    revision: str
    state: str = "queued"
    bytes_done: int = 0
    bytes_total: int | None = None
    files_done: int = 0
    files_total: int = 0
    current_file: str | None = None
    message: str = "Waiting to start."
    error: str | None = None
    snapshot: str | None = None
    downloaded: bool = False
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        total = self.bytes_total
        data["percent"] = round(100.0 * self.bytes_done / total, 1) if total else None
        data["mb_done"] = round(self.bytes_done / _MB, 1)
        data["mb_total"] = None if total is None else round(total / _MB, 1)
        end = self.finished_at or time.time()
        data["seconds"] = None if self.started_at is None else round(end - self.started_at, 1)
        return data


# ---------------------------------------------------------------------------
# Planning and downloading one entry (no threads; used by the CLI too)
# ---------------------------------------------------------------------------

def _hub():
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - requirements.txt has it
        raise DownloadError(
            "huggingface_hub is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return huggingface_hub


def offline_reason() -> str | None:
    """Why huggingface_hub will refuse to go online, if it will."""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}:
            return (
                f"{key} is set, so Hugging Face downloads are switched off. "
                f"Unset it (or set {key}=0) to download, then set it again for offline use."
            )
    return None


def plan_files(entry: dict) -> list[tuple[str, int | None]]:
    """The (filename, size) pairs to fetch for a catalog entry.

    One API call (``model_info`` with file metadata) gives both the list and
    the sizes. GGUF entries fetch only ``gguf_file``.
    """
    hub = _hub()
    info = hub.HfApi().model_info(entry["hf_id"], revision=entry["revision"], files_metadata=True)
    sizes = {s.rfilename: getattr(s, "size", None) for s in (info.siblings or [])}
    gguf = str(entry.get("gguf_file") or "").strip()
    if gguf:
        if gguf not in sizes:
            raise DownloadError(
                f"{entry['hf_id']} at {entry['revision'][:12]} has no file {gguf!r}. "
                "Check gguf_file in config/models_catalog.json."
            )
        return [(gguf, sizes[gguf])]
    names = paths.select_filenames(sorted(sizes))
    if not names:
        raise DownloadError(f"{entry['hf_id']} at {entry['revision'][:12]} has no model files.")
    return [(name, sizes[name]) for name in names]


def entry_snapshot(entry: dict, weights_dir: Path) -> Path | None:
    """The pinned snapshot on disk with everything this entry needs, or None."""
    from modelwidget import catalog  # local import: catalog lazily imports this module

    return catalog.entry_snapshot(entry, weights_dir)


def disk_size_mb(snapshot: Path | str | None) -> float | None:
    """Real bytes used by a snapshot (symlinks followed, each blob counted once)."""
    if not snapshot:
        return None
    seen: set[Path] = set()
    total = 0
    for item in Path(snapshot).rglob("*"):
        try:
            real = item.resolve()
            if real in seen or not real.is_file():
                continue
            seen.add(real)
            total += real.stat().st_size
        except OSError:
            continue
    return round(total / _MB, 1)


def _progress_class(on_file_bytes: Callable[[int], None]):
    """A tqdm class huggingface_hub drives; we only read its counters.

    huggingface_hub may open more than one bar per file (the Xet transfer
    path reports both network and written bytes), so per-file progress is
    the largest counter seen, not the sum of every update.
    """
    from tqdm.auto import tqdm

    state = {"max": 0}

    class _Progress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # keep the server log quiet
            super().__init__(*args, **kwargs)
            self._seen = int(kwargs.get("initial") or 0)
            self._report()

        def _report(self):
            if self._seen > state["max"]:
                state["max"] = self._seen
                on_file_bytes(self._seen)

        def update(self, n=1):
            if n:
                self._seen += int(n)
                self._report()
            return super().update(n)

    def reset():
        state["max"] = 0

    _Progress.reset_file = staticmethod(reset)
    return _Progress


def download_entry(
    entry: dict,
    weights_dir: Path,
    *,
    on_plan: Callable[[list[tuple[str, int | None]]], None] | None = None,
    on_file: Callable[[str, int], None] | None = None,
    on_bytes: Callable[[int], None] | None = None,
) -> tuple[Path, bool]:
    """Fetch an entry into ``weights_dir``. Returns (snapshot, downloaded).

    ``on_bytes(total)`` gets the running byte count for the whole entry,
    never more than the planned total.
    """
    weights_dir = Path(weights_dir)
    found = entry_snapshot(entry, weights_dir)
    if found is not None:
        return found, False
    reason = offline_reason()
    if reason:
        raise DownloadError(reason)
    weights_dir.mkdir(parents=True, exist_ok=True)
    hub = _hub()
    try:
        files = plan_files(entry)
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(_explain(entry, exc)) from exc
    total = sum(size for _, size in files if size)
    clean_stale_partials(weights_dir, entry["hf_id"])
    _check_disk(entry, weights_dir, total)
    if on_plan:
        on_plan(files)
    done_before = 0
    current_size = 0

    def file_bytes(n: int) -> None:
        if on_bytes:
            capped = min(n, current_size) if current_size else n
            on_bytes(min(done_before + capped, total) if total else done_before + capped)

    tqdm_class = _progress_class(file_bytes) if on_bytes else None
    last = None
    for index, (name, size) in enumerate(files):
        current_size = size or 0
        if tqdm_class is not None:
            tqdm_class.reset_file()
        if on_file:
            on_file(name, index)
        try:
            kwargs = dict(
                repo_id=entry["hf_id"],
                filename=name,
                revision=entry["revision"],
                cache_dir=str(weights_dir),
            )
            if tqdm_class is not None:
                kwargs["tqdm_class"] = tqdm_class
            last = hub.hf_hub_download(**kwargs)
        except Exception as exc:
            raise DownloadError(_explain(entry, exc, name)) from exc
        done_before += size or 0
        if on_bytes:
            on_bytes(min(done_before, total) if total else done_before)
    if last is None:
        raise DownloadError(f"No files were downloaded for {entry['hf_id']}.")
    snapshot = paths.snapshot_from_download(Path(last))
    return snapshot, True


def clean_stale_partials(weights_dir: Path, repo_id: str, older_than_s: float = 60.0) -> int:
    """Delete ``*.incomplete`` leftovers for one repo. Returns bytes freed."""
    root = Path(weights_dir) / paths.model_dir_name(repo_id)
    freed = 0
    if not root.is_dir():
        return 0
    now = time.time()
    for partial in root.rglob("*.incomplete"):
        try:
            stat = partial.stat()
            if now - stat.st_mtime < older_than_s:
                continue  # may belong to a download running right now
            partial.unlink()
            freed += stat.st_size
        except OSError:
            continue
    if freed:
        print(f"[downloads] removed {freed / _MB:.1f} MB of interrupted partial files for {repo_id}", flush=True)
    return freed


def _check_disk(entry: dict, weights_dir: Path, total_bytes: int) -> None:
    if not total_bytes:
        return
    try:
        import shutil

        free = shutil.disk_usage(str(weights_dir)).free
    except OSError:
        return
    if free < total_bytes * 1.1:
        raise DownloadError(
            f"Not enough disk space for {entry['name']}: needs about {total_bytes / _MB:.0f} MB, "
            f"{free / _MB:.0f} MB free in {weights_dir}."
        )


def _explain(entry: dict, exc: Exception, filename: str | None = None) -> str:
    what = f"{entry['hf_id']}" + (f" file {filename}" if filename else "")
    text = f"{type(exc).__name__}: {exc}"
    hints = []
    lowered = text.lower()
    if "offline" in lowered:
        hints.append("HF_HUB_OFFLINE may be set; unset it to download")
    if any(k in lowered for k in ("connection", "resolve", "timed out", "timeout", "proxy", "ssl", "max retries")):
        hints.append("check the internet connection (and HTTPS_PROXY if you use a proxy)")
    if "401" in lowered or "403" in lowered or "gated" in lowered:
        hints.append("this repo may need a Hugging Face token (HF_TOKEN) and accepted licence")
    if "404" in lowered or "not found" in lowered:
        hints.append("the repo, revision or file name in the catalog may be wrong")
    if "no space" in lowered:
        hints.append("free some disk space")
    hint = f" Hint: {'; '.join(hints)}." if hints else ""
    return f"Could not download {what}: {text}.{hint}"


# ---------------------------------------------------------------------------
# Background jobs for the web app
# ---------------------------------------------------------------------------

class DownloadManager:
    """One worker thread, one job per catalog id, status kept in memory."""

    def __init__(self, weights_dir: Path, downloader: Callable = download_entry):
        self.weights_dir = Path(weights_dir)
        self._download = downloader
        self._jobs: dict[str, Job] = {}
        self._entries: dict[str, dict] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(self, entry: dict) -> dict:
        """Queue a download. An active or finished job for the same id is reused."""
        with self._lock:
            job = self._jobs.get(entry["id"])
            if job is not None and job.state in ("queued", "downloading"):
                return job.to_dict()
            job = Job(id=entry["id"], name=entry["name"], hf_id=entry["hf_id"], revision=entry["revision"])
            self._jobs[entry["id"]] = job
            self._entries[entry["id"]] = dict(entry)
            self._queue.put(entry["id"])
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="model-downloads", daemon=True)
                self._worker.start()
            return job.to_dict()

    def get(self, model_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(model_id)
            return None if job is None else job.to_dict()

    def all(self) -> list[dict]:
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]

    def busy(self, model_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(model_id)
            return job is not None and job.state in ("queued", "downloading")

    def wait(self, model_id: str, timeout: float = 30.0) -> dict | None:
        """Block until a job finishes (tests and the CLI use this)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.busy(model_id):
                return self.get(model_id)
            time.sleep(0.02)
        return self.get(model_id)

    # -- worker ---------------------------------------------------------------
    def _update(self, job: Job, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)

    def _run(self) -> None:
        while True:
            try:
                model_id = self._queue.get(timeout=5)
            except queue.Empty:
                return
            with self._lock:
                job = self._jobs[model_id]
                entry = self._entries[model_id]
            self._process(job, entry)

    def _process(self, job: Job, entry: dict) -> None:
        self._update(job, state="downloading", started_at=time.time(), message="Checking files…")
        order: list[tuple[str, int | None]] = []

        def on_plan(files):
            order[:] = list(files)
            total = sum(s for _, s in files if s) or None
            self._update(job, files_total=len(files), bytes_total=total,
                         message=f"Downloading {len(files)} file(s).")

        def on_file(name, index):
            # Files before this one are complete even if no progress was reported.
            before = sum(size or 0 for _, size in order[:index])
            with self._lock:
                job.current_file = name
                job.files_done = index
                job.bytes_done = max(job.bytes_done, before)

        def on_bytes(total_so_far):
            with self._lock:
                job.bytes_done = max(job.bytes_done, total_so_far)

        try:
            snapshot, downloaded = self._download(
                entry, self.weights_dir, on_plan=on_plan, on_file=on_file, on_bytes=on_bytes
            )
        except Exception as exc:  # surfaced in the job, never swallowed
            message = str(exc) if isinstance(exc, DownloadError) else f"{type(exc).__name__}: {exc}"
            self._update(job, state="error", error=message, message=message,
                         finished_at=time.time(), current_file=None)
            print(f"[downloads] {entry['id']} failed: {message}", flush=True)
            return
        with self._lock:
            if job.bytes_total:
                job.bytes_done = job.bytes_total
            job.files_done = job.files_total
        size = disk_size_mb(snapshot)
        self._update(
            job, state="done", snapshot=str(snapshot), downloaded=downloaded,
            current_file=None, finished_at=time.time(),
            message=("Downloaded" if downloaded else "Already on disk") + f" ({size} MB) to {snapshot}",
        )
        print(f"[downloads] {entry['id']}: {job.message}", flush=True)
