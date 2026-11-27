"""Branch lanes for First Book original episodes.

The branch map lives in ``config/branches.json``. Episodes carry ``branch``
and ``parent`` (the earlier episode they build on). The video page nests a
branch under the episode it grows from, and leaves a branch on the trunk
when that episode is not filmed yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from firstbook import remote_library, videos
from firstbook.library import VIDEOS_CATALOG_PATH

BRANCHES_PATH = VIDEOS_CATALOG_PATH.parent / "branches.json"


def load_branch_defs(path: Path | None = None) -> list[dict]:
    """Branch rows in display order. A missing file gives an empty map."""
    data = videos._read_json(Path(path or BRANCHES_PATH)) or {}
    rows = []
    for raw in data.get("branches") or []:
        if not isinstance(raw, dict) or not str(raw.get("id") or "").strip():
            return_id = ""
        else:
            return_id = videos.branch_id(raw.get("id"))
        if not return_id:
            continue
        rows.append({
            "id": return_id,
            "title": str(raw.get("title") or return_id),
            "summary": str(raw.get("summary") or ""),
            "parent_episode": videos.parent_episode(raw.get("parent_episode")),
            "parent_label": str(raw.get("parent_label") or ""),
            "order": raw.get("order") if isinstance(raw.get("order"), int) else 50,
            "open": bool(raw.get("open")),
        })
    rows.sort(key=lambda row: (row["order"], row["title"].lower()))
    return rows


def _clock(seconds: object) -> str:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return ""
    total = int(round(float(seconds)))
    if total < 0:
        return ""
    return f"{total // 60}:{total % 60:02d}"


def _card(file: str, remote: dict | None, local: dict | None) -> dict | None:
    remote = remote or {}
    local = local or {}
    sidecar = remote.get("sidecar") if isinstance(remote.get("sidecar"), dict) else {}
    branch = videos.branch_id(local.get("branch") or remote.get("branch") or sidecar.get("branch"))
    episode = str(local.get("episode") or remote.get("episode") or sidecar.get("episode") or "").strip()
    if local.get("parent") is not None or "parent" in local:
        parent = videos.parent_episode(local.get("parent"))
    else:
        parent = videos.parent_episode(remote.get("parent", sidecar.get("parent", sidecar.get("parent_episode"))))
    if not branch and not episode:
        return None
    title = local.get("title") or remote.get("title") or sidecar.get("title") or videos.title_from_filename(Path(file).name)
    topic = str(local.get("topic") or remote.get("topic") or sidecar.get("topic") or "other").strip().lower() or "other"
    tags = local.get("tags") if isinstance(local.get("tags"), list) else None
    if tags is None:
        tags = remote.get("tags") if isinstance(remote.get("tags"), list) else sidecar.get("tags") or []
    duration = local.get("duration_s")
    if duration is None:
        duration = remote.get("duration_s", sidecar.get("duration_s"))
    creator = str(local.get("creator") or remote.get("creator") or sidecar.get("creator") or "")
    description = str(local.get("description") or remote.get("description") or sidecar.get("description") or "")
    license_text = str(local.get("license") or remote.get("license") or sidecar.get("license") or "")
    if local and local.get("size_mb"):
        size_label = f"{local['size_mb']} MB"
    else:
        size_label = str(remote.get("size_label") or "") or remote_library.size_label(remote.get("size"))
    return {
        "id": file,
        "file": file,
        "manifest_id": str(remote.get("id") or Path(file).stem),
        "watch_id": Path(file).stem,
        "title": str(title),
        "episode": episode,
        "branch": branch,
        "parent": parent,
        "topic": topic,
        "topic_name": videos.TOPIC_NAMES.get(topic, topic[:1].upper() + topic[1:] if topic else ""),
        "tags": [str(t) for t in tags],
        "duration_s": duration if isinstance(duration, (int, float)) and not isinstance(duration, bool) else None,
        "duration_label": _clock(duration),
        "size_label": size_label,
        "license": license_text,
        "creator": creator,
        "description": description,
        "downloaded": bool(local),
        "forks": [],
    }


def list_episodes(videos_dir: Path) -> list[dict]:
    """Originals from the active manifest, plus any branched file already on disk."""
    videos_dir = Path(videos_dir)
    local_by_file = {item["file"]: item for item in videos.index(videos_dir)}
    manifest = remote_library.active_manifest(videos_dir)
    remote_by_file: dict[str, dict] = {}
    if manifest:
        for entry in manifest["videos"]:
            if isinstance(entry, dict) and isinstance(entry.get("file"), str):
                remote_by_file[entry["file"]] = entry
    cards = []
    seen: set[str] = set()
    for file, entry in remote_by_file.items():
        card = _card(file, entry, local_by_file.get(file))
        if card is not None:
            cards.append(card)
            seen.add(file)
    for item in local_by_file.values():
        if item["file"] in seen:
            continue
        card = _card(item["file"], None, item)
        if card is not None:
            cards.append(card)
    return cards


def nest_branches(episodes: list[dict], branch_defs: list[dict], *, keep_empty: bool = True) -> list[dict]:
    """Group episodes into lanes and hang each lane off its parent episode.

    ``keep_empty`` keeps planned branches that have no episodes yet. A search
    passes False so only lanes with a matching episode remain.
    """
    by_id: dict[str, dict] = {}
    for raw in branch_defs:
        by_id[raw["id"]] = {
            **raw,
            "parent_title": "",
            "theme_file": f"audio/branches/{raw['id']}.ogg",
            "theme_mp3": f"audio/branches/{raw['id']}.mp3",
            "episodes": [],
        }
    for ep in episodes:
        bid = videos.branch_id(ep.get("branch")) or "other"
        if bid not in by_id:
            parent = videos.parent_episode(ep.get("parent"))
            by_id[bid] = {
                "id": bid,
                "title": bid.replace("-", " ").title(),
                "summary": "",
                "parent_episode": parent,
                "parent_label": f"Branches from {parent}" if parent else "",
                "parent_title": "",
                "order": 80,
                "open": False,
                "theme_file": f"audio/branches/{bid}.ogg",
                "theme_mp3": f"audio/branches/{bid}.mp3",
                "episodes": [],
            }
        item = dict(ep)
        item["forks"] = []
        by_id[bid]["episodes"].append(item)
    for lane in by_id.values():
        lane["episodes"].sort(key=lambda e: (str(e.get("episode") or "zzz"), str(e.get("title") or "").lower()))
        if not lane.get("parent_label") and lane.get("parent_episode"):
            lane["parent_label"] = f"Branches from {lane['parent_episode']}"

    episode_by_num: dict[str, dict] = {}
    for lane in by_id.values():
        for ep in lane["episodes"]:
            if ep.get("episode"):
                episode_by_num[str(ep["episode"])] = ep

    roots = []
    for lane in by_id.values():
        parent = lane.get("parent_episode")
        host = episode_by_num.get(str(parent)) if parent else None
        if host is not None and lane["id"] != videos.branch_id(host.get("branch")):
            if keep_empty or lane["episodes"]:
                lane["parent_title"] = str(host.get("title") or "")
                host["forks"].append(lane)
            continue
        roots.append(lane)

    def sort_forks(lane: dict) -> None:
        for ep in lane["episodes"]:
            ep["forks"].sort(key=lambda child: (child.get("order", 50), child["title"].lower()))
            for fork in ep["forks"]:
                sort_forks(fork)

    for lane in roots:
        sort_forks(lane)
    if not keep_empty:
        roots = [lane for lane in roots if lane["episodes"]]
    roots.sort(key=lambda lane: (lane.get("order", 50), lane["title"].lower()))
    return roots


def files_in_tree(lanes: list[dict]) -> set[str]:
    """Relative paths already shown in the branch tree."""
    found: set[str] = set()

    def walk(lane: dict) -> None:
        for ep in lane.get("episodes") or []:
            if ep.get("file"):
                found.add(ep["file"])
            for fork in ep.get("forks") or []:
                walk(fork)

    for lane in lanes:
        walk(lane)
    return found


def for_page(videos_dir: Path, q: str = "", topic: str = "") -> list[dict]:
    """Lanes for /videos. A search hides empty planned branches."""
    filtering = bool((q or "").strip() or (topic or "").strip())
    episodes = list_episodes(videos_dir)
    if filtering:
        episodes = videos.search(episodes, q, topic)
    return nest_branches(episodes, load_branch_defs(), keep_empty=not filtering)


def playable(videos_dir: Path, rel: str) -> dict | None:
    """One watch-page record: the file on disk, or the remote URL when it is not."""
    videos_dir = Path(videos_dir)
    local = next((item for item in videos.index(videos_dir) if item["id"] == rel), None)
    remote = None
    manifest = remote_library.active_manifest(videos_dir)
    if manifest:
        for entry in manifest["videos"]:
            if entry.get("file") == rel or entry.get("id") == rel:
                remote = entry
                break
    if local is None and remote is None:
        return None
    if local is not None:
        item = dict(local)
        item["playback_url"] = local["url"]
        item["downloaded"] = True
        item["manifest_id"] = str((remote or {}).get("id") or Path(rel).stem)
        item["watch_id"] = Path(rel).stem
        if not item.get("episode"):
            item["episode"] = str((remote or {}).get("episode") or "")
        if not item.get("branch"):
            item["branch"] = str((remote or {}).get("branch") or "")
        return item
    return _remote_watch(videos_dir, remote)


def remote_media_url(videos_dir: Path, rel: str) -> str | None:
    """The http(s) file URL for one manifest path, if the active manifest has it."""
    manifest = remote_library.active_manifest(videos_dir)
    if not manifest:
        return None
    for entry in manifest["videos"]:
        if entry.get("file") == rel:
            url = str(entry.get("url") or "")
            if url.startswith(("http://", "https://")):
                return url
    return None


def _remote_watch(videos_dir: Path, entry: dict) -> dict:
    sidecar = entry.get("sidecar") if isinstance(entry.get("sidecar"), dict) else {}
    file = entry["file"]
    subs = ""
    sub_rel = entry.get("subtitles") or ""
    if isinstance(sub_rel, str) and sub_rel and (videos_dir / sub_rel).is_file():
        subs = f"/videos/subtitles/{sub_rel}"
    source = str(entry.get("source") or entry.get("source_url") or sidecar.get("source") or "")
    attribution = str(entry.get("attribution") or entry.get("credit") or sidecar.get("attribution") or "")
    tags = entry.get("tags") if isinstance(entry.get("tags"), list) else sidecar.get("tags") or []
    return {
        "id": file,
        "file": file,
        "title": str(entry.get("title") or sidecar.get("title") or ""),
        "topic": str(entry.get("topic") or "other"),
        "tags": [str(t) for t in tags],
        "duration_s": entry.get("duration_s"),
        "description": str(entry.get("description") or sidecar.get("description") or ""),
        "source": source,
        "license": str(entry.get("license") or ""),
        "license_url": str(entry.get("license_url") or sidecar.get("license_url") or ""),
        "creator": str(entry.get("creator") or sidecar.get("creator") or ""),
        "creator_url": str(entry.get("creator_url") or sidecar.get("creator_url") or ""),
        "attribution": attribution,
        "fact_check": str(sidecar.get("fact_check") or entry.get("fact_check") or ""),
        "mimetype": videos.MIMETYPES.get(Path(file).suffix.lower(), "video/mp4"),
        "url": str(entry.get("url") or ""),
        "playback_url": f"/videos/stream/{file}",
        "subtitles": sub_rel if subs else "",
        "subtitles_url": subs,
        "downloaded": False,
        "manifest_id": str(entry.get("id") or Path(file).stem),
        "watch_id": Path(file).stem,
        "episode": str(entry.get("episode") or sidecar.get("episode") or ""),
        "branch": str(entry.get("branch") or sidecar.get("branch") or ""),
        "parent": videos.parent_episode(entry.get("parent", sidecar.get("parent"))),
        "size_mb": None,
        "has_metadata": True,
    }
