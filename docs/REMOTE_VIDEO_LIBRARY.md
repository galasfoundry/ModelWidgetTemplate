# Remote video library

The First Book keeps video *files* out of git. The app plays whatever is
already in `library/videos/` with no network. To add or update those
files it can fetch a small versioned manifest from a configurable host,
then download only what is new or has changed.

## Settings

Set these in `config/.env` or the shell:

| Variable | Meaning |
| --- | --- |
| `FIRSTBOOK_LIBRARY_URL` | Base URL of the remote library. The manifest is `{base}/manifest.json` unless the value itself ends in `.json`. Default: `https://github.com/galasfoundry/ModelWidgetTemplate/releases/download/episodes-v1/` (the published **episodes-v1** release). |
| `FIRSTBOOK_LIBRARY_MANIFEST` | Optional override: a local file path or a direct URL to the manifest. Useful for tests and for a USB copy of the catalog. |

## Manifest

Schema id: `firstbook.library.v1`. Each video lists `id`, `title`,
`topic`, `license`, `credit`, `source_url`, file `url`, `size`,
`sha256`, `duration_s`, `subtitles_url`, and a `sidecar` object of extra
metadata (description, tags, creator, fact-check notes, and so on).
Original episodes also list `branch` (the lane id), `parent` (the episode
they build on, or null), and `episode`. See
`docs/remote-manifest.example.json` for the shape. The hashes in that
file are illustrative. The live manifest is the `manifest.json` asset on
the `episodes-v1` release. A copy of that file is committed as
`config/library_manifest.json` so the video page can list the episodes
before the first online check.

Build it from local files (hashes, sizes, sidecar and catalog metadata):

```bash
python scripts/build_video_manifest.py --remote \
  --base-url https://github.com/galasfoundry/ModelWidgetTemplate/releases/download/episodes-v1/ \
  --url-mode flat \
  -o /tmp/manifest.json
```

`--url-mode path` (default) keeps `topic/name.mp4` in the URL, which
suits a folder on a web server or the Internet Archive.
`--url-mode flat` uses only the filename, which GitHub Releases need.

## In the app

On `/videos`, **Download library** fetches the manifest, lists new and
updated items with sizes, and downloads the selection (or all pending
items) in the background. Transfers stream in 64 KiB chunks, resume from
`library/videos/.downloading/<file>.part`, verify sha256, and `os.replace`
into place. A failed hash check deletes the partial and leaves any
existing file untouched. Sidecars and subtitles are written next to the
video so the existing watch page can show credits and captions.

When offline, `/videos` still lists and plays what is already on disk,
and shows when the manifest was last checked (`library/videos/.library-state.json`).
An episode that is not downloaded yet can still be played while online:
the watch page asks this app for `/videos/stream/...`, and the app reads
the release (Range requests included) and passes the bytes to the player.
After Download, the watch page uses the local file and does not touch the
network.

`scripts/fetch_videos.py` (yt-dlp) stays as a fallback for the
third-party Creative Commons catalog in `config/videos_catalog.json`.

## Hosting options

### GitHub Releases (free, 2 GB per file)

The originals are published on `galasfoundry/ModelWidgetTemplate` as
tag `episodes-v1` (release name "First Book episodes v1"). It is a
normal published release, not a draft or a prerelease. The app default is:

```bash
FIRSTBOOK_LIBRARY_URL=https://github.com/galasfoundry/ModelWidgetTemplate/releases/download/episodes-v1/
```

That release holds each `.mp4`, `.srt` and `manifest.json` as assets.
Build the manifest with `--url-mode flat`. GitHub does not allow folders
in a release, so two files cannot share a filename even if they would
live in different topics. The 2 GB-per-file limit is the practical cap;
watch the account's release bandwidth as the library grows. A URL that
still contains `library-placeholder` is treated as not live. `episodes-v1`
ships originals 001 to 004, with branch and parent fields. An earlier
release, `library-v1`, has only 001 and 002. The Creative Commons YouTube
catalog is not on either release.

### Internet Archive (good for Creative Commons)

The Internet Archive is a natural home for CC BY material: durable,
public, and happy with larger files and folder-like item names. Create
an item, upload the videos, captions and `manifest.json`, and set
`FIRSTBOOK_LIBRARY_URL` to the item's download base, typically:

```text
https://archive.org/download/<item-name>/
```

`--url-mode path` works if you upload files with their relative paths
(or matching names). Credit each third-party video as the sidecar
already records.

### Self-hosted server (later)

Any static file host works: a folder behind nginx, Caddy or Apache, or
an object bucket with a public prefix. Serve `manifest.json` and the
video paths from one base URL, keep Range requests enabled so downloads
can resume, and point `FIRSTBOOK_LIBRARY_URL` at that base. No app
change is required.

## Adding an episode

1. Put the `.mp4`, `.srt` and `.mp4.json` sidecar in `library/videos/originals/` (the sidecar and captions can be committed; the `.mp4` is gitignored).
2. `python scripts/build_video_manifest.py --remote --base-url <URL> --url-mode <path|flat> -o manifest.json`
3. Upload the new files and the new `manifest.json` to the host.
4. People already running the app press **Download library** and only the new or changed files are fetched.
