# The First Book: architecture

The First Book is a fully offline UK survival and know-how library: UK maps,
a practical "tech-hacky" video library (repair, electronics, power, radio,
tools, 3D printing…), short guides, and small local AI models. It grows out
of this model widget template and is built for a 4 GB RAM laptop.

The guiding rule: **after setup, nothing touches the network.** The only
exceptions are an explicit model download (`POST /api/models/select` or
`POST /api/models/<id>/download`) and an explicit Hugging Face search from
Discover (`GET /api/models/discover`). Search results are cached on disk so
a previous search still shows offline. Map packs and videos are copied in
by hand.

## Code layout

```text
app.py                      Flask app: model widget routes + registers the First Book blueprint
paths.py                    project-relative paths, env loading, weight cache search
firstbook/library.py        data root and section folders (LIBRARY_DIR, MAPS_DIR, …)
firstbook/maps.py           map pack listing, MBTiles tile reads (stdlib sqlite3)
firstbook/videos.py         video index from videos.json + sidecars, search, topics
firstbook/guides.py         Markdown/text guides, keyword search
firstbook/chat.py           First Book chat prompt, history trim and status messages
firstbook/web.py            routes for /home, /maps, /videos, /guides, /chat and /api/status
modelwidget/catalog.py      model catalog (category, status), filtering, install-if-missing
modelwidget/discover.py     Hugging Face GGUF search, fit labels, user catalog, disk cache
modelwidget/capabilities.py RAM, disk, CUDA, backends (incl. optional llama_cpp)
modelwidget/downloads.py    background download jobs, progress, single-file GGUF fetch
modelwidget/gguf.py         llama.cpp wrapper for GGUF models (optional llama-cpp-python)
modelwidget/adapters/llama_cpp.py  text-generation adapter for GGUF
scripts/download_models.py  CLI: list and download catalog models into weights/
scripts/discover_models.py  CLI: search Hugging Face for GGUF models that fit this machine
config/models_catalog.json  curated model catalog
config/models_user.json     user-added discovery picks (gitignored)
config/maps_catalog.json    known UK map packs, sources and licences
scripts/build_video_manifest.py  scan a folder into videos.json (ffprobe optional)
scripts/fetch_demo_map.py   setup-time UK map extracts from the Protomaps build
static/js/maps.js           the /maps viewer
static/vendor/              MapLibre GL JS, pmtiles.js, Protomaps style, glyphs, sprites
templates/, static/         bare pages; no CDN, no remote assets
library/                    offline content (see below)
weights/                    model snapshots (unchanged)
```

## Data layout

```text
library/                 LIBRARY_DIR (default: <project>/library)
  maps/                  MAPS_DIR    *.mbtiles, *.pmtiles          (gitignored)
  videos/                VIDEOS_DIR  *.mp4 *.webm *.mkv *.m4v
                                     videos.json, <file>.json      (gitignored)
  guides/                GUIDES_DIR  *.md, *.txt                   (committed)
weights/                 WEIGHTS_DIR Hugging Face cache layout     (gitignored)
```

Point `LIBRARY_DIR` at an external drive to keep big content off the laptop's
internal disk. Each section can be moved on its own. Relative paths resolve
from the project root; set them in `config/.env` or the shell.

## Routes

| Route | What |
| --- | --- |
| `GET /home` | Hub page with counts and machine summary |
| `GET /` | Model widget (unchanged) |
| `GET /api/status` | Content present (map packs, videos, guides, models) and machine capability |
| `GET /maps`, `GET /api/maps` | Map packs on disk + known UK packs |
| `GET /tiles/<pack>/<z>/<x>/<y>[.ext]` | One XYZ tile from `<pack>.mbtiles`; 204 when absent |
| `GET /maps/files/<pack>.pmtiles` | Whole PMTiles file with HTTP Range |
| `GET /videos`, `GET /api/videos?q=&topic=&tag=` | List and search videos |
| `GET /videos/watch/<path>` | Bare player page |
| `GET /videos/file/<path>` | Stream a video with HTTP Range (206) |
| `GET /guides`, `GET /api/guides?q=` | List and search guides |
| `GET /guides/<slug>`, `GET /api/guides/<slug>` | One guide |
| `GET /chat`, `GET /api/chat/status` | Offline GGUF chat page and readiness |
| `POST /api/chat` | Stream a chat reply (SSE). Body: `{ "messages": [...] }` |
| `POST /api/chat/stop` | Stop the current chat reply |
| `GET /api/models`, `POST /api/models/select` | Model picker (rows carry `downloadable`, `warnings`, `download`; `planned` listed separately; user picks have `source: user`) |
| `GET /api/models/discover?q=&task=chat` | Search Hugging Face for GGUF chat models that fit this machine; cached for offline use |
| `POST /api/models/discover/add`, `POST /api/models/discover/remove` | Write or drop a pick in `config/models_user.json` (never the curated catalog) |
| `POST /api/models/<id>/download`, `GET /api/models/<id>/status`, `GET /api/models/downloads` | Background downloads into `weights/` with progress |

## Maps

* **MBTiles** (SQLite) are read with the standard library. Tiles are stored
  in TMS order, so the route flips `y`. Gzipped vector tiles are sent with
  `Content-Encoding: gzip`; the type comes from the pack's `format` metadata.
* **PMTiles** are served as a file with Range support, so a viewer reads them
  by byte range with no server-side decoding.
* `config/maps_catalog.json` lists UK packs: a Protomaps UK extract
  (PMTiles, via `pmtiles extract … --bbox=-8.65,49.86,1.77,60.86`), a
  self-built OpenStreetMap MBTiles from the Geofabrik UK extract
  (planetiler or tilemaker), and OS Open Zoomstack (Ordnance Survey vector
  MBTiles, about 2.7 GB, Great Britain only, OGL v3).
* **Licences.** OpenStreetMap data is ODbL 1.0 and must be credited
  "© OpenStreetMap contributors"; OpenMapTiles-schema tiles also credit
  "© OpenMapTiles"; OS data credits "Contains OS data © Crown copyright and
  database right". The attribution is shown on the Maps page and must be
  shown by any viewer.

### Adding a map pack

0. Quickest: `python scripts/fetch_demo_map.py` (see below).
1. While online, fetch or build a pack using the `how_to_get` notes in
   `config/maps_catalog.json`. A low `--maxzoom` or small bbox keeps it small.
2. Copy it into `library/maps/` with the catalog `filename`
   (e.g. `uk.pmtiles`). Any other name works too; it just won't pick up the
   catalog's title and licence.
3. Restart nothing: `/api/maps` rescans the folder on every request.

### Demo packs and bigger UK extracts

`scripts/fetch_demo_map.py` is a setup tool: run it once while online. It
fetches the go-pmtiles CLI (v1.31.2) into `tools/` if `pmtiles` is not on
PATH, finds the newest Protomaps daily build, and cuts extracts with HTTP
range requests (the 100+ GB planet file is never downloaded):

| Command | Pack | Size (18 Nov 2026 build) |
| --- | --- | --- |
| `python scripts/fetch_demo_map.py` | `uk-overview.pmtiles` (UK bbox, zoom 0-8) + `bristol.pmtiles` (Bristol and Bath, bbox -2.75,51.38,-2.45,51.56, zoom 0-15) | ~6 MB + ~23 MB |
| `… --uk 12` | `uk-z12.pmtiles`, whole UK to zoom 12 (towns, main roads) | ~350 MB |
| `… --uk 14` | `uk-z14.pmtiles`, whole UK to zoom 14 (streets) | ~1.5 GB |
| `… --only custom --name leeds --bbox=-1.65,53.75,-1.45,53.85` | any city or region, zoom 15 | a few MB to tens of MB |

`--dry-run` prints the size first; `--force` replaces an existing pack. The
viewer overzooms past a pack's max zoom, so a z12 UK pack still works zoomed
in, just with less detail.

### Viewer

`/maps` is a MapLibre GL JS viewer. Everything it loads comes from this
server:

* `static/vendor/maplibre-gl/` MapLibre GL JS 5.24.1 (UMD build, BSD-3-Clause)
* `static/vendor/pmtiles/` pmtiles.js 4.5.0 (BSD-3-Clause), registered as the
  `pmtiles://` protocol and pointed at `/maps/files/<pack>.pmtiles`
* `static/vendor/protomaps-basemaps/` @protomaps/basemaps 5.7.2 style
  generator (code BSD-3-Clause, design CC0), "light" flavour, English labels
* `static/vendor/basemaps-assets/` Noto Sans Regular/Medium/Italic glyphs for
  Latin ranges (0-255, 256-511, 8192-8447; SIL OFL) and the v4 "light"
  sprite (derived from MIT tangrams/icons)

See `static/vendor/README.md` for exact sources and how to update them.

How a pack is drawn (`static/js/maps.js`):

| Pack | Source | Style |
| --- | --- | --- |
| PMTiles, vector, Protomaps schema | `pmtiles://` | Protomaps light basemap with labels |
| MBTiles `pbf`, Protomaps schema | `/tiles/<pack>/{z}/{x}/{y}` | same |
| Vector, other schema (OS Open Zoomstack, OpenMapTiles) | either | plain generic style: every layer as fill / line / point, **no labels** yet |
| PMTiles or MBTiles raster (png/jpg/webp) | either | raster layer |

The map opens on the pack's bounds (catalog `bounds`, MBTiles metadata or the
PMTiles header), falling back to the whole UK. Attribution comes from the
catalog entry, then the pack metadata, then the catalog default
("© OpenStreetMap contributors"), and is shown in the map corner.

Labels outside Latin script (e.g. foreign country names at low zoom) have no
vendored glyphs; MapLibre logs a 404 and draws them with a local system font.
Adding more ranges is just copying more `.pbf` files into the font folders.

## Videos

* Metadata is JSON: `library/videos/videos.json` (see
  `library/videos/videos.example.json`) and optional per-file sidecars
  `<video>.json` that override the manifest entry.
* Fields: `file` (relative path), `title`, `topic`, `tags`, `duration_s`,
  `description`, `source`, `license`. Files without metadata still appear.
* Suggested topics (`firstbook.videos.TOPICS`): repair, electronics, power,
  radio, computing, 3d-printing, tools, making, water, food, shelter,
  first-aid, navigation, other.
* Streaming uses `send_file(conditional=True)`, so seeking works (Range/206).

### Adding videos

1. Copy files into `library/videos/` (sub-folders allowed).
2. `python scripts/build_video_manifest.py --topic electronics` adds new files
   to `videos.json`, keeps existing notes and fills `duration_s` when
   `ffprobe` is installed.
3. Edit titles and tags in `videos.json`. Record source and licence.

## Guides

Plain `.md`/`.txt` files in `library/guides/`. The first line or heading is
the title; search ranks by keyword counts (title hits weigh more). Pages show
raw text for now. The two shipped guides are **placeholders** that only
explain the format; real guides should cite where their content came from.

## Models

The picker is extended, not replaced:

* Every catalog entry has `category` (`vision`, `text`, `llm`, `audio`,
  `other`; inferred from `task` when missing) and `status`:
  * `available`: download and use from the picker;
  * `downloadable`: download any time; use only when its backend is
    installed (the GGUF chat models need `llama-cpp-python`);
  * `planned`: listed under `planned` only; download (409) and select (404)
    refuse it.
* **Downloads are separate from activation.** `modelwidget/downloads.py` has
  a `DownloadManager` (one background worker thread, one job per model, state
  in memory) behind `POST /api/models/<id>/download`, `GET
  /api/models/<id>/status` and `GET /api/models/downloads`, and the same
  `download_entry()` powers `scripts/download_models.py`. Files land in
  `weights/` through `hf_hub_download(cache_dir=weights/)`. Progress comes from
  huggingface_hub's own progress bars, capped at the planned size. Errors are
  turned into readable messages (offline flag, network/proxy, gated, 404,
  disk) and kept on the job.
* RAM and disk limits block downloads as well as use. A missing backend only
  blocks use. LLM entries of 700 MB or more on a machine with 4.5 GB RAM or
  less carry a warning.
* Four small GGUF chat models, each pinned to a verified Hugging Face commit
  and **one** quantised file: Gemma 3 270M (Q4_K_M, ~253 MB), SmolLM2 360M
  (Q8_0, ~386 MB), Qwen2.5 0.5B (Q4_K_M, ~491 MB), Llama 3.2 1B (Q4_K_M,
  ~808 MB). Gemma and Llama have their own licence terms (noted per entry).
  With `llama-cpp-python` installed, `modelwidget/adapters/llama_cpp.py`
  loads the file through `modelwidget/gguf.py`, `MODEL_FILE` names it in
  `config/.env`, and `ModelService` fetches only that file if it is missing
  (`paths.ensure_file`).
* `/chat` is a bare streaming chat page for those GGUF models. It sends
  `POST /api/chat` (SSE) with the session history, a short First Book system
  prompt, and `CHAT_MAX_TOKENS` (default 384). The model's GGUF chat template
  is applied. History is trimmed to fit `LLAMA_N_CTX` (default 2048). Stop
  (`POST /api/chat/stop`) and Clear are on the page. `/api/predict` is
  unchanged and still uses `MAX_NEW_TOKENS` (40). If llama-cpp-python is
  missing or no chat model is downloaded and selected, the page shows a
  short message, the install command from `requirements-optional.txt`, and a
  link to the model page, and does not crash.
* See the README section "Downloading models" for the CLI and
  troubleshooting (MODEL_WEIGHTS, HF_HUB_OFFLINE, disk space, proxies).

## Roadmap

1. Fix the UI bugs Robin found in the model picker page (see Known issues).
2. Map viewer follow-ups: labels and a proper style for OS Open Zoomstack /
   OpenMapTiles packs, search by place name, GPS position, offline routing.
3. Chat-style page for the GGUF models: done (`/chat`, streaming, history, system prompt).
4. Render Markdown guides (small vendored or stdlib renderer); real guides
   with cited sources.
5. Video thumbnails (ffmpeg optional) and a nicer browse-by-topic page.
6. Optional: full-text search index (SQLite FTS5) across guides and video notes.
7. Packaging for a 4 GB laptop: one start script, content on external drive.

## Known issues

* **UI bugs in the model picker page** (from the merged PR #2) were seen by
  Robin in the browser and are **not fixed yet**; this structure work leaves
  that page as it was apart from the shared navigation bar. The picker now
  has bare Download / Use buttons and a progress line per model.
* Download jobs live in memory: restarting the app forgets their history
  (files on disk are still detected). There is no cancel button yet; stop the
  app to stop a download. With huggingface_hub 2.x an interrupted file
  restarts from zero (Xet transfers do not resume); partial leftovers are
  cleaned on the next attempt.
* The new section pages are deliberately bare (plain lists, raw-text guides).
* The map viewer is bare: one pack at a time, no place search, and the
  generic style for non-Protomaps vector packs has no labels.
* `scripts/build_video_manifest.py` drops entries whose files are gone.
