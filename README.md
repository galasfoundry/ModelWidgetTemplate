# Model widget template

> **The First Book.** This repo is also the base of *The First Book*, a fully
> offline UK maps, how-to video, guides and local-model library for a 4 GB
> laptop. Open http://127.0.0.1:5000/home after `python app.py`. For a map to
> look at, run `python scripts/fetch_demo_map.py` once while online, then open
> http://127.0.0.1:5000/maps . For videos, `pip install yt-dlp` and run
> `python scripts/fetch_videos.py` once (seven Creative Commons videos on
> off-grid hydro, batteries and Starlink power, about 290 MB at 360p; credits in
> [library/videos/README.md](library/videos/README.md)). For the original
> "Head and flow" episode, copy the `.mp4` into `library/videos/originals/`
> (the sidecar and subtitles are already in the repo). Then open
> http://127.0.0.1:5000/videos . Content goes
> in `library/` (maps, videos, guides); see
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the layout, routes, how to
> add map packs and videos, the roadmap and known issues (including the
> pending UI fixes on the model picker page).

A small Flask app that loads one Hugging Face model and serves it as a local web page plus a JSON API. Clone it, set a repo id in `config/.env`, run one weight install, and open the page.

Paths are resolved from the project root (the folder that contains `app.py`), not from the working directory and not from any one computer. The layout follows the same rules as [woodsapp](https://github.com/galasfoundry/woodsapp): a project-relative `.env`, `hf_hub_download` into `weights/` at a pinned revision, and `install_weights.py` for the first download.

The shipped default is `timm/mobilenetv3_small_100.lamb_in1k` (MobileNetV3-Small, 2.5M parameters, about 10 MB). It classifies ImageNet images on CPU.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate it (`source .venv/bin/activate` on macOS and Linux, `.venv\Scripts\activate` on Windows), then install the pins in `requirements.txt`:

```bash
pip install -r requirements.txt
```

On a machine with no NVIDIA GPU, including a 4 GB laptop, install the CPU build of PyTorch first so pip does not fetch a multi-gigabyte CUDA wheel. `torch==2.6.0` and `torchvision==0.21.0` are the same versions either way:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If the second command tries to replace the CPU build, the CPU wheels are already enough and the rest of the file can be installed around them.

Fetch the weights:

```bash
python install_weights.py
```

That command reads `config/.env`, then a project-root `.env` if you have one. With neither file it downloads the default MobileNet snapshot:

```text
weights/models--timm--mobilenetv3_small_100.lamb_in1k/snapshots/76b303ec28c466e9df37f4745dfb7aef20ab87ae/
```

An existing copy is reused, in this order, and nothing is downloaded:

- `MODEL_WEIGHTS`, when it is set (a checkpoint file, a snapshot directory, or a folder that contains the snapshot tree). A missing override stops the install.
- The pinned revision under `weights/`.
- The same revision in the local Hugging Face cache (`HF_HOME` / `HUGGINGFACE_HUB_CACHE`, otherwise `~/.cache/huggingface/hub`). A branch name such as `main` is followed through the cache `refs` file.
- Any other snapshot of this same repo already on disk. The pinned revision is preferred.

Starting the app downloads the same way if the snapshot is still missing:

```bash
python app.py
```

Then open http://127.0.0.1:5000/ . The first prediction loads the module when `LAZY_LOAD=1` (the default). `LAZY_LOAD=0` loads it before the server listens.

`python app.py` and `python install_weights.py` can be run from any working directory. Pass the path to the script if the shell is somewhere else.

## Swap the model

Copy the example and edit the four `MODEL_*` lines:

```bash
cp config/.env.example config/.env
```

`config/.env` is the file to keep. A root `.env` fills in any keys that `config/.env` did not set. Shell exports win over both. Do not commit either `.env`.

| Variable | Default | What it does |
| --- | --- | --- |
| `MODEL_ID` | `timm/mobilenetv3_small_100.lamb_in1k` | Hugging Face repo id |
| `MODEL_REVISION` | `76b303ec28c466e9df37f4745dfb7aef20ab87ae` | Commit SHA. Empty resolves the repo's current default and prints the SHA to pin |
| `MODEL_TASK` | `image-classification` | `image-classification`, `text-generation`, or `text-classification` |
| `MODEL_BACKEND` | `timm` for a `timm/` image repo, otherwise `transformers` | `timm`, `transformers`, or `auto` |
| `DEVICE` | `auto` | `auto` (CUDA, then MPS, then CPU), or `cpu`, `cuda`, `mps` |
| `TORCH_DTYPE` | `float32` | `float32`, `float16`, or `bfloat16` |
| `ALLOW_CPU_HALF` | `0` | Keep float16 or bfloat16 when the device is CPU |
| `LOW_CPU_MEM_USAGE` | `1` | Pass `low_cpu_mem_usage` into transformers `from_pretrained` |
| `LAZY_LOAD` | `1` | Load on the first request |
| `TOP_K` | `5` | Classes returned for classifiers |
| `MAX_NEW_TOKENS` | `40` | Tokens appended by the generator |
| `MAX_INPUT_CHARS` | `4000` | Longest prompt accepted |
| `MAX_UPLOAD_MB` | `8` | Largest image upload |
| `TORCH_NUM_THREADS` | unset | Cap PyTorch CPU threads |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Bind address |
| `WEIGHTS_DIR` | `weights` | Where downloads are stored |
| `MODEL_WEIGHTS` | unset | Use this file or directory and do not download |
| `LABEL_MAP_PATH` | unset | Optional class-name JSON. A 1000-way timm head uses ImageNet-1k names |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | the default hub cache | Extra place to reuse a snapshot |
| `HF_TOKEN` | unset | Token for a gated or private repo |

After editing the repo id, run `python install_weights.py` again. The page asks `/api/info` and shows an image upload or a text box to match the task.

Presets with pinned revisions are commented in `config/.env.example`:

- `google/mobilenet_v2_1.0_224` at `75e607b00aeae1297cc89d026a118bce012f5c5a`, image classification through transformers, about 14 MB.
- `roneneldan/TinyStories-1M` at `77f1b168e219585646439073245fe87e56b3023e`, text generation, a few MB. The prose is simple. It is a real language model, small enough to load beside the browser on a 4 GB machine.
- `Facececersek/bedrock-sentiment-tiny` at `16175109a6e476cee0c388d2aac37429ecf964dd`, text classification, about 17 MB. It is a 2-layer BERT fine-tuned for positive or negative sentiment. For a stronger SST-2 model, use `distilbert/distilbert-base-uncased-finetuned-sst-2-english` at `714eb0fa89d2f80546fda750413ed43d93601a13` (about 260 MB).
- `timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21` at `72a53b6ad40843f079d85b2ee315f5fb222e2f0f`, the woodsapp ViT-L checkpoint, about 1.2 GB. Set `LABEL_MAP_PATH` to a 10,000-class JSON map or the results are `class_0`, `class_1`, and so on.

`LABEL_MAP_PATH` accepts three JSON shapes:

```json
["tench", "goldfish"]
```

```json
{"0": "tench", "1": {"name": "goldfish", "description": "a fish"}}
```

```json
{"label_map": {"0": {"name": "Oak", "description": "Quercus"}}}
```

The third shape is the woodsapp app-config object. Descriptions are returned on each result row and shown under the label.

## Downloading models

Models download from Hugging Face **into this project's `weights/` folder**
(`weights/models--<org>--<name>/snapshots/<revision>/`), once, while online.
After that the app uses them with no internet. Nothing downloads at start-up:
until you download a model (or run a first prediction), `weights/` stays empty.

**Easiest on a laptop: the command line.**

```bash
python scripts/download_models.py --list                  # catalog, sizes, what is on disk
python scripts/download_models.py mobilenetv3-small gemma-3-270m-it-gguf
python scripts/download_models.py --all --dry-run         # what --all would fetch
python scripts/download_models.py --all                   # everything that fits this machine
```

It shows progress, skips anything already on disk without going online, and
can simply be run again after an interruption. GGUF chat models fetch **one**
quantised file (e.g. `gemma-3-270m-it-Q4_K_M.gguf`, ~253 MB), not the whole
repo.

**From the page.** Each model under **Models** has a **Download** button. The
download runs in the background (one at a time); its progress shows under the
model and nothing else is blocked. When it is on disk, **Use this model**
switches to it. The same thing over HTTP:

```bash
curl -X POST http://127.0.0.1:5000/api/models/gemma-3-270m-it-gguf/download   # 202, starts a job
curl http://127.0.0.1:5000/api/models/gemma-3-270m-it-gguf/status             # queued/downloading/done/error, bytes, %
curl http://127.0.0.1:5000/api/models/downloads                               # every job
```

**Chat models (GGUF).** Catalog entries with `"status": "downloadable"` and
`"backend": "llama_cpp"` (Gemma 3 270M, SmolLM2 360M, Qwen2.5 0.5B, Llama 3.2
1B) always download. Using them needs llama.cpp's Python binding; a prebuilt
CPU wheel avoids compiling:

```bash
pip install llama-cpp-python --prefer-binary \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Then **Use this model** works and `config/.env` gets `MODEL_FILE=<the .gguf>`
alongside the other `MODEL_*` keys. `MAX_NEW_TOKENS` (default 40) sets the
reply length; `LLAMA_N_CTX` (default 2048) the context. On a 4 GB laptop the
270M-500M models are comfortable; Llama 3.2 1B (~808 MB) works but shows a
warning: close other apps first.

### Troubleshooting downloads

| Symptom | Cause and fix |
| --- | --- |
| "HF_HUB_OFFLINE is set" | `HF_HUB_OFFLINE=1` (or `TRANSFORMERS_OFFLINE=1`) blocks all downloads. Unset it while downloading; set it again afterwards if you like. |
| "MODEL_WEIGHTS is set, so switching is turned off" | `MODEL_WEIGHTS` in the shell, `config/.env` or `.env` pins one checkpoint. Downloads still work; remove it to switch models. |
| "Not enough disk space" | The download needs its size plus 10% free in `weights/`. Free space, or move `weights/` with `WEIGHTS_DIR=/path/on/bigger/disk`. |
| "check the internet connection" | No network, DNS, or a proxy. Behind a proxy, export `HTTPS_PROXY=http://host:port` before starting the app or the script. |
| Download stopped half-way | Run it again. Leftover `*.incomplete` files older than a minute are removed first. With huggingface_hub 2.x the interrupted file restarts from zero (its Xet transfer does not resume a file), but finished files are kept. |
| "401/403" or "gated" | That repo needs a Hugging Face account token: put `HF_TOKEN=...` in `config/.env`. The shipped catalog only uses public repos. |
| GGUF model "cannot run here: backend 'llama_cpp' is not installed" | Install `llama-cpp-python` as above. The file is already downloaded. |
| Files also in `~/.cache/huggingface` | Earlier versions or other tools may have used the global cache. This app reads both but downloads only into `weights/`. |

## Pick a model from the page

`config/models_catalog.json` is a small offline catalog of public models that fit a low-resource machine (set `MODELS_CATALOG_PATH` to use another file). The page lists the ones this machine can run under **Models**, and `GET /api/models` returns the same data.

Each entry is checked against this machine before it is offered:

- `min_ram_gb` against total RAM (10% slack, since a "2 GB" machine reports a little under 2 GiB),
- `approx_size_mb` against free disk at `weights/` (twice the size, skipped when the snapshot is already there),
- the backend (`timm`, `transformers` or `llama_cpp`) is importable and an adapter is registered for the task. A missing backend only stops *using* a model; it can still be downloaded.

RAM and disk come from `psutil` when it is installed (`pip install -r requirements-optional.txt`). Without it, Linux reads `/proc/meminfo` and disk space comes from `shutil.disk_usage`. CUDA is reported from `torch.cuda.is_available()`.

Choosing a model (`POST /api/models/select` with `{"id": "<catalog id>"}`):

1. reuses the exact pinned snapshot when it is already under `weights/` or in the Hugging Face cache, so it works with no network,
2. otherwise downloads it first (the page offers **Download** separately, see [Downloading models](#downloading-models); select refuses with 409 while that model's background download is still running),
3. loads it in a fresh `ModelService` and switches the page and `/api/predict` to it,
4. writes `MODEL_ID`, `MODEL_REVISION`, `MODEL_TASK`, `MODEL_BACKEND` and `MODEL_FILE` (GGUF only, otherwise empty) to `config/.env` (other lines are kept), so the next start uses the same model.

If the download or the load fails, the error is shown, nothing is written, and the previous model stays active. Selection is turned off while `MODEL_WEIGHTS` is set. Shell exports still beat `config/.env`, and the response warns when one would.

## API

`GET /api/health` returns `{"status": "ok"}`.

`GET /api/info` returns the configured repo, task, backend, device, dtype, and whether the module is loaded. The page uses this to choose an image input or a text input.

`GET /api/models` returns the catalog filtered for this machine: `models` (can run here), `excluded` (with `reasons`), `capabilities`, `active`, and `installed` (catalog ids whose pinned snapshot is on disk).

`POST /api/models/<id>/download` starts a background download into `weights/` and returns 202 with the job (200 if already on disk, 404 unknown id, 409 planned or too big for this machine). `GET /api/models/<id>/status` returns that job: `state` (`idle`, `queued`, `downloading`, `done`, `error`), `bytes_done`, `bytes_total`, `percent`, `current_file`, `message`, `error`, `installed`, `snapshot`. `GET /api/models/downloads` lists every job and how many are active. Rows in `GET /api/models` also carry `downloadable`, `warnings` and their `download` job.

`POST /api/models/select` switches to a catalog model, as described above. Errors are 404 for an unknown id, 409 when it cannot run here or `MODEL_WEIGHTS` is set, 502 when it is missing and cannot be downloaded, and 500 when it will not load.

`POST /api/predict` runs the model.

```bash
curl -s -F "image=@photo.jpg" http://127.0.0.1:5000/api/predict
```

```bash
curl -s -H "Content-Type: application/json" \
  -d '{"text":"Once upon a time"}' \
  http://127.0.0.1:5000/api/predict
```

A classifier response looks like this:

```json
{
  "model_id": "timm/mobilenetv3_small_100.lamb_in1k",
  "task": "image-classification",
  "backend": "timm",
  "device": "cpu",
  "dtype": "float32",
  "results": [
    {"label": "goldfish", "score": 0.42, "rank": 1}
  ]
}
```

A generator returns the full decoded string in `text` and again as `results[0].text`. Errors are JSON, with status 400 for a bad upload or an empty prompt and 500 when the model itself fails. The server log keeps the traceback.

## Add a task adapter

Adapters live in `modelwidget/adapters/`. Each one is a class with `load` and `predict`, registered for a task name and a backend name.

1. Add `modelwidget/adapters/my_task.py`:

```python
from pathlib import Path

import torch

from modelwidget.adapters.base import ModelAdapter, register


@register("my-task", "transformers")
class MyTaskAdapter(ModelAdapter):
    input_kind = "text"  # or "image"

    def __init__(self, settings, snapshot: Path):
        self.settings = settings
        self.snapshot = Path(snapshot)
        self.model = None

    def load(self, *, device: str, dtype: torch.dtype) -> None:
        # Read files only from self.snapshot. local_files_only=True keeps
        # the process off the network after install_weights.py has run.
        ...

    def predict(self, *, image=None, text: str | None = None) -> dict:
        return {
            "results": [{"label": "example", "score": 1.0, "rank": 1}],
            "text": None,
        }
```

2. Import the class from `modelwidget/adapters/__init__.py` so the `@register` decorator runs.
3. Set `MODEL_TASK=my-task` and `MODEL_BACKEND=transformers` in `config/.env`.

`input_kind` chooses the form. `"image"` sends a multipart file field named `image`. `"text"` sends JSON `{"text": "..."}`. `predict` returns `results` (a list of `{label, score, rank}`, plus optional `description` or `text`) and a top-level `text` field for generation.

`ModelService` in `modelwidget/service.py` calls `ensure_weights`, then `load`, then `predict`. A lock serializes those calls so two browser requests cannot load two copies. The page and `/api/predict` both go through that service.

The timm image adapter reads `architecture`, `num_classes`, and `pretrained_cfg` from the snapshot `config.json`, builds the module with `pretrained=False`, and loads `model.safetensors` (or `pytorch_model.bin` when that is all the repo published). Normalization and crop come from `pretrained_cfg`. The transformers adapters call `from_pretrained` on the snapshot directory with `low_cpu_mem_usage`.

## Hardware

PyTorch is the part that uses memory. The default MobileNet weights are about 10 MB. On a 4 GB laptop the practical setup is the CPU wheel above, `DEVICE=auto` or `DEVICE=cpu`, `TORCH_DTYPE=float32`, and `LAZY_LOAD=1`. `TORCH_NUM_THREADS=2` keeps the runtime from starting one thread per core. An enlarged swap file gives the first load somewhere to spill, the same way a larger vision model already needs it.

`LOW_CPU_MEM_USAGE=1` asks transformers to stream the checkpoint into the module. The timm loader reads safetensors onto CPU and only then moves the module, so a second full copy is not kept on GPU during the read.

`TORCH_DTYPE=float16` (or `bfloat16`) is for CUDA and MPS. On CPU the app stays on float32, because half-precision convolutions there are often unsupported or slower. `ALLOW_CPU_HALF=1` keeps the requested dtype anyway.

One model is loaded at a time. Pick another one on the page, or change `MODEL_ID` and restart the process. A ViT-L checkpoint around 1.2 GB needs a machine with room for the weights, the framework, and the activations. On a 4 GB machine that means swap and a quiet desktop. On a GPU machine, `DEVICE=auto` selects CUDA when it is present, and float16 cuts the weight storage. The install path does not change: the same `install_weights.py` call writes the snapshot under `weights/`.

Gated repos need `HF_TOKEN` in the environment or in `config/.env`. Public repos used by the defaults do not.

## Tests

```bash
python -m pytest
```

`tests/test_paths.py`, `tests/test_labels.py`, `tests/test_registry.py`, `tests/test_api.py`, `tests/test_models.py`, the `tests/test_firstbook_*.py` files and `tests/test_offline.py` stay offline (`python -m pytest --ignore=tests/test_smoke.py` runs only those). `tests/test_smoke.py` downloads the four tiny presets into `weights/` the first time, runs each one through `POST /api/predict`, and reuses the snapshots after that. The weight tree is gitignored.

## Layout

```text
app.py                      Flask routes
install_weights.py          download or reuse the configured snapshot
paths.py                    portable paths, cache search, hf_hub_download
config/.env.example         model id, revision, task, device
config/models_catalog.json  offline model catalog for the picker
modelwidget/settings.py     environment values
modelwidget/service.py      lazy load and the request lock
modelwidget/capabilities.py RAM, disk, CUDA and backend checks
modelwidget/catalog.py      catalog filtering, install-if-missing, config/.env writes
modelwidget/adapters/       image and text adapters
templates/index.html        upload or text form
static/                     page styles and the fetch client
weights/                    local snapshots, not committed
firstbook/                  The First Book: library paths, maps, videos, guides, routes
config/maps_catalog.json    known UK map packs, sources and licences
config/videos_catalog.json  Creative Commons videos with credits
modelwidget/gguf.py         llama.cpp backend for GGUF models (optional)
scripts/                    download_models.py, fetch_demo_map.py, fetch_videos.py, build_video_manifest.py
library/                    offline content: maps/, videos/, guides/
docs/ARCHITECTURE.md        The First Book structure and roadmap
```
