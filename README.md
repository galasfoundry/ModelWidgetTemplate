# Model widget template

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

## API

`GET /api/health` returns `{"status": "ok"}`.

`GET /api/info` returns the configured repo, task, backend, device, dtype, and whether the module is loaded. The page uses this to choose an image input or a text input.

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

One model is loaded at a time. Changing `MODEL_ID` and restarting the process is how you switch. A ViT-L checkpoint around 1.2 GB needs a machine with room for the weights, the framework, and the activations. On a 4 GB machine that means swap and a quiet desktop. On a GPU machine, `DEVICE=auto` selects CUDA when it is present, and float16 cuts the weight storage. The install path does not change: the same `install_weights.py` call writes the snapshot under `weights/`.

Gated repos need `HF_TOKEN` in the environment or in `config/.env`. Public repos used by the defaults do not.

## Tests

```bash
python -m pytest
```

`tests/test_paths.py`, `tests/test_labels.py`, `tests/test_registry.py`, and `tests/test_api.py` stay offline. `tests/test_smoke.py` downloads the four tiny presets into `weights/` the first time, runs each one through `POST /api/predict`, and reuses the snapshots after that. The weight tree is gitignored.

## Layout

```text
app.py                      Flask routes
install_weights.py          download or reuse the configured snapshot
paths.py                    portable paths, cache search, hf_hub_download
config/.env.example         model id, revision, task, device
modelwidget/settings.py     environment values
modelwidget/service.py      lazy load and the request lock
modelwidget/adapters/       image and text adapters
templates/index.html        upload or text form
static/                     page styles and the fetch client
weights/                    local snapshots, not committed
```
