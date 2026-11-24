# Notes: `kalypso_app.py`

This file sets up Kalypso on Modal and runs the baseline benchmark. It defines a container image with PyTorch and Kalypso installed, a `verify()` function that checks the install on an A100, a `download_model()` function that saves the model weights to Modal storage, and two benchmark functions that start Kalypso's server and run a workload: `run_semeval()` for our SemEval-2014 restaurant sample and `run_medec()` for Kalypso's own MEDEC benchmark.

## How to run it

From the root of the ERSP repo. The part after `::` picks which entrypoint to run.

```powershell
python -m modal run experiments/modal/kalypso_app.py::check      # verify the install
python -m modal run experiments/modal/kalypso_app.py::download   # download model weights (once)
python -m modal run experiments/modal/kalypso_app.py::semeval    # SemEval restaurant sample
python -m modal run experiments/modal/kalypso_app.py::medec      # Kalypso's MEDEC benchmark
```

`::semeval` needs the sample CSV first. Build it with `python experiments/scripts/prepare_semeval_sample.py` (see `experiments/scripts/notes/prepare_semeval_sample.md`). To use a different sample file: `...::semeval --csv semeval2014_restaurants_train_sample500_seed7.csv`.

Before `download` works, you need two things on Hugging Face: approved access to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (accept the license on that page), and a token stored in Modal as a secret named `huggingface`:

```powershell
python -m modal secret create huggingface HF_TOKEN=hf_yourtoken
```

A fine-grained token with read access to just that model is enough. Each person uses their own token. It never goes in the code or the repo.

The first run of anything builds the image, which took us about 10 to 20 minutes. The build happens on Modal's CPU machines, so it's cheap. After that Modal reuses the cached image and starts in seconds, unless you change the image recipe or anything in `kalypso/`.

## This file runs in two places

When you type `python -m modal run ...`, Python reads the file on your own computer. That side only describes what we want (the image recipe and which functions exist) and sends it to Modal. Nothing gets installed on your machine.

Modal then builds the image, starts a container from it on a machine with an A100, loads this same file inside the container, and runs `verify()` there. The output streams back to your terminal. (The other GPU and CPU functions work the same way.)

| Part | What it is | Where it runs |
|---|---|---|
| `image = ...` | Recipe for the software environment | Built on Modal |
| `verify()`, `run_semeval()`, `run_medec()` | Code that runs on the GPU | Modal, on an A100 |
| `download_model()` | Saves model weights to Modal storage | Modal, CPU only |
| `check()`, `download()`, `semeval()`, `medec()` | Entrypoints that start the remote functions and save results | Your computer |

## Section by section

### Pinned versions

```python
PYTHON_VERSION = "3.12"
TORCH_INDEX = "https://download.pytorch.org/whl/cu129"
VLLM_BASE_COMMIT = "55f1fc1b..."  # vLLM v0.13.0rc4
KALYPSO_COMMIT = "a77c625f..."     # upstream Kalypso
```

These are all the versions that matter, kept at the top so they're easy to find.

- Python 3.12 is supported by both vLLM 0.13 and PyTorch 2.9.0. Newer versions like 3.14 aren't. Your own Python version doesn't matter, since locally it only runs the Modal CLI.
- `TORCH_INDEX` is where PyTorch hosts its CUDA 12.9 builds. PyTorch publishes a separate build for each CUDA version, and this URL picks one.
- `VLLM_BASE_COMMIT` is the vLLM version Kalypso is built on.
- `KALYPSO_COMMIT` is the upstream Kalypso commit our `kalypso/` folder came from (via git subtree).

Change these together. Bumping PyTorch alone, for example, would break vLLM's compiled code.

### Model and server settings

```python
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SERVER_PORT = 8003
SERVER_ARGS = [...]
```

`MODEL_ID` is the Hugging Face ID of the model. The token only proves who is downloading; this string is what picks the model. To try another model, change it here.

We use Llama 3.1 8B because Kalypso's benchmarks default to Llama 3.3 70B, which needs about 140 GB of GPU memory and won't fit on one A100. The 8B model takes about 16 GB, it's from the same family, and Kalypso's own code already uses it as its secondary model. Results from an 8B model won't necessarily carry over to 70B.

`SERVER_ARGS` are the flags passed to `vllm serve`. The first four come from Kalypso's README (port, 90% GPU memory, prefix caching on, 32k max context). The other three turn on stats that Kalypso reads after each query:

- `--enable-prompt-tokens-details` gives per-operator cached-token counts.
- `--kv-cache-metrics` with `--kv-cache-metrics-sample 1.0` counts KV-cache block evictions. Without the sample setting at 1.0, the count is only a sample.

Port 8003 is what Kalypso's benchmark clients connect to by default.

### Storage for model weights and the Hugging Face token

```python
models_volume = modal.Volume.from_name("kalypso-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
```

A Modal Volume is storage that sticks around between runs. Containers are wiped when they stop, so without a Volume we'd download about 16 GB of weights every run. The Volume gets mounted at `/models`, and the last step of the image sets `HF_HUB_CACHE=/models/hub` so Hugging Face's download code saves there.

`hf_secret` refers to the secret each person creates with `modal secret create`. Functions that list it in `secrets=[...]` get `HF_TOKEN` as an environment variable, which is where Hugging Face's library looks for it.

### Finding the repo folders

```python
if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    KALYPSO_DIR = REPO_ROOT / "kalypso"
    SCRIPTS_DIR = REPO_ROOT / "experiments" / "scripts"
    DATA_DIR = REPO_ROOT / "experiments" / "data"
    DATA_DIR.mkdir(exist_ok=True)
else:
    KALYPSO_DIR = Path("/root/kalypso")
    SCRIPTS_DIR = Path("/root/experiments/scripts")
    DATA_DIR = Path("/root/experiments/data")
```

`__file__` is the path of this script, `.../ERSP/experiments/modal/kalypso_app.py`. `.parents[2]` goes up three levels (`modal`, then `experiments`, then `ERSP`). With `Path` objects, `/` joins paths instead of dividing. This way it works wherever someone cloned the repo.

The `if` is there because Modal also loads this file inside the container, where it sits at `/root/kalypso_app.py`. There aren't enough parent folders there, so `.parents[2]` would crash. `modal.is_local()` is only `True` on your own machine. Inside the container, the same names point to where those folders get placed.

`experiments/data/` is gitignored, so it won't exist on a fresh clone. The `mkdir` creates it so the image definition below doesn't fail.

### The image

```python
image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install(...)
    .pip_install(...)
    .env(...)
    .add_local_dir(...)
    .run_commands(...)
)
```

An image is a snapshot of a Linux system with software installed. Each `.something(...)` call adds a step and returns the updated recipe, which is why they can be chained. Modal caches every step, so changing a later step reuses everything before it. The slow steps that rarely change go first for that reason.

1. `debian_slim(...)` starts from a minimal Debian Linux with Python 3.12.
2. `apt_install("git", "build-essential")` adds system tools. vLLM's install calls `git`, and some Python packages need C/C++ compilers.
3. `pip_install("torch==2.9.0", ...)` installs PyTorch 2.9.0 with matching `torchvision` and `torchaudio`, from the CUDA 12.9 index. Kalypso's requirements pin `torch==2.9.0`, and vLLM 0.13 targets CUDA 12.9. PyTorch is about 1 GB, so it gets its own step and stays cached when later steps change.
4. `.env({...})` sets environment variables that vLLM's installer reads.
   - `VLLM_USE_PRECOMPILED=1` downloads vLLM's prebuilt GPU kernels instead of compiling them, which would take an hour or more. This is safe because Kalypso only changes Python files. We diffed it against upstream, and its `csrc/` folder (the CUDA code) is identical to vLLM v0.13.0rc4.
   - `VLLM_PRECOMPILED_WHEEL_COMMIT` picks which kernels to download. They have to match the vLLM version Kalypso is based on.
   - `SETUPTOOLS_SCM_PRETEND_VERSION` sets the version number directly. The installer normally reads it from Git history, but `kalypso/` came in through git subtree and has no `.git` of its own. Without this the install fails with a version error.
5. `add_local_dir(...)` copies `ERSP/kalypso` into the image at `/root/kalypso`. `copy=True` makes the files part of the image, which the install step needs. CI configs and docs are skipped. Since this comes after PyTorch, editing Kalypso code only reruns this step and the next one.
6. `run_commands(...)` runs the install steps from Kalypso's README: upgrade pip and the build tools, install `requirements/build.txt` and `requirements/common.txt`, then `pip install -e . --no-build-isolation`.
   - `-e` (editable) makes Python run the code straight from `/root/kalypso`, so Kalypso's modified vLLM is what runs.
   - `--no-build-isolation` tells the build to use the PyTorch that's already installed instead of downloading a separate copy.
   - `--extra-index-url` lets pip also check PyTorch's site, so it doesn't replace our PyTorch with a different build.
7. `.env({"HF_HUB_CACHE": ...})` tells Hugging Face to keep downloaded models on the Volume (more on that below).
8. Two more `add_local_dir` calls attach `experiments/scripts/` and `experiments/data/` (minus the raw XML). These don't use `copy=True`, so Modal attaches them when a container starts instead of building them into the image. Editing a script or making a new sample doesn't trigger a rebuild.

### The app

```python
app = modal.App("kalypso-baseline", image=image)
```

This groups our functions under one name, `kalypso-baseline` in the Modal dashboard. Every function in the app runs inside the image above.

### `verify()`, which runs on the GPU

```python
@app.function(gpu="A100-80GB", timeout=600)
def verify() -> dict:
```

The `@app.function(...)` decorator replaces `verify` with a Modal Function object. That object keeps the code along with the settings (A100 80GB, 10-minute timeout, the image) and has its own methods: `.remote()` runs it on Modal, `.local()` runs it on your machine, and `.spawn()` starts it on Modal without waiting for the result. These come from the `modal` library, in the `modal._functions` module.

When you call `verify.remote()`, Modal registers the app on its servers, sends a request with the function's ID and arguments (packed with Python's `pickle`), starts a container from the image on an A100, imports this file there, finds `verify` by name, and runs it. The return value gets pickled and sent back.

The imports are inside the function because `torch` and `vllm` only exist in the image. With `import torch` at the top of the file, it would crash on your computer before anything reached Modal.

The checks, in order:

- `torch.cuda.is_available()` tells us whether PyTorch can use the GPU. If it's `False`, nothing else will work.
- GPU name, memory, and driver version tell us what hardware a run used. The driver version comes from `nvidia-smi`, called through `subprocess`.
- `vllm_location` should be `/root/kalypso/vllm`. Anything else means a stock vLLM got installed over Kalypso's, and we'd be measuring the wrong system.
- `import vllm._C` loads the compiled GPU code we downloaded. This is where a mismatch between those kernels and our PyTorch would show up, so it's the check most likely to fail.
- `import vllm.kalypso` confirms Kalypso's own code loads.
- The versions of `transformers`, `flashinfer-python`, `xformers`, and `triton` get recorded because they affect performance. The `try`/`except` stores `None` for a package that isn't installed instead of crashing.

At the end the dictionary is printed as JSON and returned. Before returning, it goes through `json.dumps` and `json.loads` so it only holds plain values (strings, numbers, booleans, null). Some values look like strings but aren't. `torch.__version__`, for example, is a PyTorch type that can only be unpickled with PyTorch installed. Without this step your machine crashes with `DeserializationError: ... 'torch' module is not available in the local environment`, even though the remote run worked.

### `download_model()`

Runs on a CPU container (no `gpu=` setting, so it's cheap) with the Volume and the Hugging Face secret attached. `snapshot_download` fetches every file in the model's Hugging Face repo except `original/`, which holds a second copy of the weights in Meta's own format that vLLM doesn't use. Skipping it saves about 16 GB.

`models_volume.commit()` saves the new files so later containers can see them. The function prints the model's revision, the folder name Hugging Face uses for the exact version of the files, which we record for reproducibility.

You only need to run this once per Modal account, or again if you change `MODEL_ID`.

### `_serve_and_run()`: the shared benchmark runner

Every workload needs the same steps around it: start the server, wait for it, run a client, shut down, collect stats. This helper does that once so each workload function only has to say which client to run.

1. Starts a background thread that checks GPU memory every half second with `nvidia-smi` and keeps the peak, separately for server startup and for the workload.
2. Starts the server with `vllm serve MODEL_ID ...` as a separate process, the same way Kalypso's README does, with `VLLM_ENABLE_V1_MULTIPROCESSING=0` from the README. Weights load from the Volume; vLLM only checks with Hugging Face that they're current. Server output goes to `/tmp/server.log`.
3. Waits for `/v1/semantic/healthz` to report `query_processor_initialized: true`. Loading the model and setting up the KV cache takes a few minutes. If the server crashes during startup, the function raises an error with the end of the server log.
4. Runs the client command it was given and times it.
5. Stops the server and collects the logs, Kalypso's stats lines, and the client's output file if there is one.

The `try`/`finally` makes sure the server gets shut down even if something fails, so a crashed run doesn't sit there holding a GPU until the timeout.

What gets collected for every workload:

- `server_startup_seconds`: how long the server took to load the model and get ready. This isn't part of the workload time.
- `client_wall_seconds`: our own timing around the client process, including HTTP overhead and Python startup.
- `query_kv_stats`: Kalypso's `QUERY_KV_STATS` line from the server log, with prefix-cache tokens queried and reused, prompt and generated tokens, prefill and decode time, and evictions.
- `op_token_stats`: Kalypso's `OP_TOKEN_STATS` line, with LLM calls, prompt tokens, cached tokens, and generated tokens for each operator.
- `peak_gpu_memory_mib`: the peak from the background thread. vLLM reserves 90% of GPU memory up front for the model and KV cache (because of `--gpu-memory-utilization 0.9`), so this number mostly reflects that setting, not how much the workload needed.

Kalypso doesn't record time per operator. Its operators run as an overlapping pipeline, so it counts tokens per operator but not seconds. We report that as a limitation instead of estimating it.

`_parse_stats_lines` finds those stats lines in the server log and reads the JSON after the tag. `_run_info` records the settings for each run (time, workload, dataset, row count, model, server flags, Kalypso commit), and `_split_outputs` separates the small result summary from the bulky logs.

`GPU_FUNCTION` holds the settings both benchmark functions share (A100 80GB, the Volume, the Hugging Face secret, a one-hour timeout), so they're written once.

### `run_semeval(csv_name)`

Runs our SemEval-2014 restaurant workload: a service filter followed by a service-sentiment map, using `experiments/scripts/client_semeval_service.py` (see its notes for the query and scoring). It checks that the CSV exists, counts its rows, runs the client through `_serve_and_run`, and adds:

- `server_latency_sec`: time the server spent on the query, as reported by Kalypso
- `throughput_rows_per_sec`: input rows divided by that latency
- `scores`: filter accuracy, precision, recall, observed vs. ground-truth selectivity, and map accuracy

### `run_medec()`

Runs Kalypso's own MEDEC benchmark (a filter and two maps over clinical notes) with its client unmodified. The copy bundled with Kalypso only has 10 notes, even though the file is named `MEDEC-TrainingSet-1000.csv`. The client only prints a summary, so `response_summary` is pulled from its output.

### Entrypoints, which run on your computer

```python
@app.local_entrypoint()
def check(): ...
def download(): ...
def semeval(csv: str = "semeval2014_restaurants_train_sample200_seed42.csv"): ...
def medec(): ...
```

`@app.local_entrypoint()` marks a function `modal run` can call on your machine, picked with `::name` on the command line. `check` and `download` just start their remote functions. `semeval` and `medec` start their benchmark and pass the result to `_save_run`, which writes it to `experiments/results/runs/<name>_<timestamp>/`:

- `result.json` with the measurements and settings
- `client_stdout.txt` and `client_stderr.txt` from the benchmark client
- `server.log`, the full server output
- `client_output.json` (SemEval only), with the scores and every row's answers

Arguments to an entrypoint become command-line options, which is how `--csv` works for `semeval`.

Files are written as UTF-8 so Windows doesn't choke on characters in the logs.

## Expected output of `::check`

```json
{
  "python": "3.12.16",
  "torch": "2.9.0+cu129",
  "torch_cuda_build": "12.9",
  "cuda_available": true,
  "gpu": "NVIDIA A100-SXM4-80GB",
  "gpu_memory_gb": 85.1,
  "driver": "620.70.06",
  "vllm": "0.13.0rc4",
  "vllm_location": "/root/kalypso/vllm",
  "vllm_C_loaded": true,
  "kalypso_module_loaded": true,
  "transformers": "4.57.6",
  "flashinfer-python": "0.5.3",
  "xformers": null,
  "triton": "3.5.0"
}
```

This is from our run on 2026-11-23, saved in `experiments/results/environment_modal_2026-11-23.json`.

`gpu_memory_gb` says 85.1 instead of 80 because it counts decimal GB (10^9 bytes) rather than binary GiB. It's the same amount of memory. `xformers` being null is fine, since vLLM 0.13 uses FlashAttention and FlashInfer on A100s. The driver version can change between runs because Modal may put you on a different machine.

## Problems we hit

If PowerShell says `modal : The term 'modal' is not recognized`, the `modal` command isn't on your Windows PATH. Use `python -m modal ...` instead.

`DeserializationError: ... 'torch' module is not available` happened because `verify()` returned a PyTorch-specific object. We fixed it by converting everything to plain JSON values before returning.

Our first SemEval run failed with `500 Server Error` from `/v1/semantic/query`, and the server log showed `The model /models/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/... does not exist` once per row. We had started the server with `HF_HUB_OFFLINE=1`. With that setting, vLLM replaces the model name with the local folder path (`kalypso/vllm/engine/arg_utils.py`, around line 590). Kalypso's executor sends that internal name with every request, but the API only accepts the name the server was started with, so every request was rejected. Removing `HF_HUB_OFFLINE=1` fixed it. Don't add it back.
