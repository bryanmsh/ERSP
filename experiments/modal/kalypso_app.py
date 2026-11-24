"""Kalypso on Modal: container image, environment check, and baseline runs.

Run from the root of the ERSP repo. Each command runs one entrypoint:
    python -m modal run experiments/modal/kalypso_app.py::check     # verify the install
    python -m modal run experiments/modal/kalypso_app.py::download  # get model weights (once)
    python -m modal run experiments/modal/kalypso_app.py::semeval   # SemEval restaurant sample
    python -m modal run experiments/modal/kalypso_app.py::medec     # Kalypso's MEDEC benchmark

The first run builds the image, which takes a while. Modal caches it after
that, so later runs start quickly unless the image definition changes.

The image is:
  - Debian with Python 3.12
  - PyTorch 2.9.0 built for CUDA 12.9 (Kalypso pins torch==2.9.0)
  - Kalypso installed from the kalypso/ folder in this repo, following the
    upstream README install steps

We set VLLM_USE_PRECOMPILED=1 so the build downloads vLLM's compiled CUDA
kernels instead of compiling them, which saves about an hour. This is safe
because Kalypso only changes Python files. Its csrc/ folder (the CUDA code)
is identical to vLLM v0.13.0rc4.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

# Pinned versions. Change these together, not one at a time.
PYTHON_VERSION = "3.12"
TORCH_INDEX = "https://download.pytorch.org/whl/cu129"
VLLM_BASE_COMMIT = "55f1fc1b1b4044e5b5055c968cf6d956a865476f"  # vLLM v0.13.0rc4
KALYPSO_COMMIT = "a77c625f88ef878825e9c3f43df13ca809433ecf"  # upstream Kalypso

# Model and server settings for baseline runs.
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SERVER_PORT = 8003  # the port Kalypso's benchmark clients use by default
SERVER_ARGS = [
    "--port", str(SERVER_PORT),
    "--gpu-memory-utilization", "0.9",
    "--enable-prefix-caching",
    "--max-model-len", "32768",
    # Needed for Kalypso's per-operator cached-token counts.
    "--enable-prompt-tokens-details",
    # Needed for Kalypso's KV eviction counts. Sample 1.0 = count every block.
    "--kv-cache-metrics",
    "--kv-cache-metrics-sample", "1.0",
]
BENCH_DIR = "/root/kalypso/vllm/kalypso/benchmark"

# Model weights live in a Modal Volume (persistent storage) so they download
# once instead of on every run.
MODELS_DIR = "/models"
models_volume = modal.Volume.from_name("kalypso-models", create_if_missing=True)
# Modal secret holding HF_TOKEN. Each person creates their own with:
#   python -m modal secret create huggingface HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface")

# This file also gets imported inside the container, where the repo layout
# doesn't exist, so only look up the local kalypso/ folder when running locally.
if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    KALYPSO_DIR = REPO_ROOT / "kalypso"
    SCRIPTS_DIR = REPO_ROOT / "experiments" / "scripts"
    DATA_DIR = REPO_ROOT / "experiments" / "data"
    DATA_DIR.mkdir(exist_ok=True)  # gitignored, so it may not exist yet on a fresh clone
else:
    KALYPSO_DIR = Path("/root/kalypso")
    SCRIPTS_DIR = Path("/root/experiments/scripts")
    DATA_DIR = Path("/root/experiments/data")

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "build-essential")
    # PyTorch goes in its own step since it's the biggest download. That way
    # Modal can reuse it when later steps change.
    .pip_install(
        "torch==2.9.0", "torchvision==0.24.0", "torchaudio==2.9.0",
        index_url=TORCH_INDEX,
    )
    .env(
        {
            "VLLM_USE_PRECOMPILED": "1",
            "VLLM_PRECOMPILED_WHEEL_COMMIT": VLLM_BASE_COMMIT,
            # kalypso/ was added with git subtree, so it has no .git folder
            # for the build to read its version from. Set it directly.
            "SETUPTOOLS_SCM_PRETEND_VERSION": "0.13.0rc4",
        }
    )
    # CI configs and docs aren't needed to run anything.
    .add_local_dir(
        KALYPSO_DIR,
        "/root/kalypso",
        copy=True,
        ignore=[".buildkite", ".github", "docs"],
    )
    # Same install steps as the Kalypso README. The -e (editable) install
    # means Python runs the files in /root/kalypso directly.
    .run_commands(
        "pip install -U pip 'setuptools>=77.0.3,<81' wheel ninja cmake packaging",
        f"cd /root/kalypso && pip install -r requirements/build.txt --extra-index-url {TORCH_INDEX}",
        f"cd /root/kalypso && pip install -r requirements/common.txt --extra-index-url {TORCH_INDEX}",
        f"cd /root/kalypso && pip install -e . --no-build-isolation --extra-index-url {TORCH_INDEX}",
    )
    # Tell Hugging Face to keep downloaded models on the Volume.
    .env({"HF_HUB_CACHE": f"{MODELS_DIR}/hub"})
    # Our own scripts and datasets. Without copy=True these are attached when a
    # container starts instead of being built into the image, so editing them
    # doesn't trigger a rebuild.
    .add_local_dir(SCRIPTS_DIR, "/root/experiments/scripts")
    .add_local_dir(DATA_DIR, "/root/experiments/data", ignore=["raw"])
)

app = modal.App("kalypso-baseline", image=image)


@app.function(gpu="A100-80GB", timeout=600)
def verify() -> dict:
    """Check that the install works on a GPU and print the versions in use."""
    import importlib.metadata as md
    import platform
    import subprocess

    import torch

    info = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if torch.cuda.is_available() else None,
        "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "kalypso_commit": KALYPSO_COMMIT,
        "vllm_base_commit": VLLM_BASE_COMMIT,
    }

    import vllm
    info["vllm"] = str(vllm.__version__)
    # Should be /root/kalypso/vllm. Anything else means a stock vLLM got
    # installed over Kalypso's version.
    info["vllm_location"] = str(Path(vllm.__file__).parent)

    # vllm._C is the compiled CUDA code. If the downloaded kernels don't match
    # the installed PyTorch, this import is where it breaks.
    import vllm._C  # noqa: F401
    info["vllm_C_loaded"] = True

    import vllm.kalypso  # noqa: F401
    info["kalypso_module_loaded"] = True

    for pkg in ["transformers", "flashinfer-python", "xformers", "triton"]:
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = None

    print(json.dumps(info, indent=2, default=str))
    # Return plain JSON types only. Some values (like torch.__version__) are
    # library-specific objects, and the local machine can't unpack them without
    # that library installed.
    return json.loads(json.dumps(info, default=str))


@app.function(
    volumes={MODELS_DIR: models_volume},
    secrets=[hf_secret],
    timeout=3600,
)
def download_model() -> dict:
    """Download the model weights to the Volume. Runs on CPU, no GPU needed."""
    from huggingface_hub import snapshot_download

    # original/ holds a second copy of the weights in Meta's own format, which
    # vLLM doesn't use. Skipping it saves about 16 GB.
    path = snapshot_download(MODEL_ID, ignore_patterns=["original/*"])
    models_volume.commit()  # save changes so other containers can see them
    # The last folder name in the path is the model's exact Hugging Face revision.
    info = {"model": MODEL_ID, "revision": Path(path).name, "path": path}
    print(json.dumps(info, indent=2))
    return info


def _gpu_memory_used_mib() -> int:
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def _wait_for_server(proc, log_path: str, timeout_s: int = 1200):
    """Poll Kalypso's health endpoint until its query processor is ready."""
    import requests

    url = f"http://localhost:{SERVER_PORT}/v1/semantic/healthz"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = Path(log_path).read_text(errors="replace")[-4000:]
            raise RuntimeError(f"Server exited early. Last log lines:\n{tail}")
        try:
            r = requests.get(url, timeout=5)
            if r.ok and r.json().get("query_processor_initialized"):
                return
        except requests.RequestException:
            pass  # server not listening yet
        time.sleep(5)
    raise TimeoutError(f"Server not ready after {timeout_s}s")


def _parse_stats_lines(log_text: str, tag: str) -> list:
    """Pull the JSON out of lines like '[QueryProcessor] QUERY_KV_STATS {...}'."""
    found = []
    for line in log_text.splitlines():
        marker = f"{tag} "
        if marker in line:
            try:
                found.append(json.loads(line.split(marker, 1)[1]))
            except json.JSONDecodeError:
                pass
    return found


def _serve_and_run(client_cmd: list, client_cwd: str, client_output_file: str | None = None) -> dict:
    """Start Kalypso's server, run one benchmark client against it, and collect stats.

    Shared by every workload. Returns timings, logs, Kalypso's stats lines, peak
    GPU memory, and the client's JSON output file if it wrote one.
    """
    import os
    import subprocess
    import threading

    log_path = "/tmp/server.log"

    # Track GPU memory in the background, every half second.
    peak = {"startup": 0, "workload": 0}
    phase = {"name": "startup"}
    stop = threading.Event()

    def watch_gpu():
        while not stop.is_set():
            used = _gpu_memory_used_mib()
            peak[phase["name"]] = max(peak[phase["name"]], used)
            time.sleep(0.5)

    threading.Thread(target=watch_gpu, daemon=True).start()

    # Start the server the way Kalypso's README does.
    # Don't set HF_HUB_OFFLINE=1 here. With it, vLLM swaps the model name for the
    # local folder path, Kalypso then requests that path as the model name, and
    # every request fails with "model does not exist". Weights still load from
    # the Volume without it; vLLM only checks with Hugging Face.
    env = {
        **os.environ,
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    server_start = time.perf_counter()
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            ["vllm", "serve", MODEL_ID, *SERVER_ARGS],
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
    try:
        _wait_for_server(proc, log_path)
        startup_seconds = time.perf_counter() - server_start
        print(f"Server ready after {startup_seconds:.1f}s")

        phase["name"] = "workload"
        client_start = time.perf_counter()
        client = subprocess.run(
            client_cmd, cwd=client_cwd, capture_output=True, text=True, timeout=3000,
        )
        client_seconds = time.perf_counter() - client_start
        time.sleep(2)  # let the server finish writing its stats lines
    finally:
        # Always shut the server down, even after an error, so the GPU is released.
        stop.set()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("---- client output ----")
    print(client.stdout[-6000:])
    if client.returncode != 0:
        print("---- client errors ----")
        print(client.stderr[-6000:])

    server_log = Path(log_path).read_text(errors="replace")
    client_output = None
    if client_output_file and Path(client_output_file).exists():
        client_output = json.loads(Path(client_output_file).read_text(encoding="utf-8"))

    return {
        "server_startup_seconds": round(startup_seconds, 1),
        "client_wall_seconds": round(client_seconds, 3),
        "client_returncode": client.returncode,
        "query_kv_stats": _parse_stats_lines(server_log, "QUERY_KV_STATS"),
        "op_token_stats": _parse_stats_lines(server_log, "OP_TOKEN_STATS"),
        "peak_gpu_memory_mib": dict(peak),
        "client_stdout": client.stdout,
        "client_stderr": client.stderr,
        "server_log": server_log,
        "client_output": client_output,
    }


def _run_info(workload: str, dataset: str, input_rows: int) -> dict:
    """Settings recorded with every run."""
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workload": workload,
        "dataset": dataset,
        "input_rows": input_rows,
        "model": MODEL_ID,
        "server_args": SERVER_ARGS,
        "kalypso_commit": KALYPSO_COMMIT,
    }


def _split_outputs(run: dict, result: dict) -> dict:
    """Separate the small result summary from the bulky logs before sending them back."""
    for key in ("server_startup_seconds", "client_wall_seconds", "client_returncode",
                "query_kv_stats", "op_token_stats", "peak_gpu_memory_mib"):
        result[key] = run[key]
    return {
        "result": json.loads(json.dumps(result, default=str)),
        "client_stdout": run["client_stdout"],
        "client_stderr": run["client_stderr"],
        "server_log": run["server_log"],
        "client_output": run["client_output"],
    }


GPU_FUNCTION = dict(
    gpu="A100-80GB",
    volumes={MODELS_DIR: models_volume},
    secrets=[hf_secret],
    timeout=3600,
)


@app.function(**GPU_FUNCTION)
def run_medec() -> dict:
    """Kalypso's own MEDEC benchmark (filter + two maps), unmodified client."""
    import csv
    import re
    import sys

    data_csv = f"{BENCH_DIR}/sample_data/MEDEC-TrainingSet-1000.csv"
    with open(data_csv, newline="", encoding="utf-8") as f:
        input_rows = sum(1 for _ in csv.DictReader(f))

    run = _serve_and_run(
        [sys.executable, "client_medec_filter_map_map.py",
         "--model-name", MODEL_ID, "--port", str(SERVER_PORT)],
        client_cwd=BENCH_DIR,
    )
    summary_match = re.search(r"Response Summary:\s*(\{.*?\n\})", run["client_stdout"], re.S)
    result = _run_info(
        "medec_filter_map_map",
        "MEDEC sample bundled with Kalypso (sample_data/MEDEC-TrainingSet-1000.csv)",
        input_rows,
    )
    result["response_summary"] = json.loads(summary_match.group(1)) if summary_match else None
    return _split_outputs(run, result)


@app.function(**GPU_FUNCTION)
def run_semeval(csv_name: str) -> dict:
    """SemEval-2014 restaurant sample: service filter + service sentiment map."""
    import csv
    import sys

    data_csv = DATA_DIR / csv_name
    if not data_csv.exists():
        raise FileNotFoundError(
            f"{csv_name} not found in experiments/data. "
            "Run experiments/scripts/prepare_semeval_sample.py first."
        )
    with open(data_csv, newline="", encoding="utf-8") as f:
        input_rows = sum(1 for _ in csv.DictReader(f))

    out_file = "/tmp/client_output.json"
    run = _serve_and_run(
        [sys.executable, str(SCRIPTS_DIR / "client_semeval_service.py"),
         "--data-path", str(data_csv), "--model-name", MODEL_ID,
         "--port", str(SERVER_PORT), "--out", out_file],
        client_cwd=str(SCRIPTS_DIR),
        client_output_file=out_file,
    )
    result = _run_info("semeval_service_filter_map", f"SemEval-2014 Task 4 restaurants: {csv_name}", input_rows)
    client_output = run["client_output"] or {}
    latency = client_output.get("server_latency_sec")
    result["server_latency_sec"] = latency
    result["throughput_rows_per_sec"] = input_rows / latency if latency else None
    result["scores"] = client_output.get("scores")
    return _split_outputs(run, result)


# ---- Entrypoints (these run on your computer) ----

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" if modal.is_local() else None


def _save_run(name: str, out: dict):
    """Write a run's results and logs to experiments/results/runs/<name>_<timestamp>/."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / "runs" / f"{name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(out["result"], indent=2), encoding="utf-8")
    (run_dir / "client_stdout.txt").write_text(out["client_stdout"], encoding="utf-8")
    (run_dir / "client_stderr.txt").write_text(out["client_stderr"], encoding="utf-8")
    (run_dir / "server.log").write_text(out["server_log"], encoding="utf-8")
    if out.get("client_output") is not None:
        (run_dir / "client_output.json").write_text(
            json.dumps(out["client_output"], indent=2), encoding="utf-8"
        )
    print(json.dumps(out["result"], indent=2))
    print(f"\nSaved to {run_dir}")


@app.local_entrypoint()
def check():
    verify.remote()


@app.local_entrypoint()
def download():
    download_model.remote()


@app.local_entrypoint()
def medec():
    _save_run("medec", run_medec.remote())


@app.local_entrypoint()
def semeval(csv: str = "semeval2014_restaurants_train_sample200_seed42.csv"):
    _save_run("semeval", run_semeval.remote(csv))
