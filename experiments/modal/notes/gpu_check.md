# Notes: `gpu_check.py`

A small test that starts one A100 80GB on Modal and prints what `nvidia-smi` sees. Run it to make sure your Modal account works and has GPU access before building anything bigger. It costs a few cents.

## How to run it

From the root of the ERSP repo:

```powershell
python -m modal run experiments/modal/gpu_check.py
```

We use `python -m modal` instead of `modal` because on Windows, pip often puts the `modal` command in a folder that isn't on your PATH, and PowerShell says "modal is not recognized." `python -m modal` runs the same tool through Python and avoids that.

Run `python -m modal setup` once before this. It links the Modal CLI to your account and saves a token in your home folder (`~/.modal.toml`). Treat that token like a password and never commit it.

## What the file does

```python
app = modal.App("kalypso-gpu-check")
```
Creates a Modal app, which is basically a project name. It shows up under this name in the Modal dashboard.

```python
image = modal.Image.debian_slim(python_version="3.12")
```
Describes the container the code runs in: a small Debian Linux with Python 3.12. Nothing else gets installed, because Modal provides `nvidia-smi` on GPU machines and that's all this test needs.

```python
@app.function(gpu="A100-80GB", image=image, timeout=300)
def check():
    ...
```
The `@app.function(...)` decorator turns `check` into a Modal function. Calling it with `.remote()` runs it on Modal's servers with an A100 80GB attached. `timeout=300` stops it after 5 minutes so a stuck run can't keep billing.

Inside, `subprocess.run(["nvidia-smi"], ...)` runs NVIDIA's status tool and prints the result.

```python
@app.local_entrypoint()
def main():
    check.remote()
```
`main` runs on your own computer when you type `modal run`, and it tells Modal to run `check` remotely.

## What the output should look like

You should get a table showing the GPU name (`NVIDIA A100-SXM4-80GB`) and about `81920MiB` of memory with almost none in use. The driver and CUDA versions are at the top. That CUDA number is the highest version the driver supports, and our PyTorch build uses CUDA 12.9, so anything 12.9 or higher works.

When we ran it on 2026-11-23, we got an A100-SXM4-80GB with driver 610.57.04, which supports CUDA up to 13.3. Modal can put you on different machines, so your driver version may not match.
