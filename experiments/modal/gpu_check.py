"""Smoke test: start one A100 80GB container on Modal and print GPU info.

Run from the ERSP folder:  modal run experiments/modal/gpu_check.py
"""
import modal

app = modal.App("kalypso-gpu-check")

# A minimal Linux container with Python 3.12. 
image = modal.Image.debian_slim(python_version="3.12")


@app.function(gpu="A100-80GB", image=image, timeout=300)
def check():
    import subprocess

    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(out.stdout or out.stderr)


@app.local_entrypoint()
def main():
    check.remote()