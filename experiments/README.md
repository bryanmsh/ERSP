# Kalypso baseline experiments

This folder has everything needed to run Kalypso on a GPU and reproduce our first baseline numbers. We're measuring Kalypso as it is, with no changes. Shared-prompt blocking and our other ideas come later and get compared against these numbers.

## Layout

```text
experiments/
├── README.md                  this file
├── modal/
│   ├── gpu_check.py           smoke test: start an A100 on Modal, print nvidia-smi
│   ├── kalypso_app.py         container image, install check, model download, benchmark runs
│   └── notes/                 walkthrough of each file above
├── scripts/
│   ├── prepare_semeval_sample.py   builds the SemEval-2014 sample CSV
│   ├── client_semeval_service.py   sends the query to Kalypso and scores the answers
│   └── notes/                      walkthrough of each script
├── data/                      gitignored; rebuilt by prepare_semeval_sample.py
└── results/
    ├── baseline.csv           one row per run, plus a median row
    ├── environment_modal_2026-11-23.json   software versions from the install check
    └── runs/<workload>_<timestamp>/        raw output of each run
```

Each Python file has a matching notes file explaining what it does line by line. Start there if something here is unclear.

## Setup (once per person)

You need Python 3.10 or newer on your own machine. Kalypso itself runs in a Modal container with Python 3.12, so your local version only runs the Modal command-line tool.

1. **Modal.** Make an account at modal.com, then:
   ```powershell
   pip install modal
   python -m modal setup
   ```
   We use `python -m modal` everywhere because on Windows the plain `modal` command often isn't on your PATH.
2. **Hugging Face.** Request access to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (approval is usually quick). Then create a token at huggingface.co/settings/tokens. A fine-grained token with read access to just that model is enough. Store it in Modal:
   ```powershell
   python -m modal secret create huggingface HF_TOKEN=hf_yourtoken
   ```
   Never put the token in the code or the repo.

## Running it

From the root of the repo:

```powershell
python experiments/scripts/prepare_semeval_sample.py               # build the 200-sentence sample (local, free)
python -m modal run experiments/modal/kalypso_app.py::check        # verify the install on an A100
python -m modal run experiments/modal/kalypso_app.py::download     # download model weights to Modal storage (once)
python -m modal run experiments/modal/kalypso_app.py::semeval      # run the benchmark
```

The first Modal command builds the container image, which takes 10 to 20 minutes. After that it's cached. A benchmark run takes about 1.5 minutes to start the server plus a few seconds for the query, and costs roughly $0.15 of A100 time. Results save to `experiments/results/runs/semeval_<timestamp>/`.

To run on a different GPU machine instead of Modal, the install steps are the `run_commands` in `kalypso_app.py` (they follow Kalypso's README), and the server flags are `SERVER_ARGS` in the same file.

## Setup used for the baseline

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (Modal) |
| Model | meta-llama/Llama-3.1-8B-Instruct (bf16) |
| Kalypso | upstream commit `a77c625` (vLLM 0.13.0rc4) |
| Software | Python 3.12, PyTorch 2.9.0+cu129, FlashInfer 0.5.3, Triton 3.5.0 |
| Server flags | `--gpu-memory-utilization 0.9 --enable-prefix-caching --max-model-len 32768`, plus stats flags (see `kalypso_app.py`) |
| Data | 200 random sentences from the SemEval-2014 Task 4 restaurant **train** split, seed 42 (47 about service) |
| Query | Filter: is this sentence about service? Then map: sentiment toward service (positive / negative / neutral / conflict) |
| Mode | Kalypso default, `blocking: false` |

Kalypso's own benchmarks use Llama 3.3 70B across 4 GPUs. We used the 8B model to fit on one A100, so these numbers won't necessarily carry over to 70B.

## Results (3 runs, 2026-11-24)

| Metric | Run 1 | Run 2 | Run 3 | Median |
|---|---|---|---|---|
| Query time (s) | 5.94 | 6.68 | 7.88 | **6.68** |
| Throughput (sentences/s) | 33.6 | 29.9 | 25.4 | **29.9** |
| Server startup (s) | 75.1 | 75.1 | 105.2 | 75.1 |

Everything else was identical in all three runs, including every model answer:

| Metric | Value |
|---|---|
| LLM requests | 252 (200 filter, 52 map) |
| Prompt tokens | 51,367 |
| Prompt tokens reused from the prefix cache | 17,824 (34.7%) |
| Generated tokens | 3,578 (filter 1,600, map 1,978) |
| KV-cache evictions / preemptions | 0 / 0 |
| Peak GPU memory | about 74,190 MiB (mostly vLLM's up-front 90% reservation) |
| Rows passing the filter | 51 of 200 (25.5%); ground truth 47 (23.5%) |
| Filter accuracy / precision / recall | 91% / 78% / 85% |
| Map sentiment accuracy | 80% (32 of 40 correctly filtered service sentences) |

Per-run detail is in `results/baseline.csv` and the `results/runs/` folders.

## What these numbers do and don't show

- **Timing varied a lot between runs** (5.9 to 7.9 seconds) even though the work was identical. Modal can put each run on a different machine, and run 3 also had a slower startup, so this spread is probably machine-to-machine variation. A comparison between two Kalypso configurations should run both on the same container to cancel that out.
- **The outputs were deterministic.** The same answers came back all three times, so the accuracy numbers are stable for this model and prompt, but only for this model and prompt.
- **The model writes more than we ask.** The map was asked for one word but averaged about 38 generated tokens, often adding an "Explanation of reasoning" section. One answer hit its token limit and was retried, which is why there are 52 map calls for 51 rows. The filter always used Kalypso's full 8-token cap. Since generating tokens is the slow part, this inflates the runtime. It's worth tightening the prompts or capping answer length before comparing against blocking, applied the same way to both setups.
- **Per-operator time isn't available.** Kalypso runs operators as an overlapping pipeline and only counts tokens per operator.
- **200 sentences is small.** No KV-cache pressure came up, so this run doesn't exercise Kalypso's memory management. Larger inputs are needed for that.

## Planned next

The full SemEval-2014 restaurant dataset (3,041 train / 800 test sentences), keeping the original labels and split:

- short inputs (bottom 25% by token length) and long inputs (top 25%), with thresholds computed from the training split using the model's tokenizer
- the full dataset as a general benchmark
- filters built from the existing labels, for example "discusses service" and "negative about service"
- selectivity for each filter

## Not committed

Model weights, datasets, secrets, and `client_output.json` files (which contain SemEval sentences) stay out of Git. See `.gitignore`. Anyone can rebuild the data with `prepare_semeval_sample.py`.
