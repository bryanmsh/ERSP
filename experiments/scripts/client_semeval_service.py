"""SemEval-2014 restaurant benchmark client for Kalypso: service filter + sentiment map.

Same request format as Kalypso's own benchmark clients
(kalypso/vllm/kalypso/benchmark/client_medec_filter_map_map.py): it sends one
semantic query plan to the server's /v1/semantic/query endpoint and times it.
It also scores the answers against the SemEval labels in the CSV.

Usually started by experiments/modal/kalypso_app.py inside the Modal container.
To run it yourself against a running Kalypso server:
    python client_semeval_service.py --data-path sample.csv --out response.json
"""
import argparse
import csv
import json
import time
from collections import Counter

import requests

# Kalypso wraps a filter condition in its own TRUE/FALSE instructions and a map
# instruction in its own "apply this mapping" instructions (see
# kalypso/vllm/kalypso/sem_ops/prompt_utils.py), so these only describe the task.
SERVICE_FILTER = (
    "The context is one sentence from a restaurant review. Answer TRUE if the "
    "sentence talks about the restaurant's service, such as the staff, waiters, "
    "hosts, wait times, or how customers were treated. Answer FALSE if it does not."
)
SERVICE_SENTIMENT_MAP = (
    "What sentiment does this sentence express about the restaurant's service? "
    "Answer with exactly one word: positive, negative, neutral, or conflict "
    "(conflict means both positive and negative)."
)
POLARITIES = ("positive", "negative", "neutral", "conflict")


def build_payload(data_path: str, model_name: str) -> dict:
    return {
        "data_path": data_path,
        "ops": [
            {"op": "sem_filter", "args": {"prompt": SERVICE_FILTER}},
            {"op": "sem_map", "args": {"prompt": SERVICE_SENTIMENT_MAP}},
        ],
        "blocking": False,  # Kalypso's normal mode; we're measuring the baseline
        "model_name": model_name,
    }


def first_polarity_word(text: str) -> str:
    """Pull the first polarity label out of the model's map answer."""
    cleaned = text.strip().lower()
    for word in cleaned.replace(".", " ").replace(",", " ").split():
        if word in POLARITIES:
            return word
    return "unparsed"


def score(rows: list[dict], response: dict) -> dict:
    """Compare Kalypso's answers to the SemEval labels.

    Rows that fail the filter don't come back, so a row is a predicted "yes"
    exactly when its index appears in the results.
    """
    passed = {}
    for r in response.get("results", []):
        map_text = None
        for out in r.get("outputs", []):
            if "Map" in str(out.get("op")):
                map_text = str(out.get("value"))
        passed[r["idx"]] = map_text

    tp = fp = fn = tn = 0
    polarity_pairs = []
    for idx, row in enumerate(rows):
        truth = row["has_service"] == "1"
        predicted = idx in passed
        if truth and predicted:
            tp += 1
            if passed[idx] is not None:
                polarity_pairs.append((row["service_polarity"], first_polarity_word(passed[idx])))
        elif predicted:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1

    n = len(rows)
    correct_polarity = sum(1 for gold, pred in polarity_pairs if gold == pred)
    return {
        "input_rows": n,
        "rows_passed_filter": len(passed),
        "selectivity_observed": len(passed) / n if n else 0.0,
        "selectivity_ground_truth": (tp + fn) / n if n else 0.0,
        "filter_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "filter_accuracy": (tp + tn) / n if n else 0.0,
        "filter_precision": tp / (tp + fp) if tp + fp else None,
        "filter_recall": tp / (tp + fn) if tp + fn else None,
        # Map accuracy only counts true service sentences that passed the filter,
        # since the others have no service label to compare against.
        "map_rows_scored": len(polarity_pairs),
        "map_polarity_accuracy": correct_polarity / len(polarity_pairs) if polarity_pairs else None,
        "map_predicted_labels": dict(Counter(pred for _, pred in polarity_pairs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="CSV built by prepare_semeval_sample.py")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--out", required=True, help="where to write the full response and scores")
    args = parser.parse_args()

    with open(args.data_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    payload = build_payload(args.data_path, args.model_name)
    print(json.dumps(payload, indent=2))

    start = time.perf_counter()
    resp = requests.post(f"http://localhost:{args.port}/v1/semantic/query", json=payload)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    response = resp.json()

    scores = score(rows, response)
    out = {
        "client_request_seconds": elapsed,
        "server_latency_sec": response.get("latency_sec"),
        "num_output_rows": response.get("num_output_rows"),
        "scores": scores,
        "response": response,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nRequest time: {elapsed:.3f}s (server says {response.get('latency_sec')}s)")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
