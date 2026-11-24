"""Build a small random sample of SemEval-2014 Task 4 restaurant sentences.

Run from the root of the ERSP repo (standard library only, any Python 3.10+):
    python experiments/scripts/prepare_semeval_sample.py
    python experiments/scripts/prepare_semeval_sample.py --n 500 --seed 7

Downloads the official restaurant training file (Restaurants_Train_v2.xml) from a
pinned Hugging Face mirror, checks its SHA-256, and writes a CSV that Kalypso can
read. Only the training split is sampled; the test split is left alone.

Output: experiments/data/semeval2014_restaurants_train_sample<N>_seed<S>.csv
The data/ folder is gitignored. Anyone can rebuild the same file with this script.
"""
import argparse
import csv
import hashlib
import json
import random
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Pinned mirror revision and checksum, so everyone gets byte-identical data.
MIRROR_REVISION = "3fd43769112ef6517c263e303fbb7dc3002c35df"
TRAIN_URL = (
    "https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/resolve/"
    f"{MIRROR_REVISION}/SemEval'14-ABSA-TrainData_v2%20%26%20AnnotationGuidelines/"
    "Restaurants_Train_v2.xml"
)
TRAIN_SHA256 = "223601da1bded6caa4ef9cf91a7007578141ca6d8ed50d5a5c217565f89d2fc5"
EXPECTED_SENTENCES = 3041  # official count for the restaurant training split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "raw" / "Restaurants_Train_v2.xml"


def download_train_xml() -> Path:
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {TRAIN_URL}")
        urllib.request.urlretrieve(TRAIN_URL, RAW_PATH)
    digest = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    if digest != TRAIN_SHA256:
        raise SystemExit(
            f"Checksum mismatch for {RAW_PATH}\n  got      {digest}\n  expected {TRAIN_SHA256}\n"
            "Delete the file and run again."
        )
    return RAW_PATH


def parse_sentences(xml_path: Path) -> list[dict]:
    rows = []
    for s in ET.parse(xml_path).getroot().findall("sentence"):
        categories = [
            {"category": c.get("category"), "polarity": c.get("polarity")}
            for c in s.iter("aspectCategory")
        ]
        terms = [
            {"term": t.get("term"), "polarity": t.get("polarity")}
            for t in s.iter("aspectTerm")
        ]
        service = [c["polarity"] for c in categories if c["category"] == "service"]
        rows.append({
            # Kalypso sends the "data" column to the model. Everything else is a
            # label column it ignores, kept so we can score the results.
            "data": s.findtext("text"),
            "sentence_id": s.get("id"),
            "has_service": int(bool(service)),
            "service_polarity": service[0] if service else "",
            "aspect_categories": json.dumps(categories),
            "aspect_terms": json.dumps(terms),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of sentences to sample")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the sample")
    args = parser.parse_args()

    rows = parse_sentences(download_train_xml())
    if len(rows) != EXPECTED_SENTENCES:
        raise SystemExit(f"Expected {EXPECTED_SENTENCES} sentences, found {len(rows)}")

    # Simple random sample, so the share of service sentences stays close to
    # the full training split (about 20%).
    sample = random.Random(args.seed).sample(rows, args.n)

    out_path = DATA_DIR / f"semeval2014_restaurants_train_sample{args.n}_seed{args.seed}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        writer.writeheader()
        writer.writerows(sample)

    n_service = sum(r["has_service"] for r in sample)
    full_service = sum(r["has_service"] for r in rows)
    print(f"Wrote {len(sample)} rows to {out_path}")
    print(f"Service sentences: {n_service} in sample ({n_service / len(sample):.1%}), "
          f"{full_service} in full train split ({full_service / len(rows):.1%})")


if __name__ == "__main__":
    main()
