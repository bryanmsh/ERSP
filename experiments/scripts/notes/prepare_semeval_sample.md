# Notes: `prepare_semeval_sample.py`

Builds a small random sample of SemEval-2014 Task 4 restaurant sentences as a CSV that Kalypso can read. We use it for the first restaurant-review runs before moving to the full dataset.

## How to run it

From the root of the ERSP repo. It only uses Python's standard library, so it runs on any Python 3.10 or newer without installing anything.

```powershell
python experiments/scripts/prepare_semeval_sample.py               # 200 sentences, seed 42
python experiments/scripts/prepare_semeval_sample.py --n 500 --seed 7
```

The output goes to `experiments/data/semeval2014_restaurants_train_sample<N>_seed<S>.csv`. The `experiments/data/` folder is gitignored, so each person runs this script once to make the file. With the same `--n` and `--seed`, everyone gets the same rows.

## About the dataset

SemEval-2014 Task 4 is an aspect-based sentiment analysis benchmark. The restaurant part has 3,041 training sentences and 800 test sentences, each labeled by people with:

- **aspect categories**: food, service, price, ambience, and anecdotes/miscellaneous, each with a polarity (positive, negative, neutral, or conflict, which means both positive and negative)
- **aspect terms**: the actual words in the sentence that the opinion is about, like "staff" or "pasta", also with a polarity

The official page is https://alt.qcri.org/semeval2014/task4/. We download the official training file, `Restaurants_Train_v2.xml`, from a Hugging Face mirror (`alexcadillon/SemEval2014Task4`), because the original site needs a META-SHARE account. The script checks that the download has 3,041 sentences, matching the official count.

We only sample from the training split. The test split stays untouched for later experiments.

## What the script does

`download_train_xml()` downloads the XML file to `experiments/data/raw/` if it isn't there yet. The URL is pinned to a specific mirror revision, and the script checks the file's SHA-256 checksum against a fixed value. If the mirror ever changes the file, the script stops instead of silently using different data.

`parse_sentences()` reads each `<sentence>` from the XML and makes one row per sentence:

| Column | What's in it |
|---|---|
| `data` | The sentence text. This is the only column Kalypso sends to the model. |
| `sentence_id` | SemEval's ID for the sentence |
| `has_service` | 1 if the sentence has a "service" aspect category, otherwise 0 |
| `service_polarity` | The polarity of the service category, or empty if there isn't one |
| `aspect_categories` | All categories and polarities, as JSON |
| `aspect_terms` | All aspect terms and polarities, as JSON |

Kalypso's CSV reader (`kalypso/vllm/kalypso/data/data.py`) uses the `data` column as the model input and ignores the rest. So the labels can sit in the same file without the model ever seeing them, and we can score the results afterward.

`main()` takes a simple random sample with a fixed seed. Random sampling keeps the share of service sentences close to the full training split. In our 200-row sample with seed 42, 47 sentences (23.5%) are about service, compared with 597 of 3,041 (19.6%) in the full split.

## Limits

A 200-sentence sample is for getting the pipeline working and getting first numbers. Accuracy and timing on 200 sentences won't necessarily match the full dataset, and the short/long length splits we plan to use later aren't part of this sample.
