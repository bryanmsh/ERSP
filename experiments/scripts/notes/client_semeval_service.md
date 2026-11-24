# Notes: `client_semeval_service.py`

The benchmark client for the SemEval restaurant sample. It sends Kalypso one semantic query with two operators, times it, and scores the answers against the SemEval labels.

It follows the same request format as Kalypso's own clients in `kalypso/vllm/kalypso/benchmark/`. We wrote our own instead of reusing the MEDEC client because the MEDEC client has medical prompts built in, and it only prints a summary, not the per-row answers we need for scoring.

## How it gets run

Normally `experiments/modal/kalypso_app.py::semeval` starts it inside the Modal container, after Kalypso's server is up. You can also run it yourself against any running Kalypso server:

```bash
python client_semeval_service.py --data-path sample.csv --out response.json --port 8003
```

## The query

```python
"ops": [
    {"op": "sem_filter", "args": {"prompt": SERVICE_FILTER}},
    {"op": "sem_map", "args": {"prompt": SERVICE_SENTIMENT_MAP}},
]
```

1. **Filter**: does this sentence talk about the restaurant's service? Kalypso wraps our condition in its own instructions that tell the model to answer TRUE or FALSE (see `kalypso/vllm/kalypso/sem_ops/prompt_utils.py`). Sentences that get FALSE are dropped.
2. **Map**: for the sentences that pass, what's the sentiment toward the service? We ask for one word from SemEval's own label set (positive, negative, neutral, conflict) so the answer can be compared directly to the labels.

`"blocking": False` keeps Kalypso in its normal mode. Kalypso's code has a `blocking` option, but we're measuring the baseline and not using it.

## Scoring

`score()` compares the response to the label columns in the CSV.

Kalypso only returns the rows that passed the filter, each with its row index (`idx`). So a row counts as a predicted "yes" exactly when its index shows up in the results. Comparing that to `has_service` gives:

- **Confusion counts**: true positives, false positives, false negatives, true negatives
- **Filter accuracy, precision, and recall**
- **Observed selectivity**: the share of rows that passed the filter
- **Ground-truth selectivity**: the share of rows that really are about service. The gap between the two shows how far off the filter is.

For the map, `first_polarity_word()` pulls the first polarity word out of the model's answer, so "Negative." or "negative - the staff was rude" both count as `negative`. Answers with no polarity word get `unparsed`. Map accuracy only counts sentences that are really about service *and* passed the filter, because the others have no service label to compare against.

## Output

The client writes everything to the `--out` file: request time, the server's reported latency, the scores, and the full server response with every row's answers. `kalypso_app.py` saves that as `client_output.json` in the run's results folder.

## Limits

These accuracy numbers are for Llama 3.1 8B with these exact prompts. A different model or a reworded prompt can change them, so they shouldn't be read as how well Kalypso itself does at the task.
