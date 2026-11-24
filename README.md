# ERSP: Kalypso research

UMass Amherst ERSP project working with [Kalypso](https://github.com/goodluck-hojae/kalypso), a relational LLM serving system built on vLLM. We're starting by reproducing Kalypso's baseline performance, and will later study shared-prompt tuple blocking for semantic operators.

## Layout

- `kalypso/`: Kalypso's source code, imported with `git subtree` from upstream commit `a77c625f88ef878825e9c3f43df13ca809433ecf`. We haven't modified it.
- `experiments/`: setup scripts, benchmark clients, and results. Start with `experiments/README.md`.

## Updating Kalypso from upstream

```bash
git subtree pull --prefix=kalypso https://github.com/goodluck-hojae/kalypso.git main --squash
```

Record the new commit in `experiments/modal/kalypso_app.py` (`KALYPSO_COMMIT`) and rerun the baseline after updating.
