# ORACLE Orbit Wars Agent

ORACLE is a compact Python agent for Kaggle's Orbit Wars competition. The current submission path favors a fast deterministic heuristic policy over expensive search, because local testing showed the search-heavy version was not safe under turn-budget constraints.

## Current Status

The active agent is in `oracle-orbit-wars/main.py`.

Key behavior:

- ETA-aware defense with production-before-impact accounting.
- Counter-punch attacks against recently weakened enemy launch sources.
- Race-denial expansion for contested neutral planets.
- Safer grouped attacks with greedy fallback when synchronized launches are not useful.
- Endgame sweep logic to convert banked ships into captures before the episode ends.
- Per-turn intercept caching to reduce repeated orbital geometry work.
- Optional MCTS code is retained for experiments, but `use_mcts` defaults to `False`.

## Fixed-Seed Results

Measured with `oracle-orbit-wars/eval_oracle.py`.

Seeds `0..39`:

- Candidate vs starter: `27W / 13L = 67.5%`
- Candidate vs random: `10W / 0L = 100.0%`
- Baseline vs starter: `23W / 17L = 57.5%`
- Candidate vs baseline: `29W / 11L = 72.5%`

Seeds `40..69`:

- Candidate vs starter: `25W / 5L = 83.3%`
- Candidate vs random: `10W / 0L = 100.0%`
- Baseline vs starter: `21W / 9L = 70.0%`
- Candidate vs baseline: `26W / 4L = 86.7%`

These are local fixed-seed measurements, not a guarantee of live leaderboard ELO.

## Repository Layout

```text
oracle-orbit-wars/
  main.py          # Kaggle submission agent
  eval_oracle.py   # Fixed-seed evaluator
  tune_oracle.py   # Simple deterministic parameter tuner
  test_agent.py    # Quick smoke test
  replays/         # Optional opening-book replay inputs, kept empty by default
  requirements.txt
  README.md
  LICENSE
```

The root `LICENSE` is provided so GitHub detects the MIT license. The package-level license is kept for standalone submission folders.

## Setup

```bash
cd oracle-orbit-wars
python -m venv venv
venv/bin/pip install -r requirements.txt
```

## Validate

```bash
venv/bin/python -m py_compile main.py eval_oracle.py tune_oracle.py test_agent.py
venv/bin/python eval_oracle.py --agent main.py --games 40
```

If you keep an older baseline copy available, compare against it:

```bash
venv/bin/python eval_oracle.py --agent main.py --baseline path/to/baseline_main.py --games 40
```

## Submit

From the repo root:

```bash
zip -j ORACLE_1700_SUBMISSION.zip oracle-orbit-wars/main.py
kaggle competitions submit orbit-wars -f ORACLE_1700_SUBMISSION.zip -m "ORACLE optimized heuristic"
```
