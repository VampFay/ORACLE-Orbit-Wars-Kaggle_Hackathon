# ORACLE 1700+ Upgrade Implementation Plan

Goal: push ORACLE toward leaderboard strength while keeping the submitted agent fast enough for Kaggle turn limits. A 1700+ live ELO result cannot be guaranteed locally, so each change must beat the previous bot on fixed-seed evaluation before it is considered submission-ready.

## Implemented Components

### 1. Fast Default Policy
- MCTS is no longer enabled by default.
- Search remains available through `config["use_mcts"]`, but the submitted path uses the faster heuristic.
- Reason: the previous MCTS upgrade was too slow and could hang local evaluation.

### 2. ETA-Aware Defense
- Added `incoming_threats()` to return incoming fleet ETA and ship count.
- Defense now estimates production before impact.
- Reinforcements wait only when there is safe slack.
- If the defense is late, the earliest viable donor is launched instead of silently doing nothing.

### 3. Safer Coordinated Attacks
- Enemy attack candidates are grouped by target.
- Multi-source launches only claim a target when the ships actually launched are enough.
- Added a greedy fallback so the bot still takes easy captures when synchronized launches are not available.

### 4. Endgame Sweep
- After turn 440, surplus ships are converted into capture attempts if they can arrive before game end.
- This reduces late-game ship hoarding.

### 5. Evaluation and Tuning
- Added `eval_oracle.py` for fixed-seed evaluation against built-ins and a baseline file.
- Replaced the tuning scaffold with deterministic random search over exposed heuristic parameters.
- Added `reset_state()` so repeated local games do not leak cross-game memory.

### 6. Runtime Safety
- MCTS is gated behind an explicit `config["use_mcts"] is True` check.
- MCTS has a configurable `mcts_time_budget` and remains opt-in for local experiments.
- Added a per-turn intercept cache so repeated source-target checks reuse orbital projection and sun-path results.

## Current Fixed-Seed Result

Command:

```bash
venv/bin/python eval_oracle.py --agent main.py --baseline path/to/previous_baseline_main.py --games 40
```

Result on seeds 0..39:

- Candidate vs starter: 27W / 13L = 67.5%
- Candidate vs random: 10W / 0L = 100%
- Baseline vs starter: 23W / 17L = 57.5%
- Candidate vs baseline: 29W / 11L = 72.5%

## Next Optimization Targets

1. Add stronger opponent pool from submitted historical versions.
2. Tune heuristic parameters on 100+ seeds.
3. Add replay diagnostics for losses.
4. Test any MCTS/search variant only behind hard per-turn timing guards.
