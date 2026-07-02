"""Small fixed-seed random tuner for ORACLE heuristic parameters.

This is intentionally simple and deterministic. It tests candidate parameter
sets against the built-in starter and keeps only changes that improve the
fixed-seed score.
"""
import argparse
import random
import time

from kaggle_environments import make

import main


SEARCH_SPACE = {
    "defense_horizon": [6, 8, 10, 12],
    "defense_margin": [3, 5, 7],
    "defense_wait_slack": [1, 2, 3],
    "expand_avail_min": [6, 8, 10, 12],
}


def reset():
    if hasattr(main, "reset_state"):
        main.reset_state()


def score_params(params, seeds):
    score = 0
    wins = losses = ties = 0
    for seed in seeds:
        reset()
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([lambda o, c=None: main.agent(o, params), "starter"])
        reward = env.steps[-1][0].reward
        if reward == 1:
            wins += 1
            score += 1
        elif reward == -1:
            losses += 1
            score -= 1
        else:
            ties += 1
    return score, wins, losses, ties


def mutate(params, rng):
    candidate = dict(params)
    key = rng.choice(list(SEARCH_SPACE))
    candidate[key] = rng.choice(SEARCH_SPACE[key])
    return candidate


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.random_seed)
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    best = {
        "defense_horizon": 10,
        "defense_margin": 5,
        "defense_wait_slack": 2,
        "expand_avail_min": 8,
    }
    best_score, w, l, t = score_params(best, seeds)
    print(f"baseline {best}: score={best_score}, {w}W/{l}L/{t}T")

    started = time.perf_counter()
    for i in range(args.iterations):
        candidate = mutate(best, rng)
        score, w, l, t = score_params(candidate, seeds)
        print(f"iter {i:02d} {candidate}: score={score}, {w}W/{l}L/{t}T")
        if score > best_score:
            best = candidate
            best_score = score
            print("  accepted")

    elapsed = time.perf_counter() - started
    print(f"best {best}: score={best_score}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main_cli()
