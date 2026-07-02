"""Fixed-seed evaluator for ORACLE Orbit Wars agents.

Examples:
  python eval_oracle.py --games 40
  python eval_oracle.py --baseline oracle-orbit-wars/main.py --games 40
"""
import argparse
import importlib.util
import logging
import time

from kaggle_environments import make


logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(logging.ERROR)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset(module):
    if hasattr(module, "reset_state"):
        module.reset_state()


def run_vs_builtin(module, opponent, seeds):
    wins = losses = ties = 0
    started = time.perf_counter()
    for seed in seeds:
        reset(module)
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([lambda o, c=None: module.agent(o, c), opponent])
        reward = env.steps[-1][0].reward
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            ties += 1
    elapsed = time.perf_counter() - started
    return wins, losses, ties, elapsed


def run_head_to_head(agent_a, agent_b, seeds):
    wins = losses = ties = 0
    started = time.perf_counter()
    for seed in seeds:
        reset(agent_a)
        reset(agent_b)
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([
            lambda o, c=None: agent_a.agent(o, c),
            lambda o, c=None: agent_b.agent(o, c),
        ])
        reward = env.steps[-1][0].reward
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            ties += 1
    elapsed = time.perf_counter() - started
    return wins, losses, ties, elapsed


def print_result(label, result):
    wins, losses, ties, elapsed = result
    games = wins + losses + ties
    rate = 100.0 * wins / max(1, games)
    print(f"{label}: {wins}W/{losses}L/{ties}T = {rate:.1f}% in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="main.py")
    parser.add_argument("--baseline")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.games))
    agent = load_module("candidate_agent", args.agent)

    print(f"Evaluating {args.agent} on seeds {seeds[0]}..{seeds[-1]}")
    print_result("candidate vs starter", run_vs_builtin(agent, "starter", seeds))
    print_result("candidate vs random", run_vs_builtin(agent, "random", seeds[: min(10, len(seeds))]))

    if args.baseline:
        baseline = load_module("baseline_agent", args.baseline)
        print(f"\nBaseline: {args.baseline}")
        print_result("baseline vs starter", run_vs_builtin(baseline, "starter", seeds))
        print_result("candidate vs baseline", run_head_to_head(agent, baseline, seeds))


if __name__ == "__main__":
    main()
