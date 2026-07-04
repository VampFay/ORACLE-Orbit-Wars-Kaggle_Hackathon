"""Local Orbit Wars arena with simple Elo-style ratings.

This is not Kaggle's official leaderboard. It is a regression harness for
comparing ORACLE variants across fixed seeds, including 4-player FFA games.

Examples:
  python arena_oracle.py --agents main.py starter --games 40 --players 2
  python arena_oracle.py --agents main.py starter starter starter --games 40 --players 4
"""
import argparse
import importlib.util
import itertools
import logging
import random
import statistics
import time

from kaggle_environments import make


logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(logging.ERROR)


BUILTINS = {"starter", "random"}


class AgentSpec:
    def __init__(self, label, source, module=None):
        self.label = label
        self.source = source
        self.module = module

    def runner(self):
        if self.source in BUILTINS:
            return self.source
        return lambda obs, config=None: self.module.agent(obs, config)

    def reset(self):
        if self.module is not None and hasattr(self.module, "reset_state"):
            self.module.reset_state()


def load_agent(spec, index):
    if ":" in spec:
        source, label = spec.split(":", 1)
    else:
        source = spec
        label = spec
    if source in BUILTINS:
        return AgentSpec(label, source)
    module_name = f"arena_agent_{index}"
    module_spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return AgentSpec(label, source, module)


def elo_expected(a, b):
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def pair_score(reward_a, reward_b):
    if reward_a > reward_b:
        return 1.0
    if reward_a < reward_b:
        return 0.0
    return 0.5


def update_elos(elos, labels, rewards, k_factor):
    deltas = {label: 0.0 for label in labels}
    for i, j in itertools.combinations(range(len(labels)), 2):
        a = labels[i]
        b = labels[j]
        actual_a = pair_score(rewards[i], rewards[j])
        expected_a = elo_expected(elos[a], elos[b])
        delta = k_factor * (actual_a - expected_a)
        deltas[a] += delta
        deltas[b] -= delta
    for label, delta in deltas.items():
        elos[label] += delta


def run_game(agents, seed):
    for agent in agents:
        agent.reset()
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent.runner() for agent in agents])
    rewards = [state.reward for state in env.steps[-1]]
    statuses = [state.status for state in env.steps[-1]]
    return rewards, statuses


def summarize(labels, reward_rows, elapsed, elos):
    games = len(reward_rows)
    print(f"\nGames: {games}  elapsed: {elapsed:.1f}s")
    print("\nLocal ratings")
    for label in sorted(labels, key=lambda name: elos[name], reverse=True):
        print(f"  {label:18s} {elos[label]:7.1f}")

    print("\nPer-agent results")
    for i, label in enumerate(labels):
        rewards = [row[i] for row in reward_rows]
        avg = statistics.mean(rewards) if rewards else 0.0
        wins = sum(1 for row in reward_rows if row[i] == max(row))
        nonwins = games - wins
        print(f"  {label:18s} avg_reward={avg:6.3f} wins={wins:4d} nonwins={nonwins:4d}")

    if len(labels) == 2:
        a_wins = sum(1 for row in reward_rows if row[0] > row[1])
        b_wins = sum(1 for row in reward_rows if row[1] > row[0])
        ties = games - a_wins - b_wins
        rate = 100.0 * a_wins / max(1, games)
        print(f"\nHead-to-head: {labels[0]} {a_wins}W/{b_wins}L/{ties}T = {rate:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=["main.py", "starter"])
    parser.add_argument("--players", type=int, choices=(2, 4), default=2)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle seats each game.")
    parser.add_argument("--k-factor", type=float, default=24.0)
    args = parser.parse_args()

    if len(args.agents) != args.players:
        raise SystemExit(f"--players={args.players} requires exactly {args.players} agents")

    base_agents = [load_agent(spec, i) for i, spec in enumerate(args.agents)]
    seen_labels = {}
    for agent in base_agents:
        count = seen_labels.get(agent.label, 0) + 1
        seen_labels[agent.label] = count
        if count > 1:
            agent.label = f"{agent.label}#{count}"
    labels = [agent.label for agent in base_agents]
    elos = {label: 1500.0 for label in labels}
    reward_rows = []

    started = time.perf_counter()
    rng = random.Random(args.seed_start)
    for game_index in range(args.games):
        seed = args.seed_start + game_index
        agents = list(base_agents)
        seat_order = list(range(len(agents)))
        if args.shuffle:
            rng.shuffle(seat_order)
            agents = [agents[i] for i in seat_order]

        rewards, statuses = run_game(agents, seed)
        if any(status not in ("DONE", "ACTIVE") for status in statuses):
            print(f"seed {seed}: statuses={statuses}")

        ordered_labels = [agent.label for agent in agents]
        update_elos(elos, ordered_labels, rewards, args.k_factor)

        row_by_label = dict(zip(ordered_labels, rewards))
        reward_rows.append([row_by_label[label] for label in labels])

        if (game_index + 1) % max(1, min(10, args.games)) == 0:
            print(f"completed {game_index + 1}/{args.games}", flush=True)

    summarize(labels, reward_rows, time.perf_counter() - started, elos)


if __name__ == "__main__":
    main()
