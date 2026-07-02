"""Test script for ORACLE agent — runs 20 games vs starter."""
import random
import logging

logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(logging.ERROR)

from kaggle_environments import make
import main

def main_test():
    print("=== ORACLE Agent Test ===\n")

    # vs starter
    w, l = 0, 0
    for i in range(20):
        seed = random.randint(0, 999999)
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([lambda o, c=None: main.agent(o, c), "starter"])
        r = env.steps[-1][0].reward
        if r == 1: w += 1
        elif r == -1: l += 1
    print(f"vs starter (20 games): {w}W / {l}L = {100*w/20:.0f}%")

    # vs random
    w, l = 0, 0
    for i in range(10):
        seed = random.randint(0, 999999)
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([lambda o, c=None: main.agent(o, c), "random"])
        r = env.steps[-1][0].reward
        if r == 1: w += 1
        elif r == -1: l += 1
    print(f"vs random (10 games):  {w}W / {l}L = {100*w/10:.0f}%")

    print("\n=== Done ===")

if __name__ == "__main__":
    main_test()
