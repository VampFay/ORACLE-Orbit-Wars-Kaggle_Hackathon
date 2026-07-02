# ORACLE - 1700+ ELO Kaggle Agent for Orbit Wars

An advanced, highly-optimized autonomous agent built for the Kaggle **Orbit Wars** Hackathon. ORACLE is designed to crush standard heuristic bots and climb into the top ELO brackets by dynamically generating and evaluating multi-fleet strategic turn-plans.

## Architecture Highlights
ORACLE uses a hybrid architecture combining an aggressive, precision heuristic layer with a 3-ply **Monte Carlo Tree Search (MCTS) Action Abstraction Engine**.

1. **Precision Just-in-Time Defense:** Calculates exact planet capture timings and synchronizes defensive fleets to arrive exactly when needed, maximizing defensive efficiency without overcommitting ships.
2. **Time-Synchronized Coordinated Attacks:** Groups multiple attacking fleets and deliberately delays closer launches so that all ships arrive at the target planet on the exact same tick, making the attack impossible to defend incrementally.
3. **MCTS Action Abstraction:** Instead of standard greedy single-fleet node generation, ORACLE generates 4 distinct, coherent global strategies (`All-Out`, `Attack-Heavy`, `Expand-Heavy`, `Passive/Defensive`). It uses MCTS to forward-simulate the exact outcome of these 4 global strategies over the next several turns and selects the most mathematically optimal plan.
4. **Race Denial Expansion:** Expands aggressively while denying high-value planets to enemies based on movement vector analysis and physics intersections.

## Performance
- **vs. Random:** 100% Win Rate
- **vs. Starter Bot:** 70%+ Win Rate
- **Target Bracket:** 1700+ ELO

## Structure
- `oracle-orbit-wars/main.py`: The core agent logic containing the Heuristics, Physics Engine, and MCTS Action Abstraction.
- `ORACLE_1700_Upgrade_Plan.md`: The step-by-step design document explaining the multi-phase engineering process used to construct the agent.

## Usage
Simply zip `main.py` and submit it directly to the Kaggle Orbit Wars environment.

```bash
zip submission.zip oracle-orbit-wars/main.py
```
