import time
import math
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import load_instance
from exact.dynamic_programming import solve_dp


def compute_profit(subset, q, P):
    """
    Compute total profit of a subset from scratch.
    Used for initialization and verification.
    """
    profit = sum(q[j] for j in subset)
    subset_list = list(subset)
    for idx, i in enumerate(subset_list):
        for j in subset_list[idx + 1:]:
            profit += P[i][j]
    return profit


def compute_delta_add(item, current_subset, q, P):
    """
    Profit gained by ADDING item to current subset.
    O(n) — only counts interactions with already selected items.
    """
    return q[item] + sum(P[item][i] for i in current_subset)


def compute_delta_remove(item, current_subset, q, P):
    """
    Profit lost by REMOVING item from current subset.
    O(n) — only counts interactions with remaining items.
    """
    return q[item] + sum(P[item][i] for i in current_subset if i != item)


def greedy_initial_solution(n, weights, q, P, c):
    """
    Build an initial feasible solution greedily.
    Items sorted by (q[j] + sum of interactions) / weight descending.
    """
    scores = []
    for j in range(n):
        total_interaction = sum(P[j][i] for i in range(n) if i != j)
        score = (q[j] + total_interaction) / weights[j]
        scores.append((score, j))

    scores.sort(reverse=True)

    subset        = set()
    total_weight  = 0

    for _, j in scores:
        if total_weight + weights[j] <= c:
            subset.add(j)
            total_weight += weights[j]

    return subset, total_weight


def repair(subset, weights, c, q, P):
    """
    If solution is infeasible, remove items one by one
    in order of lowest marginal profit until feasible.
    """
    total_weight = sum(weights[j] for j in subset)

    while total_weight > c:
        # find item with lowest marginal profit contribution
        worst_item   = None
        worst_profit = float('inf')

        for j in subset:
            marginal = compute_delta_remove(j, subset, q, P)
            if marginal < worst_profit:
                worst_profit = marginal
                worst_item   = j

        subset.remove(worst_item)
        total_weight -= weights[worst_item]

    return subset, total_weight


def simulated_annealing(n, weights, q, P, c,
                        T_initial    = None,
                        T_final      = 0.1,
                        cooling_rate = 0.995,
                        max_iter     = 50000,
                        seed         = 42):
    """
    Simulated Annealing for QKP.

    Neighborhood: single bit flip (add or remove one item).
    Acceptance:   always accept improvements,
                  accept worse solutions with probability e^(-delta/T).

    Parameters:
        T_initial    : starting temperature (auto-set if None)
        T_final      : stopping temperature
        cooling_rate : alpha in T = alpha * T each iteration
        max_iter     : hard iteration limit
        seed         : random seed for reproducibility
    """
    rng = random.Random(seed)

    # ── initialization ────────────────────────────────────────────────────────
    current_subset, current_weight = greedy_initial_solution(
        n, weights, q, P, c
    )
    current_profit = compute_profit(current_subset, q, P)

    best_subset = set(current_subset)
    best_profit = current_profit

    # auto-set initial temperature if not provided
    # rule: at T_initial, accept a solution 10% worse with probability 0.8
    if T_initial is None:
        T_initial = current_profit * 0.1 / math.log(0.8) * (-1)
        T_initial = max(T_initial, 1.0)

    T = T_initial

    # ── main loop ─────────────────────────────────────────────────────────────
    iteration        = 0
    no_improve_count = 0

    while T > T_final and iteration < max_iter:

        iteration += 1

        # ── generate neighbor ─────────────────────────────────────────────────
        # decide whether to add or remove an item
        unselected = [j for j in range(n) if j not in current_subset]

        # if knapsack is full, can only remove
        # if empty, can only add
        # otherwise choose randomly
        if not current_subset:
            move_type = "add"
        elif not unselected:
            move_type = "remove"
        elif current_weight >= c:
            move_type = "remove"
        else:
            move_type = rng.choice(["add", "remove"])

        if move_type == "add":
            item  = rng.choice(unselected)
            delta = compute_delta_add(item, current_subset, q, P)

            new_weight = current_weight + weights[item]

            # if adding makes it infeasible, try to swap instead
            if new_weight > c:
                # swap: add item, remove the least profitable item
                current_subset.add(item)
                current_subset, new_weight = repair(
                    current_subset, weights, c, q, P
                )
                new_profit = compute_profit(current_subset, q, P)
                delta      = new_profit - current_profit

                # decide acceptance
                if delta > 0 or rng.random() < math.exp(delta / T):
                    current_weight = new_weight
                    current_profit = new_profit
                else:
                    # revert
                    current_subset, current_weight = greedy_initial_solution(
                        n, weights, q, P, c
                    )
                    current_profit = compute_profit(current_subset, q, P)

            else:
                # clean add — no repair needed
                delta = compute_delta_add(item, current_subset, q, P)

                if delta > 0 or rng.random() < math.exp(delta / T):
                    current_subset.add(item)
                    current_weight += weights[item]
                    current_profit += delta

        else:  # remove
            item  = rng.choice(list(current_subset))
            delta = -compute_delta_remove(item, current_subset, q, P)

            if delta > 0 or rng.random() < math.exp(delta / T):
                current_subset.remove(item)
                current_weight -= weights[item]
                current_profit += delta

        # ── update best ───────────────────────────────────────────────────────
        if current_profit > best_profit:
            best_profit = current_profit
            best_subset = set(current_subset)
            no_improve_count = 0
        else:
            no_improve_count += 1

        # ── cooling ───────────────────────────────────────────────────────────
        T *= cooling_rate

    return best_profit, sorted(best_subset), iteration


# ── run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":

    test_files = [
        f"data/instances/n{n}_d{int(d*100)}_s{s}.json"
        for n in [10, 15, 20, 30, 50, 100]
        for d in [0.25, 0.50, 0.75, 1.0]
        for s in range(10)
    ]

    print(f"{'Instance':<30} {'SA':>8} {'OPT':>8} "
      f"{'Gap%':>8} {'Iter':>8} {'Time(s)':>10} {'Type':>10}")
    print("-" * 80)

    for path in test_files:

        if not os.path.exists(path):
            continue

        inst = load_instance(path)
        n, weights, q, P, c = (
            inst["n"],
            inst["weights"],
            inst["q"],
            inst["P"],
            inst["c"]
        )
        itype = inst.get("interaction_type", "balanced")
        # run SA
        start      = time.time()
        sa_profit, sa_subset, iters = simulated_annealing(
            n, weights, q, P, c,seed=inst.get("seed", 42)
        )
        elapsed    = time.time() - start

        # ground truth for small instances only
        if n <= 20:
            dp_profit, _ = solve_dp(n, weights, q, P, c)
            gap = (dp_profit - sa_profit) / dp_profit * 100 \
                  if dp_profit > 0 else 0.0
            opt_str = str(dp_profit)
        else:
            gap     = 0.0
            opt_str = "N/A"

        name = os.path.basename(path)
        print(f"{name:<30} {sa_profit:>8} {opt_str:>8} "
      f"{gap:>8.2f} {iters:>8} {elapsed:>10.4f} {itype:>10}")