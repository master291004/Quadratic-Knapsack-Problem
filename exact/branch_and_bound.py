import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import load_instance
from exact.dynamic_programming import solve_dp, verify_solution


def compute_upper_bound(remaining_items, selected, weights, q, P, c, current_weight, current_profit):
    """
    Greedy LP relaxation upper bound.
    For each remaining item, compute its effective profit/weight ratio
    accounting for interactions with already selected items.
    Then fill fractionally like a continuous knapsack.
    """
    remaining_capacity = c - current_weight

    if remaining_capacity <= 0:
        return current_profit

    # compute effective profit for each remaining item
    candidates = []
    for j in remaining_items:
        # individual profit + interactions with already selected items
        effective_profit = q[j] + sum(P[i][j] for i in selected) + sum(P[k][j] for k in remaining_items if k != j)
        effective_weight = weights[j]

        if effective_weight <= 0:
            continue

        candidates.append((effective_profit / effective_weight, effective_profit, effective_weight, j))

    # sort by profit/weight ratio descending
    candidates.sort(reverse=True)

    # fill greedily, allowing fractional last item
    bound = current_profit
    remaining = remaining_capacity

    for ratio, profit, weight, j in candidates:
        if remaining <= 0:
            break
        if weight <= remaining:
            bound    += profit
            remaining -= weight
        else:
            # take fractional part
            bound += ratio * remaining
            remaining = 0

    return bound


def branch_and_bound(n, weights, q, P, c):
    """
    Branch and Bound solver for QKP.

    Uses depth-first search with LP relaxation upper bound.
    Branches by fixing items one at a time to 0 or 1.
    """

    best_profit  = [0]       # use list so inner function can modify it
    best_subset  = [[]]
    nodes_explored = [0]

    def backtrack(index, selected, current_weight, current_profit, remaining_items):

        nodes_explored[0] += 1

        # update best solution
        if current_profit > best_profit[0]:
            best_profit[0] = current_profit
            best_subset[0] = list(selected)

        # base case — no more items to consider
        if index == n:
            return

        # pruning — compute upper bound on what we can still achieve
        ub = compute_upper_bound(
            remaining_items,   # items after current
            selected,
            weights, q, P, c,
            current_weight, current_profit
        )

        if ub <= best_profit[0]:
            return   # prune this branch

        item = remaining_items[0]

        # ── branch 1: include item ────────────────────────────────────────────
        if current_weight + weights[item] <= c:
            # profit gained by adding this item
            gain = q[item] + sum(P[i][item] for i in selected)
            selected.append(item)
            backtrack(
                index + 1,
                selected,
                current_weight + weights[item],
                current_profit + gain,
                remaining_items[1:]
            )
            selected.pop()

        # ── branch 2: exclude item ────────────────────────────────────────────
        backtrack(
            index + 1,
            selected,
            current_weight,
            current_profit,
            remaining_items[1:]
        )

    # sort items by profit/weight ratio for better pruning
    # use total profit potential: q[j] + sum of all possible interactions
    def item_score(j):
        total_interaction = sum(P[i][j] for i in range(n) if i != j)
        return (q[j] + total_interaction) / weights[j]

    item_order = sorted(range(n), key=item_score, reverse=True)

    backtrack(
        index            = 0,
        selected         = [],
        current_weight   = 0,
        current_profit   = 0,
        remaining_items  = item_order
    )

    # recover original indices
    best_subset_original = sorted(best_subset[0])

    return best_profit[0], best_subset_original, nodes_explored[0]


# ── run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":

    test_files = [
        f"data/instances/n{n}_d{int(d*100)}_s{s}.json"
        for n in [10, 15, 20]
        for d in [0.25, 0.50, 0.75, 1.0]
        for s in range(10)
    ]

    print(f"{'Instance':<30} {'B&B':>8} {'DP':>8} {'Nodes':>10} {'Time(s)':>10} {'Check':>8}")
    print("-" * 80)

    total_wrong = 0

    for path in test_files:
        inst = load_instance(path)
        n, weights, q, P, c = (
            inst["n"],
            inst["weights"],
            inst["q"],
            inst["P"],
            inst["c"]
        )

        # run B&B
        start            = time.time()
        bb_profit, bb_subset, nodes = branch_and_bound(n, weights, q, P, c)
        elapsed          = time.time() - start

        # run DP as ground truth
        dp_profit, _     = solve_dp(n, weights, q, P, c)

        check = "OK" if bb_profit == dp_profit else "WRONG"
        if check == "WRONG":
            total_wrong += 1

        name = os.path.basename(path)
        print(f"{name:<30} {bb_profit:>8} {dp_profit:>8} "
              f"{nodes:>10} {elapsed:>10.4f} {check:>8}")

    print("-" * 80)
    print(f"Total wrong: {total_wrong}")