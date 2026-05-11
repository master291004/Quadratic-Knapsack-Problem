import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import load_instance


def solve_dp(n, weights, q, P, c):
    """
    Exact QKP solver using dynamic programming over bitmasks.
    
    Parameters:
        n       : number of items
        weights : list of item weights
        q       : list of individual profits
        P       : n x n profit matrix (P[i][j] = interaction profit)
        c       : knapsack capacity
    
    Returns:
        best_profit  : optimal profit value
        best_subset  : list of selected item indices
    """

    num_masks = 1 << n          # 2^n subsets
    dp       = [-1] * num_masks  # -1 means infeasible
    dp[0]    = 0                 # empty set, zero profit, zero weight

    # precompute total weight for each mask
    # this avoids recomputing it inside the inner loop
    total_weight = [0] * num_masks
    for mask in range(1, num_masks):
        # find the lowest set bit
        lowest_bit = mask & (-mask)
        j = lowest_bit.bit_length() - 1
        total_weight[mask] = total_weight[mask ^ lowest_bit] + weights[j]

    # fill the dp table
    for mask in range(num_masks):

        # skip infeasible or unvisited states
        if dp[mask] == -1:
            continue

        # skip if already over capacity
        if total_weight[mask] > c:
            continue

        # try adding each item not yet in the subset
        for j in range(n):
            if mask & (1 << j):
                continue        # item j already selected

            new_mask = mask | (1 << j)

            if total_weight[new_mask] > c:
                continue        # adding j exceeds capacity

            # marginal profit of adding item j to current subset
            marginal = q[j]
            for i in range(n):
                if mask & (1 << i):
                    marginal += P[i][j]

            new_profit = dp[mask] + marginal

            if new_profit > dp[new_mask]:
                dp[new_mask] = new_profit

    # find the best feasible subset
    best_profit = 0
    best_mask   = 0

    for mask in range(num_masks):
        if dp[mask] > best_profit:
            best_profit = dp[mask]
            best_mask   = mask

    # recover which items were selected
    best_subset = [j for j in range(n) if best_mask & (1 << j)]

    return best_profit, best_subset


def verify_solution(subset, weights, q, P, c):
    """
    Given a subset of items, recompute the profit from scratch
    and check feasibility. Used to verify the DP output is correct.
    """
    total_weight = sum(weights[j] for j in subset)
    assert total_weight <= c, f"infeasible: weight {total_weight} > capacity {c}"

    profit = sum(q[j] for j in subset)
    for idx, i in enumerate(subset):
        for j in subset[idx+1:]:
            profit += P[i][j] #+ P[j][i]

    return profit


# ── run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":

    from data.generator import load_instance

    # only test small instances — DP explodes for large n
    test_files = [
        f"data/instances/n{n}_d{int(d*100)}_s{s}.json"
        for n in [10, 15, 20]
        for d in [0.25, 0.50, 0.75, 1.0]
        for s in range(10)
    ]

    print(f"{'Instance':<30} {'Optimal':>10} {'Weight':>8} {'Time(s)':>10} {'Check':>8}")
    print("-" * 70)

    for path in test_files:
        inst = load_instance(path)
        n, weights, q, P, c = (
            inst["n"],
            inst["weights"],
            inst["q"],
            inst["P"],
            inst["c"]
        )

        start = time.time()
        best_profit, best_subset = solve_dp(n, weights, q, P, c)
        elapsed = time.time() - start

        # verify the solution independently
        verified_profit = verify_solution(best_subset, weights, q, P, c)
        check = "OK" if verified_profit == best_profit else "WRONG"

        name = os.path.basename(path)
        print(f"{name:<30} {best_profit:>10} "
              f"{sum(weights[j] for j in best_subset):>8} "
              f"{elapsed:>10.4f} {check:>8}")