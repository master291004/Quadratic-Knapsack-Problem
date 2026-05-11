import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import load_instance
from exact.dynamic_programming import solve_dp, verify_solution


def analyze_instance(n, weights, q, P, c):
    """
    Computes two key metrics to characterize the instance:
    - q_ratio : avg individual profit / avg interaction profit
                > 1 means individual profits dominate
                < 1 means interactions dominate
    - density : fraction of nonzero pair profits
    """
    avg_q = sum(q) / n

    total_interactions = 0
    nonzero_pairs      = 0
    num_pairs          = n * (n - 1) / 2

    for i in range(n):
        for j in range(i + 1, n):
            if P[i][j] > 0:
                total_interactions += P[i][j]
                nonzero_pairs      += 1

    avg_interaction = total_interactions / nonzero_pairs if nonzero_pairs > 0 else 0
    density         = nonzero_pairs / num_pairs if num_pairs > 0 else 0
    q_ratio         = avg_q / avg_interaction if avg_interaction > 0 else float('inf')

    return {
        "q_ratio" : q_ratio,
        "density" : density,
        "avg_q"   : avg_q,
        "avg_interaction" : avg_interaction
    }


def compute_upper_bound(remaining_items, selected, weights, q, P, c,
                        current_weight, current_profit, instance_stats):
    """
    Adaptive LP relaxation upper bound.

    Strategy A (q dominates / sparse interactions):
        Only count top-k interactions per item, where k scales with density.
        Much tighter when most pair profits are zero.

    Strategy B (interactions dominate / dense):
        Scale interaction_remaining by capacity ratio tau — the fraction
        of remaining items that can realistically be selected together.

    Both strategies always satisfy UB >= z* (valid upper bound).
    """
    remaining_capacity = c - current_weight

    if remaining_capacity <= 0:
        return current_profit

    if not remaining_items:
        return current_profit

    q_ratio = instance_stats["q_ratio"]
    density = instance_stats["density"]

    candidates = []

    for j in remaining_items:

        # interactions with already selected items — always count fully
        interaction_selected = sum(P[i][j] for i in selected)

        # ── adaptive interaction_remaining ────────────────────────────────────
        if q_ratio > 1.0 or density < 0.3:
            # Strategy A — individual profits dominate or instance is sparse
            # only keep the top-k interactions as an upper estimate
            k = max(1, int(density * len(remaining_items)))
            all_interactions = sorted(
                [P[kk][j] for kk in remaining_items if kk != j],
                reverse=True
            )
            interaction_remaining = sum(all_interactions[:k])

        else:
            # Strategy B — interactions dominate, instance is dense
            # scale by tau: fraction of remaining capacity vs remaining weight
            total_remaining_weight = sum(weights[kk] for kk in remaining_items)
            tau = min(remaining_capacity / total_remaining_weight, 1.0) \
                  if total_remaining_weight > 0 else 0.0
            interaction_remaining = tau * sum(
                P[kk][j] for kk in remaining_items if kk != j
            )

        effective_profit = q[j] + interaction_selected + interaction_remaining
        effective_weight = weights[j]

        if effective_weight > 0:
            candidates.append((
                effective_profit / effective_weight,
                effective_profit,
                effective_weight,
                j
            ))

    # sort by profit/weight ratio descending
    candidates.sort(reverse=True)

    # continuous knapsack — fractional last item allowed
    bound     = current_profit
    remaining = remaining_capacity

    for ratio, profit, weight, _ in candidates:
        if remaining <= 0:
            break
        if weight <= remaining:
            bound     += profit
            remaining -= weight
        else:
            bound     += ratio * remaining
            remaining  = 0

    return bound


def branch_and_bound(n, weights, q, P, c):
    """
    Branch and Bound solver for QKP.

    - Analyzes instance once at the start to choose bound strategy
    - Items ordered by total profit potential for better early pruning
    - Depth-first search, include branch explored before exclude branch
    - Upper bound computed at every node to prune invalid branches
    """

    # analyze instance once — result passed to every bound computation
    stats = analyze_instance(n, weights, q, P, c)

    best_profit    = [0]
    best_subset    = [[]]
    nodes_explored = [0]

    def backtrack(index, selected, current_weight,
                  current_profit, remaining_items):

        nodes_explored[0] += 1

        # update best solution found so far
        if current_profit > best_profit[0]:
            best_profit[0] = current_profit
            best_subset[0] = list(selected)

        # base case
        if not remaining_items:
            return

        # compute upper bound on best achievable profit from this node
        ub = compute_upper_bound(
            remaining_items,
            selected,
            weights, q, P, c,
            current_weight,
            current_profit,
            stats
        )

        # prune if even the optimistic bound cannot beat current best
        if ub <= best_profit[0]:
            return

        item = remaining_items[0]
        rest = remaining_items[1:]

        # ── branch 1: include item ────────────────────────────────────────────
        if current_weight + weights[item] <= c:
            gain = q[item] + sum(P[i][item] for i in selected)
            selected.append(item)
            backtrack(
                index + 1,
                selected,
                current_weight + weights[item],
                current_profit + gain,
                rest
            )
            selected.pop()

        # ── branch 2: exclude item ────────────────────────────────────────────
        backtrack(
            index + 1,
            selected,
            current_weight,
            current_profit,
            rest
        )

    # order items by total profit potential descending
    # good items first → tight lower bound early → more pruning
    def item_score(j):
        total_interaction = sum(P[i][j] for i in range(n) if i != j)
        return (q[j] + total_interaction) / weights[j]

    item_order = sorted(range(n), key=item_score, reverse=True)

    backtrack(
        index          = 0,
        selected       = [],
        current_weight = 0,
        current_profit = 0,
        remaining_items = item_order
    )

    return best_profit[0], sorted(best_subset[0]), nodes_explored[0]


# ── run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":

    test_files = [
        f"data/instances/n{n}_d{int(d*100)}_s{s}.json"
        for n in [10, 15, 20]
        for d in [0.25, 0.50, 0.75, 1.0]
        for s in range(10)
    ]

    print(f"{'Instance':<30} {'B&B':>8} {'DP':>8} {'Nodes':>10} "
          f"{'Time(s)':>10} {'Strategy':>12} {'Check':>8}")
    print("-" * 95)

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
        start                    = time.time()
        bb_profit, bb_subset, nodes = branch_and_bound(n, weights, q, P, c)
        elapsed                  = time.time() - start

        # ground truth
        dp_profit, _             = solve_dp(n, weights, q, P, c)

        # determine which strategy was used for display
        stats    = analyze_instance(n, weights, q, P, c)
        strategy = "A (q dom)" if stats["q_ratio"] > 1.0 \
                                   or stats["density"] < 0.3 \
                               else "B (interact)"

        check = "OK" if bb_profit == dp_profit else "WRONG"
        if check == "WRONG":
            total_wrong += 1

        name = os.path.basename(path)
        print(f"{name:<30} {bb_profit:>8} {dp_profit:>8} "
              f"{nodes:>10} {elapsed:>10.4f} {strategy:>12} {check:>8}")

    print("-" * 95)
    print(f"Total wrong: {total_wrong} / {len(test_files)}")