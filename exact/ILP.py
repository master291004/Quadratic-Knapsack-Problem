import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import load_instance
from exact.dynamic_programming import solve_dp

try:
    import pulp
except ImportError:
    print("PuLP not installed. Run: pip install pulp")
    sys.exit(1)


def solve_ilp(n, weights, q, P, c, time_limit=300, verbose=False):
    """
    Exact ILP solver for the Quadratic Knapsack Problem (QKP).

    Model:
        max  sum_i q_i x_i + sum_{i<j} P_ij y_ij
        s.t. sum_i w_i x_i <= c
             y_ij <= x_i
             y_ij <= x_j
             x_i, y_ij ∈ {0,1}

    Returns:
        best_profit: optimal objective value
        best_subset: selected item indices
        status: solver status
        solve_time: time spent solving
    """

    # ─────────────────────────────────────────────
    # Build model
    # ─────────────────────────────────────────────
    model = pulp.LpProblem("QKP", pulp.LpMaximize)

    # ── Decision variables ───────────────────────
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")

    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            if P[i][j] != 0:
                y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")

    # ── Objective function ───────────────────────
    model += (
        pulp.lpSum(q[i] * x[i] for i in range(n)) +
        pulp.lpSum(P[i][j] * y[(i, j)] for (i, j) in y)
    )

    # ── Capacity constraint ───────────────────────
    model += pulp.lpSum(weights[i] * x[i] for i in range(n)) <= c

    # ── Linearization constraints ────────────────
    for (i, j) in y:
        model += y[(i, j)] <= x[i]
        model += y[(i, j)] <= x[j]

    # ─────────────────────────────────────────────
    # Solve
    # ─────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(
        timeLimit=time_limit,
        msg=verbose
    )

    start = time.time()
    model.solve(solver)
    solve_time = time.time() - start

    status = pulp.LpStatus[model.status]

    # ─────────────────────────────────────────────
    # Extract solution
    # ─────────────────────────────────────────────
    best_subset = [
        i for i in range(n)
        if x[i].value() is not None and x[i].value() > 0.5
    ]

    best_profit = pulp.value(model.objective)
    if best_profit is None:
        best_profit = 0
    else:
        best_profit = int(round(best_profit))

    # ─────────────────────────────────────────────
    # Optional: feasibility check (debug-safe)
    # ─────────────────────────────────────────────
    assert sum(weights[i] for i in best_subset) <= c + 1e-6, \
        "Capacity constraint violated!"

    return best_profit, best_subset, status, solve_time

# ── run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":

    import os
    import time

    sizes = [10, 15, 20, 30]
    densities = [0.25, 0.50, 0.75, 1.0]
    seeds = range(10)

    test_files = [
        f"data/instances/n{n}_d{int(d*100)}_s{s}.json"
        for n in sizes
        for d in densities
        for s in seeds
    ]

    print(f"{'Instance':<35} {'ILP':>8} {'DP':>8} {'ILP(s)':>10} {'DP(s)':>10} {'Status':>10} {'Check':>8}")
    print("-" * 95)

    total_checked = 0
    total_wrong = 0

    for path in test_files:

        if not os.path.exists(path):
            continue

        inst = load_instance(path)
        n = inst["n"]
        weights = inst["weights"]
        q = inst["q"]
        P = inst["P"]
        c = inst["c"]

        # ─────────────────────────────
        # ILP solve
        # ─────────────────────────────
        t0 = time.time()
        ilp_profit, ilp_subset, status, _ = solve_ilp(n, weights, q, P, c)
        ilp_time = time.time() - t0

        # ─────────────────────────────
        # DP reference (small instances only)
        # ─────────────────────────────
        if n <= 20:
            t1 = time.time()
            dp_profit, _ = solve_dp(n, weights, q, P, c)
            dp_time = time.time() - t1

            check = "OK" if ilp_profit == dp_profit else "WRONG"
        else:
            dp_profit = "N/A"
            dp_time = "N/A"
            check = "N/A"

        # ─────────────────────────────
        # stats
        # ─────────────────────────────
        if check == "WRONG":
            total_wrong += 1

        total_checked += 1

        name = os.path.basename(path)

        print(f"{name:<35} {ilp_profit:>8} {dp_profit:>8} "
            f"{ilp_time:>10.4f} {dp_time:>10.4f} {status:>10} {check:>8}")
    print("-" * 95)
    print(f"Checked instances: {total_checked}")
    print(f"Wrong ILP vs DP: {total_wrong} (n <= 20 only)")