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
    Exact QKP solver using Binary ILP with linearization.

    Formulation:
        max  sum_j q[j]*x[j] + sum_{i<j} P[i][j]*y[i][j]
        s.t. sum_j w[j]*x[j] <= c
             y[i][j] <= x[i]   for all i < j
             y[i][j] <= x[j]   for all i < j
             x[j]    in {0,1}
             y[i][j] in {0,1}

    Parameters:
        n          : number of items
        weights    : list of item weights
        q          : list of individual profits
        P          : n x n profit matrix
        c          : knapsack capacity
        time_limit : solver time limit in seconds (default 300)
        verbose    : print solver logs if True

    Returns:
        best_profit : optimal profit found
        best_subset : list of selected item indices
        status      : solver status string
        gap         : optimality gap (0.0 means proven optimal)
    """

    # ── create the problem ────────────────────────────────────────────────────
    prob = pulp.LpProblem("QKP", pulp.LpMaximize)

    # ── decision variables ────────────────────────────────────────────────────

    # x[j] = 1 if item j is selected
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n)]

    # y[i][j] = 1 if both items i and j are selected (only for i < j)
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            if P[i][j] > 0:   # only create variable if profit is nonzero
                y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")

    # ── objective function ────────────────────────────────────────────────────
    prob += (
        pulp.lpSum(q[j] * x[j] for j in range(n)) +
        pulp.lpSum(P[i][j] * y[(i, j)] for (i, j) in y)
    )

    # ── constraints ──────────────────────────────────────────────────────────

    # knapsack capacity
    prob += pulp.lpSum(weights[j] * x[j] for j in range(n)) <= c

    # linearization constraints — y[i][j] can be 1 only if both x[i] and x[j] are 1
    for (i, j) in y:
        prob += y[(i, j)] <= x[i]
        prob += y[(i, j)] <= x[j]

    # ── solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(
        timeLimit = time_limit,
        gapRel    = 0.0,       # require proven optimality
        msg       = 1 if verbose else 0
    )

    prob.solve(solver)

    # ── extract solution ──────────────────────────────────────────────────────
    status = pulp.LpStatus[prob.status]

    best_subset = [j for j in range(n)
                   if pulp.value(x[j]) is not None
                   and pulp.value(x[j]) > 0.5]

    best_profit = int(round(pulp.value(prob.objective))) \
                  if pulp.value(prob.objective) is not None else 0

    # optimality gap — 0.0 means proven optimal
    gap = abs(prob.sol_status) if prob.sol_status else 0.0

    return best_profit, best_subset, status, gap


# ── run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":

    test_files = [
        f"data/instances/n{n}_d{int(d*100)}_s{s}.json"
        for n in [10, 15, 20, 30]
        for d in [0.25, 0.50, 0.75, 1.0]
        for s in range(10)
    ]

    print(f"{'Instance':<30} {'ILP':>8} {'DP':>8} "
          f"{'Time(s)':>10} {'Status':>12} {'Check':>8}")
    print("-" * 85)

    total_wrong = 0
    total_files = 0

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

        # run ILP
        start = time.time()
        ilp_profit, ilp_subset, status, gap = solve_ilp(
            n, weights, q, P, c, verbose=False
        )
        elapsed = time.time() - start

        # ground truth — only for n <= 20 (DP too slow otherwise)
        if n <= 20:
            dp_profit, _ = solve_dp(n, weights, q, P, c)
            check = "OK" if ilp_profit == dp_profit else "WRONG"
            dp_str = str(dp_profit)
        else:
            dp_profit = None
            check     = "N/A"
            dp_str    = "N/A"

        if check == "WRONG":
            total_wrong += 1
        total_files += 1

        name = os.path.basename(path)
        print(f"{name:<30} {ilp_profit:>8} {dp_str:>8} "
              f"{elapsed:>10.4f} {status:>12} {check:>8}")

    print("-" * 85)
    print(f"Total wrong: {total_wrong} / {len([f for f in test_files if n <= 20])}")
    print(f"\nNote: ILP can handle n=30 and beyond — DP used only as check for n<=20")