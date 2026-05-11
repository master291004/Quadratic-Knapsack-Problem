import os
import re
import glob
import csv
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # safe for terminal / server runs
import matplotlib.pyplot as plt
import random
from data.generator import load_instance
from exact.dynamic_programming import solve_dp
from exact.ILP import solve_ilp
from exact.branch_and_bound import branch_and_bound
from approx.genetic import genetic_algorithm
from approx.ILS import iterated_local_search
from approx.simulated_annealing import simulated_annealing

# Optional improved B&B import
try:
    from exact.branch_and_bound_improved import branch_and_bound_improved
except ImportError:
    branch_and_bound_improved = None


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

TARGET_SIZES = [10, 15, 20, 30, 50, 100]
TARGET_DENSITIES = [0.25, 0.50, 0.75, 1.0]
TARGET_CAPACITY_RATIOS = [0.25, 0.50, 0.75]
TARGET_SEEDS = list(range(10))

INSTANCE_DIR = "data/instances"
OUTPUT_CSV = "benchmark_results.csv"
PLOT_DIR = "benchmark_plots"

REFERENCE_ILP_TIME_LIMIT = 60

METHODS = ["ILP", "DP", "SA", "ILS", "GA", "B&B"]
if branch_and_bound_improved is not None:
    METHODS.append("B&B Improved")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def is_close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def extract_capacity_ratio(instance, path):
    """
    Prefer the value stored in JSON.
    If missing, try to parse from filename pattern *_cXX_*.
    """
    if "capacity_ratio" in instance and instance["capacity_ratio"] is not None:
        return float(instance["capacity_ratio"])

    name = os.path.basename(path)
    m = re.search(r"_c(\d+)", name)
    if m:
        return int(m.group(1)) / 100.0

    return None


def extract_seed(instance, path):
    """
    Prefer the value stored in JSON.
    If missing, try to parse from filename pattern *_sX*.
    """
    if "seed" in instance and instance["seed"] is not None:
        return int(instance["seed"])

    name = os.path.basename(path)
    m = re.search(r"_s(\d+)", name)
    if m:
        return int(m.group(1))

    return None


def classify_instance(density, capacity_ratio):
    """
    Structural difficulty label.
    EASY  : sparse density and loose capacity
    HARD  : dense density and tight capacity
    MEDIUM: everything in between
    """
    if capacity_ratio is None:
        if density <= 0.25:
            return "EASY"
        elif density >= 0.75:
            return "HARD"
        return "MEDIUM"

    if density <= 0.25 and capacity_ratio >= 0.75:
        return "EASY"
    elif density >= 0.75 and capacity_ratio <= 0.25:
        return "HARD"
    else:
        return "MEDIUM"


def compute_profit_from_subset(subset, q, P):
    """
    Recompute QKP profit from a subset.
    """
    if subset is None:
        return None

    subset = sorted(subset)
    profit = sum(q[i] for i in subset)
    for a in range(len(subset)):
        i = subset[a]
        for b in range(a + 1, len(subset)):
            j = subset[b]
            profit += P[i][j]
    return profit


def compute_weight_from_subset(subset, weights):
    if subset is None:
        return None
    return sum(weights[i] for i in subset)


def normalize_result(raw):
    """
    Generic unpacker for solver outputs.
    Expected common pattern:
      (profit, subset, ...)
    """
    profit = None
    subset = None
    extra = []

    if isinstance(raw, tuple):
        if len(raw) >= 1:
            profit = raw[0]
        if len(raw) >= 2:
            subset = raw[1]
        if len(raw) > 2:
            extra = list(raw[2:])
    else:
        profit = raw

    if subset is not None and not isinstance(subset, (list, tuple, set)):
        subset = None

    return profit, subset, extra


def discover_test_files():
    """
    Discover JSON instances in data/instances and filter them by the
    target grid defined above.
    """
    files = sorted(glob.glob(os.path.join(INSTANCE_DIR, "*.json")))
    selected = []

    for path in files:
        try:
            inst = load_instance(path)
        except Exception:
            continue

        n = inst.get("n", None)
        density = inst.get("density", None)
        cap_ratio = extract_capacity_ratio(inst, path)
        seed = extract_seed(inst, path)

        if n not in TARGET_SIZES:
            continue
        if density is None or not any(is_close(density, d) for d in TARGET_DENSITIES):
            continue

        # If the new generator metadata is present, enforce it.
        if cap_ratio is not None:
            if not any(is_close(cap_ratio, c) for c in TARGET_CAPACITY_RATIOS):
                continue

        if seed is not None and seed not in TARGET_SEEDS:
            continue

        selected.append(path)

    return selected


def execute_method(method, n, weights, q, P, c, seed=42):
    """
    Execute one solver / heuristic and return a unified dictionary.
    """
    start = time.perf_counter()

    result = {
        "profit": None,
        "subset": None,
        "runtime": None,
        "iterations": None,
        "solver_status": "",
        "error": None,
    }

    try:
        if method == "DP":
            profit, subset = solve_dp(n, weights, q, P, c)
            result["profit"] = profit
            result["subset"] = subset
            result["solver_status"] = "OPTIMAL"

        elif method == "ILP":
            profit, subset, status, solver_time = solve_ilp(
                n, weights, q, P, c, time_limit=REFERENCE_ILP_TIME_LIMIT
            )
            result["profit"] = profit
            result["subset"] = subset
            result["solver_status"] = status
            result["solver_time"] = solver_time

        elif method == "SA":
            profit, subset, iterations = simulated_annealing(
                n, weights, q, P, c, seed=seed
            )
            result["profit"] = profit
            result["subset"] = subset
            result["iterations"] = iterations
            result["solver_status"] = "OK"

        elif method == "ILS":
            profit, subset, iterations = iterated_local_search(
                n, weights, q, P, c, seed=seed
            )
            result["profit"] = profit
            result["subset"] = subset
            result["iterations"] = iterations
            result["solver_status"] = "OK"

        elif method == "GA":
            profit, subset, generations = genetic_algorithm(
                n, weights, q, P, c, seed=seed
            )
            result["profit"] = profit
            result["subset"] = subset
            result["iterations"] = generations
            result["solver_status"] = "OK"

        elif method == "B&B":
            raw = branch_and_bound(n, weights, q, P, c)
            profit, subset, extra = normalize_result(raw)
            result["profit"] = profit
            result["subset"] = subset
            if extra:
                result["solver_status"] = str(extra[0])
            else:
                result["solver_status"] = "OK"

        elif method == "B&B Improved":
            if branch_and_bound_improved is None:
                return None
            raw = branch_and_bound_improved(n, weights, q, P, c)
            profit, subset, extra = normalize_result(raw)
            result["profit"] = profit
            result["subset"] = subset
            if extra:
                result["solver_status"] = str(extra[0])
            else:
                result["solver_status"] = "OK"

        else:
            return None

    except Exception as e:
        result["error"] = str(e)
        result["solver_status"] = "ERROR"

    result["runtime"] = time.perf_counter() - start
    return result

def exact_reference(n, weights, q, P, c, seed=42):
    """
    True optimal reference using exact methods.
    Only feasible for small instances.
    """

    start = time.time()

    if n <= 20:
        res = execute_method("DP", n, weights, q, P, c, seed=seed)
        method = "DP"
    else:
        # ILP is allowed but time-limited
        res = execute_method("ILP", n, weights, q, P, c, seed=seed)
        method = "ILP"

    end = time.time()

    return {
        "profit": res["profit"],
        "reference_method": method,
        "solver_status": res.get("solver_status", "OPT"),
        "runtime": end - start
    }   
def heuristic_reference(n, weights, q, P, c, seed=42, n_runs=3):
    """
    Best Known Solution (BKS) reference for large instances.
    Combines multiple metaheuristics and keeps the best result.
    """

    rng = random.Random(seed)

    best_profit = float("-inf")
    best_method = "BKS"
    start = time.time()

    # ─────────────────────────────
    # GA runs
    # ─────────────────────────────
    for _ in range(n_runs):
        profit, _, _ = genetic_algorithm(
            n, weights, q, P, c,
            seed=rng.randint(0, 10**9)
        )
        if profit > best_profit:
            best_profit = profit
            best_method = "GA"

    # ─────────────────────────────
    # ILS runs
    # ─────────────────────────────
    for _ in range(n_runs):
        profit, _, _ = iterated_local_search(
            n, weights, q, P, c
        )
        if profit > best_profit:
            best_profit = profit
            best_method = "ILS"

    # ─────────────────────────────
    # SA runs
    # ─────────────────────────────
    for _ in range(n_runs):
        profit, _, _ = simulated_annealing(
            n, weights, q, P, c
        )
        if profit > best_profit:
            best_profit = profit
            best_method = "SA"

    end = time.time()

    return {
        "profit": best_profit,
        "reference_method": best_method,
        "solver_status": "BKS (heuristic best known solution)",
        "runtime": end - start
    }
def get_reference(n, weights, q, P, c):
    if n <= 20:
        return exact_reference(n,weights,q,P,c)
    else:
        return heuristic_reference(n,weights,q,P,c)


def safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


# ─────────────────────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    test_files = discover_test_files()
    if not test_files:
        raise RuntimeError("No matching instances found in data/instances.")

    rows = []

    print("Running benchmark...\n")

    for path in test_files:
        inst = load_instance(path)

        n = inst["n"]
        weights = inst["weights"]
        q = inst["q"]
        P = inst["P"]
        c = inst["c"]

        density = float(inst.get("density", 0.5))
        cap_ratio = extract_capacity_ratio(inst, path)
        interaction_type = inst.get("interaction_type", "balanced")
        seed = extract_seed(inst, path)
        if seed is None:
            seed = 42

        difficulty = classify_instance(density, cap_ratio)

        reference = get_reference(n, weights, q, P, c)
        ref_profit = reference["profit"]
        ref_method = reference["reference_method"]
        ref_status = reference["solver_status"]
        ref_time = reference["runtime"]

        for method in METHODS:
            if (method in ["DP", "B&B Improved", "B&B"] and n > 20):
                continue
            if method == "ILP" and n > 50:
                continue
            if method == "B&B Improved" and branch_and_bound_improved is None:
                continue

            # Reuse the reference call when the method is the same solver.
            if method == ref_method and method in {"DP", "ILP"}:
                res = reference.copy()
            else:
                res = execute_method(method, n, weights, q, P, c, seed=seed)

            if res is None:
                continue

            profit = res["profit"]
            subset = res["subset"]
            runtime = res["runtime"]
            iterations = res["iterations"]
            solver_status = res["solver_status"]
            error = res["error"]

            if error is not None or profit is None:
                subset_size = None
                subset_weight = None
                recomputed_profit = None
                feasible = None
                profit_verified = False
                gap = None
                better_than_reference = None
                solution_str = ""
            else:
                subset = list(subset) if subset is not None else None
                subset = sorted(subset) if subset is not None else None

                subset_size = len(subset) if subset is not None else None
                subset_weight = compute_weight_from_subset(subset, weights)
                recomputed_profit = compute_profit_from_subset(subset, q, P)
                feasible = (subset_weight <= c + 1e-9) if subset_weight is not None else None
                profit_verified = (
                    recomputed_profit is not None
                    and abs(recomputed_profit - profit) <= 1e-6
                )

                if ref_profit is not None and abs(ref_profit) > 1e-9:
                    gap = (ref_profit - profit) / ref_profit * 100.0
                else:
                    gap = 0.0

                better_than_reference = (
                    profit > ref_profit + 1e-9 if ref_profit is not None else None
                )

                solution_str = "" if subset is None else ";".join(map(str, subset))

            rows.append({
                "instance": os.path.basename(path),
                "path": path,
                "n": n,
                "density": density,
                "capacity_ratio": cap_ratio,
                "interaction_type": interaction_type,
                "difficulty": difficulty,

                "method": method,
                "reference_method": ref_method,
                "reference_status": ref_status,
                "reference_time_seconds": ref_time,
                "reference_profit": ref_profit,

                "profit": profit,
                "gap_percent": gap,
                "runtime_seconds": runtime,
                "subset_size": subset_size,
                "subset_weight": subset_weight,
                "feasible": feasible,
                "profit_verified": profit_verified,
                "iterations": iterations,
                "solver_status": solver_status,
                "better_than_reference": better_than_reference,
                "solution": solution_str,
                "error": error,
            })

            print(f"{os.path.basename(path):<30} | {method:<13} | "
                  f"profit={str(profit):>8} | ref={str(ref_profit):>8} | "
                  f"gap={str(round(gap, 2)) if gap is not None else 'N/A':>7} | "
                  f"time={runtime:>8.3f}s")

    # ─────────────────────────────────────────────────────────────
    # SAVE CSV
    # ─────────────────────────────────────────────────────────────

    fieldnames = [
        "instance", "path", "n", "density", "capacity_ratio",
        "interaction_type", "difficulty",
        "method", "reference_method", "reference_status",
        "reference_time_seconds", "reference_profit",
        "profit", "gap_percent", "runtime_seconds",
        "subset_size", "subset_weight", "feasible",
        "profit_verified", "iterations", "solver_status",
        "better_than_reference", "solution", "error"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {OUTPUT_CSV}")

    # ─────────────────────────────────────────────────────────────
    # SUMMARY TABLES
    # ─────────────────────────────────────────────────────────────

    completed = [r for r in rows if r["error"] is None and r["profit"] is not None]

    print("\n--- SUMMARY BY METHOD ---")
    for m in METHODS:
        subset_rows = [r for r in completed if r["method"] == m]
        if not subset_rows:
            continue

        avg_gap = safe_mean([r["gap_percent"] for r in subset_rows])
        avg_time = safe_mean([r["runtime_seconds"] for r in subset_rows])
        avg_size = safe_mean([r["subset_size"] for r in subset_rows])
        feasible_rate = 100.0 * sum(1 for r in subset_rows if r["feasible"]) / len(subset_rows)
        verified_rate = 100.0 * sum(1 for r in subset_rows if r["profit_verified"]) / len(subset_rows)

        print(f"{m:<14} gap={avg_gap:8.2f}% | time={avg_time:8.4f}s | "
              f"size={avg_size:6.2f} | feasible={feasible_rate:6.1f}% | "
              f"verified={verified_rate:6.1f}%")

    print("\n--- SUMMARY BY DIFFICULTY ---")
    for diff in ["EASY", "MEDIUM", "HARD"]:
        subset_rows = [r for r in completed if r["difficulty"] == diff]
        if not subset_rows:
            continue

        avg_gap = safe_mean([r["gap_percent"] for r in subset_rows])
        avg_time = safe_mean([r["runtime_seconds"] for r in subset_rows])

        print(f"{diff:<8} gap={avg_gap:8.2f}% | time={avg_time:8.4f}s")

    print("\n--- SUMMARY BY METHOD AND DIFFICULTY ---")
    for m in METHODS:
        for diff in ["EASY", "MEDIUM", "HARD"]:
            subset_rows = [r for r in completed if r["method"] == m and r["difficulty"] == diff]
            if not subset_rows:
                continue

            avg_gap = safe_mean([r["gap_percent"] for r in subset_rows])
            avg_time = safe_mean([r["runtime_seconds"] for r in subset_rows])

            print(f"{m:<14} | {diff:<6} | gap={avg_gap:8.2f}% | time={avg_time:8.4f}s")

    # ─────────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────────

    def plot_gap_by_difficulty(rows):
        grouped = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["gap_percent"] is None:
                continue
            grouped[r["method"]][r["difficulty"]].append(r["gap_percent"])

        labels = METHODS
        diffs = ["EASY", "MEDIUM", "HARD"]
        x = list(range(len(labels)))
        width = 0.25

        plt.figure(figsize=(12, 6))
        for i, diff in enumerate(diffs):
            values = [
                safe_mean(grouped[m][diff]) if diff in grouped[m] else 0.0
                for m in labels
            ]
            plt.bar([xi + (i - 1) * width for xi in x], values, width=width, label=diff)

        plt.xticks(x, labels, rotation=20)
        plt.xlabel("Method")
        plt.ylabel("Average gap (%)")
        plt.title("Average Gap by Method and Difficulty")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "gap_by_difficulty.png"), dpi=200)
        plt.close()

    def plot_gap_vs_size(rows):
        grouped = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["gap_percent"] is None:
                continue
            grouped[r["method"]][r["n"]].append(r["gap_percent"])

        plt.figure(figsize=(10, 6))
        for m in METHODS:
            if m not in grouped:
                continue
            sizes = sorted(grouped[m].keys())
            values = [safe_mean(grouped[m][n]) for n in sizes]
            plt.plot(sizes, values, marker="o", label=m)

        plt.xlabel("n")
        plt.ylabel("Average gap (%)")
        plt.title("Average Gap vs Instance Size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "gap_vs_size.png"), dpi=200)
        plt.close()

    def plot_time_vs_size(rows):
        grouped = defaultdict(lambda: defaultdict(list))
        for r in rows:
            grouped[r["method"]][r["n"]].append(r["runtime_seconds"])

        plt.figure(figsize=(10, 6))
        for m in METHODS:
            if m not in grouped:
                continue
            sizes = sorted(grouped[m].keys())
            values = [safe_mean(grouped[m][n]) for n in sizes]
            plt.plot(sizes, values, marker="o", label=m)

        plt.xlabel("n")
        plt.ylabel("Average runtime (s)")
        plt.title("Average Runtime vs Instance Size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "time_vs_size.png"), dpi=200)
        plt.close()

    def plot_time_vs_difficulty(rows):
        grouped = defaultdict(lambda: defaultdict(list))
        for r in rows:
            grouped[r["method"]][r["difficulty"]].append(r["runtime_seconds"])

        labels = METHODS
        diffs = ["EASY", "MEDIUM", "HARD"]
        x = list(range(len(labels)))
        width = 0.25

        plt.figure(figsize=(12, 6))
        for i, diff in enumerate(diffs):
            values = [
                safe_mean(grouped[m][diff]) if diff in grouped[m] else 0.0
                for m in labels
            ]
            plt.bar([xi + (i - 1) * width for xi in x], values, width=width, label=diff)

        plt.xticks(x, labels, rotation=20)
        plt.xlabel("Method")
        plt.ylabel("Average runtime (s)")
        plt.title("Average Runtime by Method and Difficulty")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "time_by_difficulty.png"), dpi=200)
        plt.close()

    plot_gap_by_difficulty(completed)
    plot_gap_vs_size(completed)
    plot_time_vs_size(completed)
    plot_time_vs_difficulty(completed)

    print(f"Saved plots to {PLOT_DIR}/")


if __name__ == "__main__":
    main()