import time
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import load_instance
from exact.dynamic_programming import solve_dp
from exact.ILP import solve_ilp
from simulated_annealing import compute_profit


# ── helper functions ──────────────────────────────────────────────────────────

def compute_marginal_profit(item, subset, q, P):
    """
    Profit contribution of a single item given current subset.
    Includes individual profit and all interactions with subset members.
    """
    return q[item] + sum(P[item][k] for k in subset if k != item)


def compute_swap_score(out, inn, subset, weights, c, q, P):
    """
    Net profit change of swapping item `out` for item `inn`.

    score > 0 means the swap improves the solution.
    score < 0 means the swap worsens the solution.

    The swap is only geometrically valid if the new weight fits in c.
    """
    total_weight = sum(weights[k] for k in subset)
    new_weight   = total_weight - weights[out] + weights[inn]

    if new_weight > c:
        return None   # infeasible swap

    # profit of the new subset after the swap
    subset_after = (subset - {out}) | {inn}

    gain = compute_marginal_profit(inn, subset_after - {inn}, q, P)
    loss = compute_marginal_profit(out, subset,               q, P)

    return gain - loss


def repair(subset, weights, c, q, P):
    """
    Remove items one by one in order of lowest marginal profit
    until the solution is feasible.
    """
    subset       = set(subset)
    total_weight = sum(weights[j] for j in subset)

    while total_weight > c:
        worst_item   = None
        worst_profit = float('inf')

        for j in subset:
            m = compute_marginal_profit(j, subset, q, P)
            if m < worst_profit:
                worst_profit = m
                worst_item   = j

        subset.remove(worst_item)
        total_weight -= weights[worst_item]

    return subset


# ── greedy construction ───────────────────────────────────────────────────────

def greedy_construction(n, weights, c, q, P, rng, randomized=True, alpha=0.3):
    """
    Build a feasible solution greedily.

    If randomized=True (ILS restarts):
        Use a Restricted Candidate List (RCL) — at each step, instead of
        always picking the single best item, randomly pick from the top
        alpha fraction of remaining candidates.
        This gives different starting points for each restart.

    If randomized=False (first construction):
        Pure greedy — always pick the best item.
        Gives the best possible starting solution.

    Score of item j accounts for:
        - individual profit q[j]
        - interactions with already selected items (certain gain)
        - interactions with other remaining items (scaled by alpha — optimistic)
        divided by weight (profit per unit weight)
    """
    selected     = set()
    total_weight = 0
    remaining    = list(range(n))

    while remaining:
        # compute score for each remaining item
        scores = []
        for j in remaining:
            if total_weight + weights[j] > c:
                continue   # skip infeasible items

            interaction_selected  = sum(P[j][k] for k in selected)
            interaction_remaining = sum(P[j][k] for k in remaining if k != j)

            # adaptive score: interactions with selected are certain,
            # interactions with remaining are optimistic (scaled by alpha)
            score = (q[j] + interaction_selected + alpha * interaction_remaining) \
                    / weights[j]

            scores.append((score, j))

        if not scores:
            break   # no more items fit

        if randomized:
            # build RCL: items within alpha range of best score
            scores.sort(reverse=True)
            best_score  = scores[0][0]
            worst_score = scores[-1][0]
            threshold   = best_score - alpha * (best_score - worst_score)
            rcl         = [j for score, j in scores if score >= threshold]

            # pick randomly from RCL
            chosen = rng.choice(rcl)
        else:
            # pure greedy — always pick best
            chosen = max(scores, key=lambda x: x[0])[1]

        selected.add(chosen)
        total_weight += weights[chosen]
        remaining.remove(chosen)

    return selected


# ── local search (guided swapping) ────────────────────────────────────────────

def local_search(subset, n, weights, c, q, P, rng):
    """
    Semi-guided local search via improving swaps.

    At each step:
        1. Shuffle the order of selected items randomly
           (semi-guided: not always checking in the same order)
        2. For each selected item, find the best unselected item to swap it with
        3. If any swap improves profit, take the best one found
        4. Repeat until no improving swap exists (local optimum)

    Also tries pure additions (no removal) if capacity allows.

    Time complexity: O(n^2) per iteration, O(n^3) total in worst case.
    """
    subset         = set(subset)
    total_weight   = sum(weights[j] for j in subset)
    current_profit = compute_profit(subset, q, P)

    improved = True

    while improved:
        improved   = False
        best_delta = 0
        best_move  = None

        selected   = list(subset)
        unselected = [j for j in range(n) if j not in subset]

        # randomize order — semi-guided aspect
        rng.shuffle(selected)
        rng.shuffle(unselected)

        # ── try pure additions ────────────────────────────────────────────────
        for inn in unselected:
            if total_weight + weights[inn] > c:
                continue
            delta = compute_marginal_profit(inn, subset, q, P)
            if delta > best_delta:
                best_delta = delta
                best_move  = ("add", None, inn)

        # ── try swaps ─────────────────────────────────────────────────────────
        for out in selected:
            for inn in unselected:
                delta = compute_swap_score(out, inn, subset, weights, c, q, P)
                if delta is None:
                    continue   # infeasible
                if delta > best_delta:
                    best_delta = delta
                    best_move  = ("swap", out, inn)

        # apply the best move found
        if best_move is not None:
            move_type, out, inn = best_move

            if move_type == "add":
                subset.add(inn)
                total_weight   += weights[inn]
                current_profit += best_delta

            elif move_type == "swap":
                subset.remove(out)
                subset.add(inn)
                total_weight   += weights[inn] - weights[out]
                current_profit += best_delta

            improved = True

    return subset, current_profit


# ── perturbation ──────────────────────────────────────────────────────────────

def perturb(subset, n, weights, c, q, P, rng, strength=3):
    """
    Force `strength` random swaps regardless of profit impact.
    This kicks the solution out of its current local optimum
    so the next local search phase explores a different region.

    Strength controls how far we move from the current solution:
    - strength=2 : small perturbation, nearby region
    - strength=4 : large perturbation, very different region
    """
    subset = set(subset)

    for _ in range(strength):
        selected   = list(subset)
        unselected = [j for j in range(n) if j not in subset]

        if not selected or not unselected:
            break

        # force a random swap — profit does not matter here
        out = rng.choice(selected)
        inn = rng.choice(unselected)

        subset.remove(out)
        subset.add(inn)

    # repair in case perturbation caused infeasibility
    subset = repair(subset, weights, c, q, P)
    return subset


# ── iterated local search ─────────────────────────────────────────────────────

def iterated_local_search(n, weights, q, P, c,
                          max_iterations    = 100,
                          perturbation_strength = 3,
                          alpha             = 0.3,
                          seed              = 42):
    """
    Iterated Local Search for QKP.

    Algorithm:
        1. Build initial solution with pure greedy (best start)
        2. Improve with local search until local optimum
        3. Record best solution found
        4. Loop:
            a. Perturb current solution (escape local optimum)
            b. Improve with local search
            c. Update best if improved
            d. If no improvement for patience iterations, 
               restart from a new random greedy solution

    Parameters:
        max_iterations        : total number of ILS iterations
        perturbation_strength : how many forced swaps in perturbation
        alpha                 : RCL parameter for randomized greedy
        seed                  : random seed for reproducibility
    """
    rng = random.Random(seed)

    # ── phase 1: initial solution ─────────────────────────────────────────────
    current_subset = greedy_construction(
        n, weights, c, q, P, rng, randomized=False
    )
    current_subset, current_profit = local_search(
        current_subset, n, weights, c, q, P, rng
    )

    best_subset = set(current_subset)
    best_profit = current_profit

    no_improve  = 0
    patience    = max(20, max_iterations // 5)

    # ── main ILS loop ─────────────────────────────────────────────────────────
    for iteration in range(max_iterations):

        # ── restart if stuck for too long ─────────────────────────────────────
        if no_improve >= patience:
            # build a fresh randomized greedy solution
            current_subset = greedy_construction(
                n, weights, c, q, P, rng, randomized=True, alpha=alpha
            )
            current_subset, current_profit = local_search(
                current_subset, n, weights, c, q, P, rng
            )
            no_improve = 0

        # ── perturbation: escape current local optimum ────────────────────────
        perturbed_subset = perturb(
            current_subset, n, weights, c, q, P, rng,
            strength=perturbation_strength
        )

        # ── local search: improve from perturbed solution ─────────────────────
        improved_subset, improved_profit = local_search(
            perturbed_subset, n, weights, c, q, P, rng
        )

        # ── acceptance criterion: accept if improved ──────────────────────────
        # (unlike SA we do not accept worse solutions —
        #  the perturbation handles diversification instead)
        if improved_profit > current_profit:
            current_subset = improved_subset
            current_profit = improved_profit

        # ── update global best ────────────────────────────────────────────────
        if current_profit > best_profit:
            best_profit = current_profit
            best_subset = set(current_subset)
            no_improve  = 0
        else:
            no_improve += 1

    return best_profit, sorted(best_subset), max_iterations


# ── run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":

    test_files = [
        f"data/instances/n{n}_d{int(d*100)}_s{s}.json"
        for n in [10, 15, 20, 30, 50, 100]
        for d in [0.25, 0.50, 0.75, 1.0]
        for s in range(10)
    ]

    print(f"{'Instance':<30} {'ILS':>8} {'OPT':>8} "
          f"{'Gap%':>8} {'Time(s)':>10}")
    print("-" * 70)

    total_gap   = 0
    total_count = 0

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

        # run ILS
        start      = time.time()
        ils_profit, ils_subset, iters = iterated_local_search(
            n, weights, q, P, c
        )
        elapsed    = time.time() - start

        # ground truth
        if n <= 20:
            opt, _  = solve_dp(n, weights, q, P, c)
            gap     = (opt - ils_profit) / opt * 100 if opt > 0 else 0.0
            opt_str = str(opt)
            total_gap   += gap
            total_count += 1
        elif n <= 60:
            opt, _, status, _ = solve_ilp(
                n, weights, q, P, c, time_limit=60
            )
            gap     = (opt - ils_profit) / opt * 100 if opt > 0 else 0.0
            opt_str = str(opt)
            total_gap   += gap
            total_count += 1
        else:
            gap     = 0.0
            opt_str = "N/A"

        name = os.path.basename(path)
        print(f"{name:<30} {ils_profit:>8} {opt_str:>8} "
              f"{gap:>8.2f} {elapsed:>10.4f}")

    print("-" * 70)
    if total_count > 0:
        print(f"Average gap (verified instances): "
              f"{total_gap / total_count:.2f}%")