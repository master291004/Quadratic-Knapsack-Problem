import time
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import load_instance
from exact.dynamic_programming import solve_dp
from exact.ILP import solve_ilp


# ─────────────────────────────────────────────────────────────
# FAST PROFIT (vectorized incremental-style helper)
# ─────────────────────────────────────────────────────────────

def initial_profit(subset, q, P):
    """Compute full profit once (ONLY at initialization)."""
    profit = sum(q[i] for i in subset)
    subset_list = list(subset)
    for i in range(len(subset_list)):
        a = subset_list[i]
        for j in range(i + 1, len(subset_list)):
            b = subset_list[j]
            profit += P[a][b]
    return profit


def delta_add(item, subset, q, P):
    """Profit gain if we ADD item."""
    return q[item] + sum(P[item][i] for i in subset)


def delta_remove(item, subset, q, P):
    """Profit loss if we REMOVE item."""
    return q[item] + sum(P[item][i] for i in subset if i != item)


# ─────────────────────────────────────────────────────────────
# FAST REPAIR (same logic, no recomputation of profit)
# ─────────────────────────────────────────────────────────────

def repair(subset, weights, c, q, P):
    subset = set(subset)
    total_w = sum(weights[i] for i in subset)

    while total_w > c:
        worst = None
        worst_val = float("inf")

        for i in subset:
            val = delta_remove(i, subset, q, P)
            if val < worst_val:
                worst_val = val
                worst = i

        subset.remove(worst)
        total_w -= weights[worst]

    return subset


# ─────────────────────────────────────────────────────────────
# CHROMOSOME UTILITIES
# ─────────────────────────────────────────────────────────────

def to_subset(ch):
    return {i for i, v in enumerate(ch) if v == 1}


def to_chromosome(subset, n):
    return [1 if i in subset else 0 for i in range(n)]


# ─────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────

def greedy_solution(n, weights, c, q, P):
    scores = []
    for j in range(n):
        inter = sum(P[j][i] for i in range(n) if i != j)
        scores.append(((q[j] + inter) / weights[j], j))

    scores.sort(reverse=True)

    subset = set()
    w = 0
    for _, j in scores:
        if w + weights[j] <= c:
            subset.add(j)
            w += weights[j]

    return subset


def random_solution(n, weights, c, rng):
    items = list(range(n))
    rng.shuffle(items)

    subset = set()
    w = 0
    for i in items:
        if w + weights[i] <= c:
            subset.add(i)
            w += weights[i]

    return subset


# ─────────────────────────────────────────────────────────────
# FITNESS CACHE (CRITICAL OPTIMIZATION)
# ─────────────────────────────────────────────────────────────

def build_fitness(subset, q, P):
    return initial_profit(subset, q, P)


# ─────────────────────────────────────────────────────────────
# LOCAL SEARCH (FAST DELTA VERSION)
# ─────────────────────────────────────────────────────────────

def local_search(subset, weights, c, q, P, max_swaps=5):
    subset = set(subset)
    w = sum(weights[i] for i in subset)

    current = initial_profit(subset, q, P)

    for _ in range(max_swaps):

        improved = False
        selected = list(subset)
        unselected = [i for i in range(len(weights)) if i not in subset]

        for out in selected:
            for inn in unselected:

                new_w = w - weights[out] + weights[inn]
                if new_w > c:
                    continue

                # delta evaluation (NO full recompute)
                gain = (
                    q[inn] - q[out]
                    + sum(P[inn][i] for i in subset if i != out)
                    - sum(P[out][i] for i in subset if i != out)
                )

                if gain > 0:
                    subset.remove(out)
                    subset.add(inn)
                    w = new_w
                    current += gain
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return subset


# ─────────────────────────────────────────────────────────────
# GA MAIN
# ─────────────────────────────────────────────────────────────

def genetic_algorithm(n, weights, q, P, c,
                      pop_size=40,
                      max_generations=150,
                      mutation_rate=None,
                      tournament_size=3,
                      elite_size=2,
                      seed=42):

    rng = random.Random(seed)

    if mutation_rate is None:
        mutation_rate = 1.0 / n

    # init population
    population = []

    population.append(to_chromosome(greedy_solution(n, weights, c, q, P), n))

    while len(population) < pop_size:
        population.append(to_chromosome(random_solution(n, weights, c, rng), n))

    # initial fitness (CACHED)
    fitness = []
    for ch in population:
        fitness.append(build_fitness(to_subset(ch), q, P))

    best_idx = max(range(pop_size), key=lambda i: fitness[i])
    best = population[best_idx][:]
    best_val = fitness[best_idx]

    no_improve = 0

    # ── generations ─────────────────────────────
    for gen in range(max_generations):

        new_pop = []

        # elitism
        elite_idx = sorted(range(pop_size), key=lambda i: fitness[i], reverse=True)[:elite_size]
        for i in elite_idx:
            new_pop.append(population[i][:])

        # fill rest
        while len(new_pop) < pop_size:

            def select():
                cand = rng.sample(range(pop_size), tournament_size)
                return population[max(cand, key=lambda i: fitness[i])]

            p1 = select()
            p2 = select()

            # crossover
            child = [
                p1[i] if rng.random() < 0.5 else p2[i]
                for i in range(n)
            ]

            # mutation
            child = [
                1 - child[i] if rng.random() < mutation_rate else child[i]
                for i in range(n)
            ]

            # repair
            subset = repair(to_subset(child), weights, c, q, P)

            # local search (FAST VERSION)
            subset = local_search(subset, weights, c, q, P)

            new_pop.append(to_chromosome(subset, n))

        population = new_pop

        # recompute fitness (ONLY ONCE PER GENERATION)
        fitness = [
            build_fitness(to_subset(ch), q, P)
            for ch in population
        ]

        idx = max(range(pop_size), key=lambda i: fitness[i])

        if fitness[idx] > best_val:
            best_val = fitness[idx]
            best = population[idx][:]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 40:
            break

    return best_val, sorted(to_subset(best)), gen + 1


# ─────────────────────────────────────────────────────────────
# TESTER (compatible with generator.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_files = [
        f"data/instances/n{n}_d{int(d*100)}_s{s}.json"
        for n in [10, 15, 20, 30, 50, 100]
        for d in [0.25, 0.50, 0.75, 1.0]
        for s in range(10)
    ]

    print(f"{'Instance':<30} {'GA':>8} {'OPT':>8} {'Gap%':>8} {'Gen':>6} {'Time(s)':>10}")
    print("-" * 80)

    for path in test_files:

        if not os.path.exists(path):
            continue

        inst = load_instance(path)
        n, weights, q, P, c = inst["n"], inst["weights"], inst["q"], inst["P"], inst["c"]

        start = time.time()
        ga_profit, ga_subset, gens = genetic_algorithm(
            n, weights, q, P, c,
            seed=inst.get("seed", 42)
        )
        t = time.time() - start

        if n <= 20:
            opt, _ = solve_dp(n, weights, q, P, c)
            gap = (opt - ga_profit) / opt * 100
            opt_str = str(opt)
        else:
            gap = 0
            opt_str = "N/A"

        print(f"{os.path.basename(path):<30} {ga_profit:>8} {opt_str:>8} {gap:>8.2f} {gens:>6} {t:>10.3f}")