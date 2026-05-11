import time
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import load_instance
from exact.dynamic_programming import solve_dp
from exact.ILP import solve_ilp
from simulated_annealing import compute_profit


def compute_marginal_profit(item, subset, q, P):
    """
    Profit contribution of item given current subset.
    Used in repair operator to decide which item to remove.
    """
    return q[item] + sum(P[item][i] for i in subset if i != item)


def repair(chromosome, weights, c, q, P):
    """
    If solution exceeds capacity, remove items one by one
    in order of lowest marginal profit until feasible.
    Returns repaired chromosome as a set.
    """
    subset       = set(j for j in range(len(chromosome)) if chromosome[j] == 1)
    total_weight = sum(weights[j] for j in subset)

    while total_weight > c:
        worst_item   = None
        worst_profit = float('inf')

        for j in subset:
            marginal = compute_marginal_profit(j, subset, q, P)
            if marginal < worst_profit:
                worst_profit = marginal
                worst_item   = j

        subset.remove(worst_item)
        total_weight -= weights[worst_item]

    return subset


def subset_to_chromosome(subset, n):
    """Convert a set of selected items to a binary list."""
    return [1 if j in subset else 0 for j in range(n)]


def chromosome_to_subset(chromosome):
    """Convert a binary list to a set of selected items."""
    return set(j for j in range(len(chromosome)) if chromosome[j] == 1)


def random_feasible_solution(n, weights, c, q, P, rng):
    """
    Generate a random feasible solution.
    Shuffle items, add greedily until capacity is reached.
    """
    items = list(range(n))
    rng.shuffle(items)

    subset       = set()
    total_weight = 0

    for j in items:
        if total_weight + weights[j] <= c:
            subset.add(j)
            total_weight += weights[j]

    return subset_to_chromosome(subset, n)


def greedy_feasible_solution(n, weights, c, q, P):
    """
    Generate a greedy feasible solution sorted by profit/weight ratio.
    Used to seed the initial population with one good solution.
    """
    scores = []
    for j in range(n):
        total_interaction = sum(P[j][i] for i in range(n) if i != j)
        score = (q[j] + total_interaction) / weights[j]
        scores.append((score, j))

    scores.sort(reverse=True)

    subset       = set()
    total_weight = 0

    for _, j in scores:
        if total_weight + weights[j] <= c:
            subset.add(j)
            total_weight += weights[j]

    return subset_to_chromosome(subset, n)


def initialize_population(pop_size, n, weights, c, q, P, rng):
    """
    Build initial population:
    - 1 greedy solution (good starting point)
    - rest random feasible solutions (diversity)
    """
    population = []

    # one greedy individual
    population.append(greedy_feasible_solution(n, weights, c, q, P))

    # rest random
    while len(population) < pop_size:
        population.append(random_feasible_solution(n, weights, c, q, P, rng))

    return population


def fitness(chromosome, q, P):
    """
    Fitness = total profit of the solution.
    Since all solutions are kept feasible, no penalty needed.
    """
    subset = chromosome_to_subset(chromosome)
    return compute_profit(subset, q, P)


def tournament_selection(population, fitnesses, tournament_size, rng):
    """
    Pick tournament_size random individuals, return the best one.
    More efficient than roulette wheel and works well for QKP.
    """
    competitors = rng.sample(range(len(population)), tournament_size)
    best        = max(competitors, key=lambda i: fitnesses[i])
    return population[best]


def uniform_crossover(parent1, parent2, rng):
    """
    For each bit position, randomly inherit from parent1 or parent2.
    More disruptive than single-point but explores more combinations.
    """
    n = len(parent1)
    child = []
    for i in range(n):
        child.append(parent1[i] if rng.random() < 0.5 else parent2[i])
    return child


def mutate(chromosome, mutation_rate, rng):
    """
    Flip each bit independently with probability mutation_rate.
    Mutation introduces diversity and prevents premature convergence.
    """
    return [
        1 - chromosome[i] if rng.random() < mutation_rate else chromosome[i]
        for i in range(len(chromosome))
    ]


def local_search(chromosome, weights, c, q, P, max_swaps=10):
    """
    Simple local search: try swapping one selected item for one
    unselected item if it improves profit and stays feasible.
    Applied to each new offspring to intensify search.
    """
    subset       = chromosome_to_subset(chromosome)
    total_weight = sum(weights[j] for j in subset)
    current_profit = compute_profit(subset, q, P)

    improved = True
    swaps    = 0

    while improved and swaps < max_swaps:
        improved = False

        selected   = list(subset)
        unselected = [j for j in range(len(chromosome)) if j not in subset]

        for out in selected:
            for inn in unselected:
                new_weight = total_weight - weights[out] + weights[inn]
                if new_weight > c:
                    continue

                # compute new profit after swap
                new_subset = (subset - {out}) | {inn}
                new_profit = compute_profit(new_subset, q, P)

                if new_profit > current_profit:
                    subset         = new_subset
                    total_weight   = new_weight
                    current_profit = new_profit
                    improved       = True
                    swaps         += 1
                    break

            if improved:
                break

    return subset_to_chromosome(subset, len(chromosome))


def genetic_algorithm(n, weights, q, P, c,
                      pop_size        = 50,
                      max_generations = 200,
                      mutation_rate   = None,
                      tournament_size = 3,
                      elite_size      = 2,
                      seed            = 42):
    """
    Genetic Algorithm for QKP.

    Key design choices:
    - Tournament selection for parent choice
    - Uniform crossover for gene mixing
    - Repair operator to maintain feasibility
    - Local search on each offspring to intensify
    - Elitism to preserve best solutions across generations

    Parameters:
        pop_size        : number of individuals in population
        max_generations : number of generations to run
        mutation_rate   : probability of flipping each bit (default 1/n)
        tournament_size : number of competitors in tournament selection
        elite_size      : number of best individuals preserved each generation
        seed            : random seed for reproducibility
    """
    rng = random.Random(seed)

    if mutation_rate is None:
        mutation_rate = 1.0 / n   # standard choice: expected 1 flip per chromosome

    # ── initialization ────────────────────────────────────────────────────────
    population = initialize_population(pop_size, n, weights, c, q, P, rng)
    fitnesses  = [fitness(ch, q, P) for ch in population]

    best_idx    = max(range(pop_size), key=lambda i: fitnesses[i])
    best_chromosome = list(population[best_idx])
    best_profit     = fitnesses[best_idx]

    no_improve = 0

    # ── generational loop ─────────────────────────────────────────────────────
    for generation in range(max_generations):

        new_population = []

        # ── elitism: carry best individuals unchanged ─────────────────────────
        elite_indices = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True
        )[:elite_size]

        for idx in elite_indices:
            new_population.append(list(population[idx]))

        # ── fill rest of population ───────────────────────────────────────────
        while len(new_population) < pop_size:

            # selection
            parent1 = tournament_selection(
                population, fitnesses, tournament_size, rng
            )
            parent2 = tournament_selection(
                population, fitnesses, tournament_size, rng
            )

            # crossover
            child = uniform_crossover(parent1, parent2, rng)

            # mutation
            child = mutate(child, mutation_rate, rng)

            # repair — restore feasibility after crossover and mutation
            subset = repair(child, weights, c, q, P)
            child  = subset_to_chromosome(subset, n)

            # local search — intensify around this offspring
            child = local_search(child, weights, c, q, P)

            new_population.append(child)

        # ── update population and fitnesses ───────────────────────────────────
        population = new_population
        fitnesses  = [fitness(ch, q, P) for ch in population]

        # ── update best ───────────────────────────────────────────────────────
        gen_best_idx = max(range(pop_size), key=lambda i: fitnesses[i])

        if fitnesses[gen_best_idx] > best_profit:
            best_profit     = fitnesses[gen_best_idx]
            best_chromosome = list(population[gen_best_idx])
            no_improve      = 0
        else:
            no_improve += 1

        # early stopping — no improvement for 50 generations
        if no_improve >= 50:
            break

    best_subset = sorted(chromosome_to_subset(best_chromosome))
    return best_profit, best_subset, generation + 1


# ── run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":

    test_files = [
        f"data/instances/n{n}_d{int(d*100)}_s{s}.json"
        for n in [10, 15, 20, 30, 50, 100]
        for d in [0.25, 0.50, 0.75, 1.0]
        for s in range(10)
    ]

    print(f"{'Instance':<30} {'GA':>8} {'OPT':>8} "
          f"{'Gap%':>8} {'Gen':>6} {'Time(s)':>10}")
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

        # run GA
        start      = time.time()
        ga_profit, ga_subset, generations = genetic_algorithm(
            n, weights, q, P, c
        )
        elapsed    = time.time() - start

        # ground truth
        if n <= 20:
            opt, _  = solve_dp(n, weights, q, P, c)
            gap     = (opt - ga_profit) / opt * 100 if opt > 0 else 0.0
            opt_str = str(opt)
        elif n <= 60:
            opt, _, status, _ = solve_ilp(n, weights, q, P, c, time_limit=60)
            gap     = (opt - ga_profit) / opt * 100 if opt > 0 else 0.0
            opt_str = str(opt)
        else:
            gap     = 0.0
            opt_str = "N/A"

        name = os.path.basename(path)
        print(f"{name:<30} {ga_profit:>8} {opt_str:>8} "
              f"{gap:>8.2f} {generations:>6} {elapsed:>10.4f}")