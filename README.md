# Quadratic Knapsack Problem — Combinatorial Optimization Project

**Author:** Miled Trabelssi  
**Course:** Optimisation Combinatoire  
**Academic Year:** 2025–2026

---

## What is this project?

This project is a complete study of the **Quadratic Knapsack Problem (QKP)**, a classic
NP-hard combinatorial optimization problem. In the standard 0-1 Knapsack Problem, each
item has a weight and an individual profit, and we select items to maximize profit without
exceeding a capacity. The QKP extends this by adding **pairwise interaction profits**:
two items selected together can generate an additional bonus, making the problem
significantly harder to solve.

The goal of this project is to implement, compare, and analyze **6 different solving
methods** — 3 exact and 3 approximate — across hundreds of automatically generated
instances of varying size and difficulty.

---

## What does the code do?

The project is organized as a pipeline with 2 stages:

**1. Instance Generation** — `instance_generator.py`  
Generates a large set of QKP test instances with controlled parameters: number of items,
interaction density, capacity tightness, and random seed. Instances are saved as JSON
files in `data/instances/`.

**2. Benchmark** — `benchmark.py`  
Runs all 6 solving methods on every instance and records the result of each run: profit
found, optimality gap, runtime, feasibility, and more. Everything is saved to a CSV file
for analysis then it Reads the CSV and produces summary tables and plots comparing the methods across
instance sizes and difficulty classes.

---

## The 6 Methods

**Exact methods** — guarantee the optimal solution but only scale to small instances:
- `DP` — Dynamic Programming
- `B&B` / `B&B Improved` — Branch and Bound with an adaptive bounding strategy
- `ILP` — Integer Linear Programming solved via the CBC solver (PuLP)

**Metaheuristics** — scale to large instances, trade optimality for speed:
- `SA` — Simulated Annealing
- `GA` — Genetic Algorithm
- `ILS` — Iterated Local Search

For each instance, a **reference solution** is computed first (DP for n ≤ 20, ILP
otherwise) and used to measure the **optimality gap** of every other method.

---

## Project Structure

```text
QKP_project/
├── main.py                         # run everything with one command
├── instance_generator.py           # stage 1: generate instances
├── benchmark.py                    # stage 2: run all 6 methods
├── analyze_results.py              # stage 3: summaries and plots
├── requirements.txt
│
├── data/
│   └── instances/                  # generated .json instance files
│
├── exact/
│   ├── dynamic_programming.py
│   ├── branch_and_bound.py
│   ├── branch_and_bound_improved.py
│   └── ILP.py
│
├── approx/
│   ├── simulated_annealing.py
│   ├── genetic.py
│   └── ILS.py
│
└── results/
    ├── csv/                        # benchmark_results.csv
    └── plots/                      # all generated figures
```
## Requirements

Python 3.10+ is required (developed and tested on Python 3.13).

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

One command runs the entire pipeline — instance generation, benchmark, and analysis:

```bash
python main.py
```

> ⚠️ The benchmark step takes **15–30 minutes** depending on your machine, because
> exact methods (DP, ILP) are computationally expensive on larger instances.
> Grab a coffee.

---

## Running a Single Method

Each solver exposes a uniform interface and can be run standalone:

```python
from exact.dynamic_programming import solve_dp
from data.generator import load_instance

instance = load_instance("data/instances/qkp_n15_d0.5_c0.5_s0.json")
profit, subset = solve_dp(**instance)

print(f"Profit : {profit}")
print(f"Items  : {subset}")
```

The same pattern works for any of the 6 methods — just change the import and function name.

---

## Output Format

`results/csv/benchmark_results.csv` contains one row per (method, instance) pair:

| Column                   | Description                                                         |
|--------------------------|---------------------------------------------------------------------|
| instance                 | filename of the instance (e.g. `qkp_n10_d0.25_c0.5_s0.json`)      |
| path                     | full relative path to the instance file                             |
| n                        | number of items                                                     |
| density                  | interaction density (0.25 / 0.50 / 0.75 / 1.0)                    |
| capacity_ratio           | capacity ratio α (0.25 / 0.50 / 0.75)                             |
| interaction_type         | instance interaction structure (e.g. `balanced`)                   |
| difficulty               | structural class: `EASY`, `MEDIUM`, or `HARD`                      |
| method                   | solver used: `DP`, `B&B`, `B&B Improved`, `ILP`, `SA`, `GA`, `ILS` |
| reference_method         | method used as optimality reference (`DP` or `ILP`)                |
| reference_status         | solver status of the reference (`OPTIMAL`, `BKS`, etc.)            |
| reference_time_seconds   | wall-clock time of the reference solver (s)                         |
| reference_profit         | profit value of the reference solution                              |
| profit                   | profit found by this method                                         |
| gap_percent              | optimality gap (%) = (reference − profit) / reference × 100        |
| runtime_seconds          | wall-clock time of this method (s)                                  |
| subset_size              | number of items selected                                            |
| subset_weight            | total weight of selected items                                      |
| feasible                 | `True` if capacity constraint is satisfied                          |
| profit_verified          | `True` if recomputed profit matches reported profit                 |
| iterations               | generations (GA) or iterations (SA/ILS); empty for exact methods   |
| solver_status            | internal status string (`OPTIMAL`, `OK`, `ERROR`, etc.)            |
| better_than_reference    | `True` if this method outperformed the reference                    |
| solution                 | selected item indices, semicolon-separated (e.g. `0;3;7;12`)       |
| error                    | error message if the method crashed, empty otherwise                |

---

## Reproducibility

All instances are generated with fixed random seeds.  
All metaheuristics use the instance seed as their random seed.  
Running `benchmark.py` twice produces identical `results.csv` output.
