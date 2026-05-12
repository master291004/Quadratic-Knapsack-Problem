# Quadratic Knapsack Problem — Combinatorial Optimization Project

**Author:** Miled Trabelssi  
**Course:** Optimisation Combinatoire  
**Academic Year:** 2025–2026

---


## Requirements

Python 3.10+ is required (developed and tested on Python 3.13).

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

python main.py
---

## Running a Single Method

Each solver exposes a uniform `solve()` interface and can be run standalone:

```python
from exact.dp import solve
from instance_generator import generate_instance

instance = generate_instance(n=15, density=0.5, alpha=0.5, seed=42)
solution, profit, runtime = solve(**instance)

print(f"Profit : {profit}")
print(f"Runtime: {runtime:.4f}s")
print(f"Items  : {solution}")
```

The same call works for any of the 6 methods — just change the import.

---

## Output Format

`results/csv/benchmark_results.csv` contains one row per (method, instance) pair:

| Column                   | Description                                                        |
|--------------------------|--------------------------------------------------------------------|
| instance                 | filename of the instance (e.g. `qkp_n10_d0.25_c0.5_s0.json`)     |
| path                     | full relative path to the instance file                            |
| n                        | number of items                                                    |
| density                  | interaction density (0.25 / 0.50 / 0.75 / 1.0)                   |
| capacity_ratio           | capacity ratio α (0.25 / 0.50 / 0.75)                            |
| interaction_type         | instance interaction structure (e.g. `balanced`)                  |
| difficulty               | structural class: `EASY`, `MEDIUM`, or `HARD`                     |
| method                   | solver used: `DP`, `B&B`, `B&B Improved`, `ILP`, `SA`, `GA`, `ILS`|
| reference_method         | method used as optimality reference (`DP` or `ILP`)               |
| reference_status         | solver status of the reference (`OPTIMAL`, `BKS`, etc.)           |
| reference_time_seconds   | wall-clock time of the reference solver (s)                        |
| reference_profit         | profit value of the reference solution                             |
| profit                   | profit found by this method                                        |
| gap_percent              | optimality gap (%) = (reference − profit) / reference × 100       |
| runtime_seconds          | wall-clock time of this method (s)                                 |
| subset_size              | number of items selected                                           |
| subset_weight            | total weight of selected items                                     |
| feasible                 | `True` if capacity constraint is satisfied                         |
| profit_verified          | `True` if recomputed profit matches reported profit                |
| iterations               | generations (GA) or iterations (SA/ILS); empty for exact methods  |
| solver_status            | internal status string (`OPTIMAL`, `OK`, `ERROR`, etc.)           |
| better_than_reference    | `True` if this method outperformed the reference                   |
| solution                 | selected item indices, semicolon-separated (e.g. `0;3;7;12`)      |
| error                    | error message if the method crashed, empty otherwise               |
---

## Reproducibility

All instances are generated with fixed random seeds.  
All metaheuristics use the instance seed as their random seed.  
Running `benchmark.py` twice produces identical `results.csv` output.