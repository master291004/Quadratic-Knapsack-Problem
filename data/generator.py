import numpy as np
import json
import os

def generate_qkp_instance(n, density, seed=None):
    rng = np.random.default_rng(seed)
    
    weights = rng.integers(1, 51, size=n).tolist()
    q = rng.integers(1, 101, size=n).tolist()
    
    P = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < density:
                profit = int(rng.integers(1, 101))
                P[i][j] = profit
                P[j][i] = profit
    
    c = int(rng.integers(50, sum(weights)))
    
    return {
        "n": n,
        "density": density,
        "weights": weights,
        "q": q,
        "P": P,
        "c": c
    }

def save_instance(instance, path):
    with open(path, "w") as f:
        json.dump(instance, f, indent=2)

def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


# ── run this block when you execute: python generator.py ──────────────────────
if __name__ == "__main__":
    sizes     = [10, 15, 20, 30, 50, 100]
    densities = [0.25, 0.50, 0.75, 1.0]
    seeds     = range(10)

    os.makedirs("data/instances", exist_ok=True)

    count = 0
    for n in sizes:
        for d in densities:
            for seed in seeds:
                inst = generate_qkp_instance(n, d, seed)
                path = f"data/instances/n{n}_d{int(d*100)}_s{seed}.json"
                save_instance(inst, path)
                count += 1

    print(f"Generated {count} instances in data/instances/")

# sanity check — print one instance
sample = load_instance("data/instances/n10_d50_s0.json")
print("\nSample instance:")
print(f"  n={sample['n']}, c={sample['c']}")
print(f"  weights: {sample['weights']}")
print(f"  sum weights: {sum(sample['weights'])}")
print(f"  capacity: {sample['c']}")
assert sample['c'] <= sum(sample['weights']), "capacity too large"
assert sample['c'] >= 50, "capacity too small"
print("  looks good!")