import numpy as np
import json
import os


def generate_qkp_instance(n, density, capacity_ratio, interaction_type="balanced", seed=None):
    rng = np.random.default_rng(seed)

    # ── weights and linear profits ─────────────────────────────
    weights = rng.integers(1, 51, size=n).tolist()
    q = rng.integers(1, 101, size=n).tolist()

    # ── quadratic profits regime control ───────────────────────
    if interaction_type == "weak":
        low, high = 1, 20
    elif interaction_type == "strong":
        low, high = 50, 200
    else:  # balanced
        low, high = 1, 100

    # ── exact density construction ─────────────────────────────
    P = [[0] * n for _ in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = int(density * len(pairs))

    selected = rng.choice(len(pairs), size=num_edges, replace=False)

    for idx in selected:
        i, j = pairs[idx]
        val = int(rng.integers(low, high))
        P[i][j] = val
        P[j][i] = val

    # ── controlled capacity (IMPORTANT FIX) ────────────────────
    expected_weight = 25 * n  # stable across instances
    c = int(capacity_ratio * expected_weight)

    return {
        "n": n,
        "density": density,
        "capacity_ratio": capacity_ratio,
        "interaction_type": interaction_type,
        "weights": weights,
        "q": q,
        "P": P,
        "c": c,
        "seed": seed
    }


def save_instance(instance, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(instance, f, indent=2)


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


# ── dataset generator ─────────────────────────────────────────
if __name__ == "__main__":

    sizes = [10, 20, 30, 50, 100]
    densities = [0.25, 0.5, 0.75, 1.0]
    capacity_ratios = [0.25, 0.5, 0.75]
    seeds = range(5)

    interaction_types = ["weak", "balanced", "strong"]

    os.makedirs("data/instances", exist_ok=True)

    count = 0

    for n in sizes:
        for d in densities:
            for c_ratio in capacity_ratios:
                for itype in interaction_types:
                    for seed in seeds:

                        inst = generate_qkp_instance(
                            n=n,
                            density=d,
                            capacity_ratio=c_ratio,
                            interaction_type=itype,
                            seed=seed
                        )

                        path = f"data/instances/n{n}_d{int(d*100)}_c{int(c_ratio*100)}_{itype}_s{seed}.json"
                        save_instance(inst, path)

                        count += 1

    print(f"Generated {count} controlled QKP instances.")