import numpy as np
import json

def generate_qkp_instance(n, density, seed=None):
    """
    n       : number of items
    density : probability of nonzero pair profit (0 to 1)
    """
    rng = np.random.default_rng(seed)
    
    # individual weights and profits
    weights = rng.integers(1, 51, size=n)        # uniform [1, 50]
    q = rng.integers(1, 101, size=n)             # uniform [1, 100]
    
    # pair profits (symmetric matrix)
    P = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < density:
                profit = rng.integers(1, 101)
                P[i][j] = profit
                P[j][i] = profit
    
    # capacity
    c = int(rng.integers(50, sum(weights)))
    
    return {
        "n": n,
        "density": density,
        "weights": weights.tolist(),
        "q": q.tolist(),
        "P": P.tolist(),
        "c": int(c)
    }

def save_instance(instance, path):
    with open(path, "w") as f:
        json.dump(instance, f)