
import numpy as np

def min_k_percent_prob(token_log_probs, k_percent= 0.1, verbose=True):
    arr = np.asarray(token_log_probs).ravel()
    n = arr.size
    if n == 0:
        return float('-inf')

    frac = (k_percent / 100.0) if k_percent > 1 else float(k_percent)
    if frac <= 0:
        raise ValueError("k_percent must be > 0")
    k = int(np.ceil(n * frac))
    k = max(1, min(n, k))

    bottom_k = np.partition(arr, k - 1)[:k]
    if verbose:
        print("n =", n, "k_percent =", k_percent, "frac =", frac, "k =", k)
        print("bottom_k =", bottom_k, "mean =", float(np.mean(bottom_k)))
    return float(np.mean(bottom_k))
