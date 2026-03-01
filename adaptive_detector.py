import numpy as np

def adaptive_min_prob(token_log_probs, method='zscore', **kwargs):
    lp = np.array(token_log_probs)
    if method == 'zscore':
        z_thresh = kwargs.get('z_thresh', 1.0)
        mu, sigma = lp.mean(), lp.std(ddof=0)
        if sigma == 0:
            return mu, []
        z = (lp - mu) / sigma
        idx = np.where(z <= -abs(z_thresh))[0]
    elif method == 'iqr':
        alpha = kwargs.get('alpha', 1.0)
        q1, q3 = np.percentile(lp, 25), np.percentile(lp, 75)
        iqr = q3 - q1
        threshold = q1 - alpha * iqr
        idx = np.where(lp <= threshold)[0]
    else:
        raise ValueError("Unknown method")

    if len(idx) == 0:
        from min_k_prob import min_k_percent_prob
        return min_k_percent_prob(lp, k_percent=0.5), []

    # NEW: Sum z-scores of outliers for stronger separation
    score = float(np.sum(lp[idx]) / len(lp))  # normalize by text length
    return score, list(idx)
