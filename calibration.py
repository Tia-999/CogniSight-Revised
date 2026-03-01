import numpy as np

def calibrate_threshold(scores, labels, target_fpr=0.05):
    """Find threshold for target FPR on negatives."""
    arr = np.array(scores)
    labs = np.array(labels)

    negs = arr[labs == 0]

    # take threshold at (1 - target_fpr) quantile of negatives
    thr = np.percentile(negs, 100 * (1 - target_fpr))

    # IMPORTANT: flip direction because lower score = more likely positive
    preds = (arr <= thr).astype(int)

    tpr = (preds[labs == 1].sum() / max(1, (labs == 1).sum()))
    return float(thr), float(tpr)

