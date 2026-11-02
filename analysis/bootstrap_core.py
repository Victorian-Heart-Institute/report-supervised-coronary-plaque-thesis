"""Patient-cluster AUROC helper from the thesis bootstrap analysis."""

import numpy as np
import pandas as pd


def weighted_aucs(frame, targets, inverse, multiplicities):
    result = np.full((len(multiplicities), len(targets)), np.nan)
    for j, target in enumerate(targets):
        y = pd.to_numeric(frame[target], errors="coerce").to_numpy()
        scores = frame["p_" + target].to_numpy(float)
        keep = np.isin(y, [0, 1]) & np.isfinite(scores)
        order = np.argsort(scores[keep], kind="stable")
        labels = y[keep][order].astype(np.int64)
        sorted_scores = scores[keep][order]
        patients = inverse[keep][order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
        for first in range(0, len(multiplicities), 1000):
            last = min(first + 1000, len(multiplicities))
            weights = multiplicities[first:last, patients]
            positives = np.add.reduceat(weights * labels, starts, axis=1)
            negatives = np.add.reduceat(weights * (1 - labels), starts, axis=1)
            before = np.cumsum(negatives, axis=1) - negatives
            numerator = np.sum(positives * (before + 0.5 * negatives), axis=1)
            denominator = positives.sum(axis=1) * negatives.sum(axis=1)
            result[first:last, j] = np.divide(
                numerator,
                denominator,
                out=np.full(last - first, np.nan),
                where=denominator > 0,
            )
    return result
