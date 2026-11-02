"""Unchanged metric functions extracted from the historical classifier runner."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def summarize(frame, targets, thresholds=None):
    rows = []
    for target in targets:
        mask = frame[target].notna()
        y = frame.loc[mask, target].to_numpy(dtype=int)
        p = frame.loc[mask, "p_" + target].to_numpy(dtype=float)
        threshold = float((thresholds or {}).get(target, 0.5))
        positive, negative = (p[y == 1], p[y == 0])
        prediction = p >= threshold
        tp, tn = (
            int(((y == 1) & prediction).sum()),
            int(((y == 0) & ~prediction).sum()),
        )
        fp, fn = (
            int(((y == 0) & prediction).sum()),
            int(((y == 1) & ~prediction).sum()),
        )
        sensitivity = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        both = len(positive) > 0 and len(negative) > 0
        rows.append(
            dict(
                target=target,
                known=len(y),
                unknown=int((~mask).sum()),
                positive=len(positive),
                negative=len(negative),
                threshold=threshold,
                TP=tp,
                TN=tn,
                FP=fp,
                FN=fn,
                AUROC=float(roc_auc_score(y, p)) if both else None,
                AP=float(average_precision_score(y, p)) if len(positive) else None,
                prevalence=float(y.mean()) if len(y) else None,
                accuracy=(tp + tn) / len(y) if len(y) else None,
                sensitivity=sensitivity,
                specificity=specificity,
                precision=tp / (tp + fp) if tp + fp else None,
                balanced_accuracy=(sensitivity + specificity) / 2 if both else None,
                median_positive=float(np.median(positive)) if len(positive) else None,
                median_negative=float(np.median(negative)) if len(negative) else None,
                delta_median=(
                    float(np.median(positive) - np.median(negative)) if both else None
                ),
                positive_q25=(
                    float(np.quantile(positive, 0.25)) if len(positive) else None
                ),
                positive_q75=(
                    float(np.quantile(positive, 0.75)) if len(positive) else None
                ),
                negative_q25=(
                    float(np.quantile(negative, 0.25)) if len(negative) else None
                ),
                negative_q75=(
                    float(np.quantile(negative, 0.75)) if len(negative) else None
                ),
            )
        )
    return pd.DataFrame(rows)


def validation_thresholds(frame, targets):
    thresholds = {}
    for target in targets:
        known = frame[frame[target].notna()]
        if known[target].nunique() < 2:
            thresholds[target] = 0.5
            continue
        fpr, tpr, candidates = roc_curve(known[target], known["p_" + target])
        valid = np.isfinite(candidates) & (candidates >= 0) & (candidates <= 1)
        indices = np.flatnonzero(valid)
        assert len(indices)
        best = indices[np.argmax((tpr[indices] + 1 - fpr[indices]) / 2)]
        thresholds[target] = float(candidates[best])
    return thresholds
