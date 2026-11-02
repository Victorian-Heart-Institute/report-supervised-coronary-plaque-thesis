"""Patient-cluster percentile AUROC intervals and optional paired differences."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bootstrap_core import weighted_aucs


def calculate(frame, targets, repeats=20000, seed=20261030, paired=None):
    if not frame.scan_id.is_unique or frame.patient_id.isna().any():
        raise ValueError("Unique scans and known patient groups required")
    patients, inverse = np.unique(frame.patient_id, return_inverse=True)
    weights = (
        np.random.default_rng(seed)
        .multinomial(
            len(patients), np.full(len(patients), 1 / len(patients)), size=repeats
        )
        .astype(np.int32)
    )
    point = weighted_aucs(
        frame, targets, inverse, np.ones((1, len(patients)), dtype=np.int32)
    )[0]
    samples = weighted_aucs(frame, targets, inverse, weights)
    valid = np.isfinite(samples).all(axis=1)
    if paired is not None:
        if not paired.scan_id.is_unique or set(paired.scan_id) != set(frame.scan_id):
            raise ValueError("Paired predictions must contain exactly the same scans")
        paired = paired.set_index("scan_id").loc[frame.scan_id].reset_index()
        for name in ["patient_id", *targets]:
            if (
                not frame[name]
                .reset_index(drop=True)
                .equals(paired[name].reset_index(drop=True))
            ):
                raise ValueError("Paired patient groups or labels differ: " + name)
        other_point = weighted_aucs(
            paired, targets, inverse, np.ones((1, len(patients)), dtype=np.int32)
        )[0]
        other_samples = weighted_aucs(paired, targets, inverse, weights)
        valid &= np.isfinite(other_samples).all(axis=1)
        samples = other_samples - samples
        point = other_point - point
    if not valid.any():
        raise ValueError("No resample supports the entire fixed target set")
    low, high = np.quantile(samples[valid].mean(axis=1), [0.025, 0.975])
    return {
        "point": float(point.mean()),
        "ci95_lower": float(low),
        "ci95_upper": float(high),
        "paired_direction": "paired minus first" if paired is not None else None,
        "attempted_replicates": repeats,
        "valid_replicates": int(valid.sum()),
        "excluded_one_class_replicates": int((~valid).sum()),
        "seed": seed,
        "scope": "Fixed checkpoints and test cohort; not retraining or model-selection uncertainty",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--paired", type=Path)
    p.add_argument("--targets", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=20000)
    a = p.parse_args()
    load = lambda path: pd.read_csv(path, dtype={"scan_id": str, "patient_id": str})
    result = calculate(
        load(a.predictions),
        a.targets.split(","),
        a.repeats,
        paired=load(a.paired) if a.paired else None,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
