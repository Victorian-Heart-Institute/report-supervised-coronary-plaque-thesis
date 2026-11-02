"""Portable manifest checks; clinical manifests are not distributed."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np


def validate_manifest(frame, labels_path, targets, contract_path=None):
    required = {"scan_id", "patient_id", "split", "cohort", *targets}
    if not required <= set(frame.columns):
        raise ValueError(
            "Missing columns: " + str(sorted(required - set(frame.columns)))
        )
    if frame[["scan_id", "patient_id", "split"]].isna().any().any():
        raise ValueError("Identifiers and partitions cannot be missing")
    if (
        not frame.scan_id.is_unique
        or frame.groupby("patient_id").split.nunique().max() != 1
    ):
        raise ValueError("Duplicate scans or patient-level leakage")
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError("Expected train/val/test partitions")
    if not frame.scan_id.map(
        lambda s: bool(re.fullmatch(r"[A-Za-z0-9_-]+", str(s)))
    ).all():
        raise ValueError("Scan IDs must be safe filename stems")
    for target in targets:
        if not frame[target].dropna().isin([0, 1]).all():
            raise ValueError("Non-binary target: " + target)
        known = target + "_known"
        if (
            known in frame
            and not frame[known].eq(frame[target].notna().astype(int)).all()
        ):
            raise ValueError("Known-label indicator disagrees with target: " + target)
    sha = hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()
    if contract_path:
        contract = json.loads(Path(contract_path).read_text())
        if contract["targets"] != targets or contract["manifest_sha256"] != sha:
            raise ValueError("Target order or manifest checksum mismatch")
        if (
            "expected_split_counts" in contract
            and frame.groupby("split").size().to_dict()
            != contract["expected_split_counts"]
        ):
            raise ValueError("Unexpected split counts")
        # The full-manifest checksum also pins ancillary metadata.
    return sha
