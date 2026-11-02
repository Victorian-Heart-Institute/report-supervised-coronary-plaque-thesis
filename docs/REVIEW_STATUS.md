# Code review status — 2 November 2026

## Checked

- Explicit allowlist export: no clinical images, reports, annotations, private
  manifests, scan IDs, per-scan predictions or model weights bundled.
- Site-specific identifiers and machine-specific paths removed from runners.
- Twenty configurations mapped to existing aggregate test results.
- CPU tests for metrics/unknown masking, all-positive collapse, validation
  thresholds, occupancy pooling, empty-mask fallback, patient leakage, target
  ordering and paired patient-cluster bootstrap.
- Training, pooling and synthetic-adaptation command-line help imports.
- AST comparison confirms unchanged metric functions, validation-threshold
  selection, training/evaluation phase loops and data loaders in both exported
  classification runners relative to their recorded historical sources.
- Bibliographic/source attribution and fixed upstream revisions recorded.

## Not yet established

- Full GPU checkpoint/prediction equivalence after portability edits.
- Execution in a clean historical Python 3.11.7/CUDA 12.8 environment; local CPU
  tests instead used the available Python 3.12.7/PyTorch 2.10.0 environment.
- Reproduction of the historical four-family CORA reference.
- A public redistribution licence approved by the institution.
- Public redistribution rights for third-party source/weights; these are excluded.

## Not included in this first export

- Private report extraction/adjudication pipeline and decision logs.
- Exact source-DICOM-to-historical-Drive-crop recreation.
- Native segmentation inference and historical physical-grid QA pipeline.
- AutoPlaque training experiments, abandoned runs and clinical illustrations.

The exported runners require already reviewed manifests and prepared/aligned
inputs. These boundaries are also described in INPUTS.md and REPRODUCIBILITY.md.
The code is publicly accessible. Public availability does not establish the
uncompleted reproduction checks above or grant rights to the excluded assets.

Packaging/refactoring and test preparation were assisted by OpenAI Codex. Source
provenance is retained to support author review; authors remain responsible for
validation, attribution, institutional approval and any required AI disclosure.
