# Reproduction boundaries and additional adaptation

## Historical training versus portable export

The source-to-export hashes and changes are in `PROVENANCE.json`. Portability
edits remove site-specific data contracts, not the patient-disjointness and
binary/unknown checks. Training scripts retain seed 20260804, batch size 1,
accumulation 4, validation-best selection, 40-epoch cap and patience 8. AdamW
and weighted, unknown-masked BCE remain as in the original runners. Per-run
float16/bfloat16 choices are recorded in the JSON configurations.

The classification loop retains the historical division by four for a partial
last accumulation group. Synthetic adaptation separately scales its final group
by actual group size. These are intentionally not silently standardised. Metrics,
split composition and checkpoint semantics matter more than a generic training
template. Exact GPU checkpoint/prediction equivalence has **not** yet been tested
after export. Historical test results were examined during development and are
not a new external test.

The classifier source contains smoke diagnostics and automatic resume from
`latest.pth`. Use a new output directory for a new run. Synthetic adaptation also
resumes local state. All paths/output metadata can contain private identifiers;
keep outputs outside Git.

## Synthetic union-only adaptation

**Required first:** precompute coronary artery masks from raw CT using an
appropriate model, such as the ImageCAS-trained nnU-Net used in our experiments.
Provide native-CT-aligned masks through the directory below or `--mask-dir`.
The adaptation script does not run coronary segmentation, and installing
nnU-Net does not replace generating the masks. The masks guide coronary-centred
patch sampling and constrain synthetic lesion insertion; plaque segmentations
are not required.

`training/adapt_synthetic.py` is the original train/prepare workflow with its
atomic checkpoint helper relocated. It retains the historical requirement for
2,035 report-training scans, uses no report-validation/test patients, and prepares
two coronary-centred 96³ patches per scan. Private root layout:

```text
root/
  nifti_contrast/<scan_id>.nii.gz
  masks/imagecas_lca_rca/combined/<scan_id>.nii.gz
```

Use `--mask-dir /private/native_masks` to override the default mask directory.

Preparation:

```bash
python training/adapt_synthetic.py --mode prepare \
  --root /private/root --manifest /private/training_manifest.csv \
  --data /private/synthetic_patches
```

Adaptation (requires CUDA and separately obtained upstream synthesis source):

```bash
python training/adapt_synthetic.py --mode train \
  --manifest /private/training_manifest.csv --data /private/synthetic_patches \
  --out /private/synthetic_adaptation --source /path/to/CORA \
  --pretrained /private/cora_pretrained.pth \
  --synthesis-source /path/to/source_containing_LesionSynthesizer.py --epochs 5
```

Only the upstream `LesionSynthesizer` class is loaded from that source; it is not
vendored. Synthesis varies HU appearance but trains one binary inserted-lesion
union, independently of report composition/location. It does not supervise all
natural plaque. Twenty-five percent of presentations have no insertion.
The exported `best_transfer.pth` is the **fixed final epoch**, not a validation
winner. The transferred encoder then undergoes ordinary report classification
using a `cora_synthetic_*` config. Unavailable preparation is explicitly counted;
the original workflow's 90% preparation gate remains. Do not assert complete
coverage without inspecting the generated coverage file.

## Statistical evaluation

`analysis/bootstrap.py` samples patients with replacement, retaining all their
scans and using a fixed set of targets. A whole replicate is omitted when any
target lacks both classes. Default: 20,000 attempts, seed 20261030, percentile
95% CI. With `--paired`, the difference is **second file minus first file** on
the same patient resamples. This measures fixed-checkpoint test-cohort
uncertainty, not repeated-training stability or uncertainty from label extraction.

```bash
python analysis/bootstrap.py --predictions /private/test_predictions.csv \
  --targets calcified,non_calcified --out /private/bootstrap_summary.json
```

## Main-table four-family exception

The paper's historical CORA four-family reference reports macro AUROC/AP of
0.759/0.601. The separate matched control reports 0.603/0.421 and is retained in
the aggregate CSV. These are distinct training runs; the supplied configuration
maps to the matched control, not a recreation of the historical checkpoint.
Do not substitute one for the other. A full legacy recipe and checkpoint audit
would be needed to reproduce that historical training result.
