# Report-Supervised Coronary Plaque Classification on Computed Tomography Angiography with Foundation Models: From Whole-Scan to Segment-Level Prediction

Private thesis-code review repository for Allister Lim, Victorian Heart Institute.
The benchmark predicts calcified and non-calcified plaque from CTCA using clinical
report-derived labels, at four anatomical scales: global (2 outputs), LCA/RCA
(4), four coronary families (8), and Detailed14 (28).

## Review status

This is a portable export of the scripts used in the research, **not a claim of a
new end-to-end reproduction**. Historical training logic is retained, while
private scan IDs, machine-specific paths and private preparation gates have been
replaced with documented interfaces. See [provenance](docs/PROVENANCE.json).
Clinical data, individual predictions, annotations and checkpoint weights are
not distributed. Do not upload them to this repository.

Included: ResNet18-GroupNorm, CORA and image-only Cardiac-CLIP fine-tuning;
task-matched coronary pooling; union-only synthetic adaptation; validation-derived
thresholds, confusion counts, score distributions and patient-cluster AUROC CIs.
Twenty JSON configurations map to completed training runs. Do not mistake these for
twenty new runs. The repository does not submit jobs to any external service.

## Installation

Training requires a CUDA-capable GPU. Create a separate Python 3.11.7
environment, then install the recorded PyTorch build and dependencies:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/training.txt
```

The later statistical analysis used a separate Python 3.12.7 environment; see
`requirements/analysis.txt`. It does not need PyTorch. nnU-Net 2.8.0 was used in a
separate segmentation environment, not installed into the historical MONAI
training environment. These pins document the environments rather than promise
support for every operating system or CUDA driver.

Obtain [CORA](https://github.com/Advanced-AI-in-Medicine-and-Physics-Lab/CORA)
and [Cardiac-CLIP](https://github.com/Sliver-g/Cardiac-CLIP) source and released
weights separately, subject to their terms. Source revisions and attribution
are in [THIRD_PARTY.md](THIRD_PARTY.md). Upstream implementations and model
weights are deliberately not vendored.

## Inputs and scope

**Coronary artery segmentation masks are a prerequisite for coronary pooling
and synthetic adaptation.** Our experiments used masks generated from the raw CT
by an ImageCAS-trained nnU-Net coronary segmentor. Installing nnU-Net alone does
not provide these masks: obtain suitable coronary-model weights, run inference
on your scans, and prepare aligned mask inputs before running either extension.
Neither this repository nor the CORA classification checkpoint automatically
generates coronary masks.

| Experiment | Coronary masks required |
| --- | --- |
| Reference-model classification without pooling | None at downstream training/inference |
| Whole-coronary pooling | Prepared LCA/RCA mask cache, combined into a union by the runner |
| LCA/RCA-specific pooling | Separate LCA and RCA masks |
| Four-family or Detailed14 pooling | Finer region-labelled masks; our runs used the local 16-class segmentor, not LCA/RCA masks alone |
| Union-only synthetic adaptation | Native-CT-aligned coronary masks; ImageCAS LCA/RCA labels are combined as foreground |

Pooling needs aligned masks at both training and inference. Synthetic adaptation
needs masks during patch preparation and lesion insertion; subsequent ordinary
report-classifier fine-tuning does not need masks unless pooling is also used.
The masks describe **arteries, not plaque annotations**, and are not bundled.

Read [the input contract](docs/INPUTS.md) first. These runners consume prepared
caches, not DICOM. Preserving historical crop geometry matters: converting the
same HU crop into four windows is not proof that it equals an independently
prepared historical Drive crop. Physical mask alignment must be established
before using a mask cache. Do not infer orientation from overlap maximisation.

Place private data/checkpoints outside the repository, or in the ignored `data/`
and `checkpoints/` directories. The supplied contracts pin the historical cohort
hashes. For independently authorised data, create a new contract; do not call
its metrics a reproduction of the historical cohort.

## Run a classifier

```bash
python run_experiment.py \
  --config configs/cora_detailed14.json \
  --source /path/to/CORA \
  --labels /private/detailed14/training_manifest.csv \
  --data /private/cora_crops \
  --pretrained /private/cora_pretrained.pth \
  --out /private/runs/cora_detailed14 \
  --smoke-only
```

The launcher **prints the command without running it** by default. Add `--execute`
to perform the training-only smoke check. Remove `--smoke-only` to train, select
the validation-best checkpoint, and evaluate it on the frozen test partition.
Evaluation writes private predictions and metrics to the supplied output folder;
never commit individual prediction files. Use `--eval-only --checkpoint ...`
to evaluate an existing validation-selected checkpoint without retraining.

Baseline configs use `CT_HU` caches and need no pretrained weights; their required
`--source` may point to an empty directory. Cardiac-CLIP configs need the upstream
directory containing `clip_model.py`, the released checkpoint, and `CARDIAC_RAW`
caches. This is supervised visual-encoder adaptation, not CLIP contrastive training.

Pooling configs additionally require `--roi-data /private/aligned_rois`.
Synthetic configs refer to report classification **after** synthetic adaptation;
give `--pretrained` the fixed epoch-5 `best_transfer.pth`, not the original encoder.
See [reproduction notes](docs/REPRODUCIBILITY.md) before either extension.

## Tests and aggregate results

```bash
python -m unittest discover -s tests -v
```

CPU tests use fabricated labels/features only. Aggregate historical metrics are
in `results/verified_aggregate_results.csv`; no patient-level results are
included. The paper's historical four-family CORA reference and the exported
matched control are distinct training runs. The CSV records the matched control;
do not silently substitute it for the historical reference.

## Data access and rights

Clinical imaging/reports cannot be redistributed under the study's ethics terms.
The local 16-class segmentor and its manually annotated data are not released
here. Availability of a wrapper does not grant rights to private data, third-party
code or weights. No open-source licence is granted for this review snapshot;
publication and licensing require institutional review. This repository is for
research, not clinical deployment.
