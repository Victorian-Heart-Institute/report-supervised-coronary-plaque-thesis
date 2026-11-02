# Private input contracts

## Manifest

CSV columns: `scan_id`, `patient_id`, `split`, `cohort`, followed
by the ordered targets in `docs/TARGETS.json`. `split` is `train`, `val` or `test`.
Targets are 0, 1, or blank/NaN for unknown. Optional `<target>_known` columns must
equal the target's non-missing indicator. Patients cannot cross partitions.
Use non-identifying study codes in any shared examples; do not publish a clinical
manifest. The historical hashes in `contracts/` verify files held privately.

The supplied cohort has 2,035/260/246 scans and 1,492/185/184 patients. The
training runner excludes all-unknown rows from supervised training, not from the
frozen cohort. Report silence, visibility, mixed/unspecified composition and
broad-family propagation follow the thesis conventions. Mining or independently
adjudicating new reports is **not implemented** in this first export; the runner
requires already reviewed labels. Unknown and negative are not interchangeable.

## CT caches

Each file is `<scan_id>.npz`, loaded without pickle:

| Mode | Key | Stored shape | Preparation |
| --- | --- | --- | --- |
| `raw1` | `CT_HU` | 224 × 224 × 224 | Unscaled HU; loader clips [-1000,2000], maps to [0,1], adds one channel |
| `cora4` | `STEP4_CROP` | 4 × 224 × 224 × 224 | Four already scaled HU windows |
| `cardiac_raw` | `CARDIAC_RAW` | 1 × 128 × 144 × 144 | Linear resize of HU, then volume min–max scaling, no fixed HU clipping |

CORA channel ranges, in order: [-170,-30], [-150,250], [0,700], [-500,1500] HU.
The intended 224³ crop is 0.5-mm isotropic, centred on the volume midpoint.
Historical raw and Drive crops were separately prepared. Exact historical
Drive-cache reconstruction and source DICOM conversion are not bundled.
`preprocessing/convert_inputs.py` implements intensity/resize transforms on an
existing HU cache; it does not establish matching physical geometry.

## Pooling masks

Generate coronary artery segmentations before preparing pooling inputs. The
study's coarser masks came from an **ImageCAS-trained nnU-Net** applied to raw CT
NIfTIs. nnU-Net is a segmentation framework, not a built-in coronary model; you
also need the appropriate pretrained model and its inference preprocessing.
Another provider is not automatically interchangeable: match its label meanings
and verify physical alignment first. The current whole-coronary runner expects
the two-channel LCA/RCA cache and computes their union internally; it does not
directly load an arbitrary binary NIfTI. Territory pooling requires separate
LCA/RCA identities. Family/segment pooling requires finer regional identities
that a binary or two-territory mask cannot supply.

Mask cache files use `ROI` of shape (2,224,224,224) for LCA/RCA masks, or
`SEGMENT_ROI` of shape (224,224,224) for 0–16 labels. Every NPZ contains
`margin_mm=3`. CT stays intact; mask channels are stripped before the encoder.

`PREPARATION_OK.json` must contain absolute `roi_root`, `margin_mm=3`, and
`training_gate_passed=true`. Regional caches also use `provider=liam_scct16`
(the legacy provider identifier), `four_family_groups` and `detailed14_groups`.
`qa.csv` contains all scan IDs and `status`, plus JSON `territory_valid` (two
booleans) or `region_valid` (keys "1"–"16"). Invalid/absent regions must have
zero maps, invoking global-mean fallback, not fabricated masks or scan exclusion.

This export consumes the **already aligned and QA-recorded masks** from the
study. It does not claim an unvalidated physical alignment implementation or
rerun segmentation. Preserve the reference CT crop's origin, direction, spacing
and size; nearest-neighbour resampling and 3-mm physical dilation must reproduce
that grid. The source preparation logic and grouping rationale are described in
the thesis; no automatic orientation optimisation is part of training.

## Synthetic-adaptation masks

Before `training/adapt_synthetic.py --mode prepare`, provide one native-CT-aligned
coronary mask per training scan, such as the ImageCAS-trained nnU-Net output.
Use `<scan_id>.nii.gz` filenames and `--mask-dir` for the containing directory.
The synthetic workflow uses `mask > 0` as a coronary union; it does not require
the LCA/RCA identities for this objective. These are coronary-artery masks, not
AutoPlaque lesion masks. Without usable coronary foreground, patches cannot be
prepared. Running `--mode train` instead requires the already prepared `hu`/`ca`
patches from that step. No segmentation model is run automatically by either mode.
