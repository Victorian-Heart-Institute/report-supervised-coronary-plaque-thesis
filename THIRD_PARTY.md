# Attribution and redistribution boundary

- **CORA**: Advanced AI in Medicine and Physics Lab; Hao et al., arXiv:2603.24847v1.
  Source used: `2b76639972b3cc84a4fc37003895e78ebe477a26` at
  https://github.com/Advanced-AI-in-Medicine-and-Physics-Lab/CORA.
  Model architecture and lesion synthesis are imported from a separately obtained
  source checkout. No upstream source or weights are redistributed here.
- **Cardiac-CLIP**: Hu et al., arXiv:2507.22024v1;
  https://github.com/Sliver-g/Cardiac-CLIP, recorded source revision
  `f27501fba0140b065232956c824dbcb581b6b1f7`.
  `training/cardiac_image_model.py` is the project-specific image-tower adapter;
  it imports the separately obtained upstream `clip_model.py`.
- **MONAI** supplies the 3D ResNet18 implementation; PyTorch supplies training
  primitives; ImageCAS-trained nnU-Net supplies externally prepared anatomical
  guidance. Their respective code, dataset and weight terms continue to apply.
- **Local 16-class coronary segmentor**: developed separately with 42 annotated
  scans and applied to the larger cohort. Its weights and annotation data are not
  bundled. Grouping its outputs is not a reimplementation of its training.

No top-level licence file was present in the locally inspected CORA or
Cardiac-CLIP checkouts. This is not permission to redistribute their code/weights.
Review upstream and institutional terms before public release. This repository
does not relicense external projects or claim their architectures as original.

Project-specific contributions packaged here are downstream task adaptation,
masked report-supervised training, input interfaces, mask-guided aggregation,
the additional synthetic-adaptation workflow, evaluation and packaging. Full
automated report extraction and private manual adjudication records are not
included in this first review export.
