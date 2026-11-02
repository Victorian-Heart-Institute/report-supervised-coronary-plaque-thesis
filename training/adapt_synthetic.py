"""Training-patient-only CORA synthetic adaptation; no AutoPlaque dependency.

Prepare two fixed coronary-centred native-CT patches per report-training scan.
Insert fresh official-CORA synthetic lesions each epoch; use a fixed five-epoch
endpoint rather than selecting on report validation or test images.
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


def save_json(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def cohort(manifest):
    rows = list(csv.DictReader(manifest.open()))
    train = [r for r in rows if r["split"] == "train"]
    held = {r["patient_id"] for r in rows if r["split"] != "train"}
    assert not ({r["patient_id"] for r in train} & held), "Patient leakage"
    assert len(train) == 2035, "Unexpected frozen training cohort"
    return train


def prepare(a):
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    rows = cohort(a.manifest)
    (a.data / "patches").mkdir(parents=True, exist_ok=True)
    (a.data / "records").mkdir(exist_ok=True)
    for row in rows[a.shard :: a.shards]:
        sid = row["scan_id"]
        record = a.data / "records" / (sid + ".json")
        if record.exists():
            old = json.loads(record.read_text())
            if old["status"] == "prepared" and all(
                (a.data / "patches" / p).exists() for p in old["patches"]
            ):
                continue
        info = dict(scan_id=sid, patient_id=row["patient_id"], patches=[])
        try:
            image_path = a.root / "nifti_contrast" / (sid + ".nii.gz")
            mask_path = (a.mask_dir or a.root / "masks/imagecas_lca_rca/combined") / (
                sid + ".nii.gz"
            )
            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))
            coords = np.argwhere(sitk.GetArrayFromImage(mask) > 0)
            assert len(coords), "Empty coronary mask"
            rng = np.random.default_rng(
                int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
            )
            for index in range(2):
                zyx = coords[rng.integers(len(coords))]
                centre = mask.TransformIndexToPhysicalPoint(
                    tuple(int(v) for v in zyx[::-1])
                )
                ref = sitk.Image([96] * 3, sitk.sitkFloat32)
                ref.SetSpacing([0.5] * 3)
                ref.SetDirection(image.GetDirection())
                direction = np.array(image.GetDirection()).reshape(3, 3)
                ref.SetOrigin(
                    tuple(np.array(centre) - direction @ np.full(3, 47.5 * 0.5))
                )
                hu = sitk.GetArrayFromImage(
                    sitk.Resample(
                        image,
                        ref,
                        sitk.Transform(),
                        sitk.sitkLinear,
                        -1000,
                        sitk.sitkFloat32,
                    )
                )
                ca = (
                    sitk.GetArrayFromImage(
                        sitk.Resample(
                            mask,
                            ref,
                            sitk.Transform(),
                            sitk.sitkNearestNeighbor,
                            0,
                            sitk.sitkUInt8,
                        )
                    )
                    > 0
                )
                assert ca.any() and np.isfinite(hu).all()
                name = f"{sid}_{index}.npz"
                dest = a.data / "patches" / name
                with dest.with_suffix(".partial").open("wb") as f:
                    np.savez_compressed(
                        f, hu=hu.astype(np.float16), ca=ca.astype(np.uint8)
                    )
                os.replace(dest.with_suffix(".partial"), dest)
                info["patches"].append(name)
            info.update(status="prepared", image=str(image_path), mask=str(mask_path))
        except Exception as error:
            info.update(status="unavailable", error=str(error))
        save_json(record, info)
        print(sid, info["status"], flush=True)


def train(a):
    import torch
    from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt
    from torch.utils.data import Dataset, DataLoader
    from checkpointing import checkpoint

    rows = cohort(a.manifest)
    records = [
        json.loads((a.data / "records" / (r["scan_id"] + ".json")).read_text())
        for r in rows
    ]
    ready = [r for r in records if r["status"] == "prepared"]
    a.out.mkdir(parents=True, exist_ok=True)
    save_json(
        a.out / "coverage.json",
        dict(
            eligible=len(rows),
            prepared=len(ready),
            unavailable=[r for r in records if r["status"] != "prepared"],
        ),
    )
    assert len(ready) >= 0.9 * len(
        rows
    ), "Less than 90% preparation coverage: inspect coverage.json before proceeding"
    source = ast.parse(a.synthesis_source.read_text())
    source.body = [
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "LesionSynthesizer"
    ]
    ns = dict(
        np=np,
        random=random,
        gaussian_filter=gaussian_filter,
        binary_erosion=binary_erosion,
    )
    exec(compile(source, str(a.synthesis_source), "exec"), ns)
    synthesizer = ns["LesionSynthesizer"]()

    class Patches(Dataset):
        def __init__(self):
            self.paths = [a.data / "patches" / p for r in ready for p in r["patches"]]

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with np.load(self.paths[index], allow_pickle=False) as z:
                hu = z["hu"].astype(np.float32)
                ca = z["ca"].astype(bool)
            target = np.zeros_like(hu)
            if random.random() >= 0.25:
                augmented, _, lesion = synthesizer(hu, ca)
                permitted = distance_transform_edt(~ca, sampling=0.5) <= 3
                hu = np.where(permitted, augmented, hu)
                target = ((lesion > 0) & permitted).astype(np.float32)
            x = np.stack(
                [
                    np.clip((hu - (wc - ww / 2)) / ww, 0, 1)
                    for wc, ww in [(-100, 140), (50, 400), (350, 700), (500, 2000)]
                ]
            )
            return torch.from_numpy(x), torch.from_numpy(target[None])

    torch.manual_seed(20261015)
    random.seed(20261015)
    np.random.seed(20261015)
    torch.set_num_threads(4)
    sys.path.insert(0, str(a.source))
    from models.model import build_unet

    model = build_unet(num_input_channels=4, num_output_channels=1)
    original = torch.load(a.pretrained, map_location="cpu", weights_only=False)
    state = original.get("model_state_dict", original)
    model.load_state_dict(
        {k.removeprefix("unet."): v for k, v in state.items() if k.startswith("unet.")},
        strict=True,
    )
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    assert torch.cuda.is_bf16_supported()
    loader = DataLoader(
        Patches(), batch_size=1, shuffle=True, num_workers=2, pin_memory=True
    )
    start, history = 0, []
    if (a.out / "latest.pth").exists():
        last = torch.load(a.out / "latest.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        start, history = last["epoch"], last["history"]
        torch.set_rng_state(last["torch_rng"].cpu())
        random.setstate(last["python_rng"])
        np.random.set_state(last["numpy_rng"])
        torch.cuda.set_rng_state_all([s.cpu() for s in last["cuda_rng"]])
    save_json(
        a.out / "config.json",
        dict(
            experiment=a.out.name,
            report_manifest=str(a.manifest),
            manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
            synthesis_sha256=hashlib.sha256(
                a.synthesis_source.read_bytes()
            ).hexdigest(),
            source="native report-training CT + physically resampled ImageCAS union",
            prepared_scans=len(ready),
            eligible_scans=len(rows),
            patches_per_scan=2,
            patch_shape=[4, 96, 96, 96],
            spacing_mm=0.5,
            epochs=a.epochs,
            selection="fixed final epoch; no report val/test images",
            init="official CORA full UNet",
            loss="focal(alpha=.25,gamma=4)+Tversky(alpha=.1,beta=.9)",
            learning_rate=1e-4,
            batch=1,
            accumulation=4,
            limitation="Insertion-only targets: natural plaque is not annotated and can be treated as background; two fixed coronary centres per scan, fresh synthetic lesions per epoch. No AP masks used.",
        ),
    )
    for epoch in range(start, a.epochs):
        before = time.monotonic()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for index, (x, y) in enumerate(loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x)
                z = z[0] if isinstance(z, (tuple, list)) else z
                assert z.shape == y.shape
                prob = z.float().sigmoid()
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    z.float(), y, reduction="none"
                )
                focal = (0.25 * (1 - torch.exp(-bce)).pow(4) * bce).mean()
                tp = (prob * y).sum()
                fp = (prob * (1 - y)).sum()
                fn = ((1 - prob) * y).sum()
                loss = focal + 1 - (tp + 1e-5) / (tp + 0.1 * fp + 0.9 * fn + 1e-5)
            assert torch.isfinite(loss), "Nonfinite training loss"
            # Correctly scale a partial final accumulation group.
            group_size = min(4, len(loader) - (index // 4) * 4)
            (loss / group_size).backward()
            if index == 0:
                assert any(
                    p.grad is not None
                    and torch.isfinite(p.grad).all()
                    and p.grad.abs().max() > 0
                    for p in model.encoder.parameters()
                )
            if (index + 1) % 4 == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if index % 100 == 0:
                print(
                    "epoch",
                    epoch + 1,
                    "batch",
                    index + 1,
                    "/",
                    len(loader),
                    "loss",
                    losses[-1],
                    flush=True,
                )
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=float(np.mean(losses)),
                seconds=time.monotonic() - before,
            )
        )
        save_json(a.out / "training_history.json", history)
        with (a.out / "training_log.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "seconds"])
            writer.writeheader()
            writer.writerows(history)
        checkpoint(
            a.out / "latest.pth",
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch + 1,
                history=history,
                torch_rng=torch.get_rng_state(),
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
            ),
        )
        checkpoint(
            a.out / "best_transfer.pth",
            dict(
                model_state_dict={
                    "unet." + k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                epoch=epoch + 1,
                selection="fixed epoch, not best validation",
            ),
        )
        print(history[-1], flush=True)
    save_json(
        a.out / "TRAINING_COMPLETE.json",
        dict(
            epochs=a.epochs,
            prepared_scans=len(ready),
            report_val_test_excluded=True,
            checkpoint="best_transfer.pth",
            selection="fixed final epoch; filename retained for transfer compatibility",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prepare", "train"], required=True)
    for name in [
        "root",
        "manifest",
        "data",
        "out",
        "source",
        "pretrained",
        "synthesis-source",
    ]:
        p.add_argument("--" + name, type=Path)
    p.add_argument(
        "--mask-dir",
        type=Path,
        help="Directory of native LCA/RCA masks; overrides the default root layout",
    )
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    a = p.parse_args()
    (prepare if a.mode == "prepare" else train)(a)
