# Portable thesis export; see docs/PROVENANCE.json for scope and changes.
"""CORA 224-cube classification and validation-selected held-out evaluation.

Training recipe: unfrozen official encoder, concat pool + linear output,
AdamW 1e-4/wd1e-4, batch1/accum4, weighted BCE clamp[.5,5], seed20260804.
Unknown labels are masked; completely unknown rows are excluded from training.
No test inference until the validation-selected best checkpoint is finalized.
"""
from contracts import validate_manifest
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Crops(Dataset):

    def __init__(self, frame, data, targets, input_mode="cora4", roi_data=None):
        self.frame, self.data, self.targets = (
            frame.reset_index(drop=True),
            data,
            targets,
        )
        self.input_mode = input_mode
        self.roi_data = roi_data

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with np.load(self.data / (row.scan_id + ".npz"), allow_pickle=False) as z:
            key = {
                "cora4": "STEP4_CROP",
                "cardiac_raw": "CARDIAC_RAW",
                "cardiac_cora4": "CARDIAC_CORA4",
                "raw1": "CT_HU",
            }[self.input_mode]
            x = z[key].astype(np.float32)
            if self.input_mode == "raw1":
                x = np.clip((x + 1000.0) / 3000.0, 0, 1)[None]
        expected = (
            (1, 224, 224, 224)
            if self.input_mode == "raw1"
            else (
                (4, 224, 224, 224)
                if self.input_mode == "cora4"
                else (1 if self.input_mode == "cardiac_raw" else 4, 128, 144, 144)
            )
        )
        assert x.shape == expected and np.isfinite(x).all()
        if self.roi_data is not None:
            with np.load(
                self.roi_data / (row.scan_id + ".npz"), allow_pickle=False
            ) as z:
                if "SEGMENT_ROI" in z:
                    roi = z["SEGMENT_ROI"][None].astype(np.float32)
                    assert (
                        roi.shape == (1, 224, 224, 224)
                        and np.isin(roi, range(17)).all()
                    )
                else:
                    roi = z["ROI"].astype(np.float32)
                    assert (
                        roi.shape == (2, 224, 224, 224) and np.isin(roi, [0, 1]).all()
                    )
                assert int(z["margin_mm"]) == 3
            x = np.concatenate([x, roi], axis=0)
        y = row[self.targets].to_numpy(dtype=np.float32)
        return (torch.from_numpy(x), torch.from_numpy(y), row.scan_id)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--pretrained", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument(
        "--architecture",
        choices=["cora", "resnet18_gn", "cardiac_clip"],
        default="cora",
    )
    p.add_argument(
        "--input-mode",
        choices=["cora4", "cardiac_raw", "cardiac_cora4", "raw1"],
        default="cora4",
    )
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--dataset-contract", type=Path)
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--roi-data", type=Path, required=True)
    p.add_argument(
        "--pool-mode",
        choices=["global", "territory", "routed", "segment"],
        required=True,
    )
    p.add_argument(
        "--disable-guidance",
        action="store_true",
        help="Matched readout/capacity control; use global mean instead of CA average",
    )
    a = p.parse_args()
    assert a.architecture == "cora" and a.input_mode == "cora4"
    gate = json.loads((a.roi_data / "PREPARATION_OK.json").read_text())
    assert gate["training_gate_passed"] and gate["margin_mm"] == 3
    assert Path(gate["roi_root"]).resolve() == a.roi_data.resolve()
    expected_targets = (
        ["LCA_calcified", "LCA_non_calcified", "RCA_calcified", "RCA_non_calcified"]
        if a.pool_mode == "territory"
        else json.loads(a.dataset_contract.read_text())["targets"]
    )
    assert a.targets.split(",") == expected_targets
    targets = a.targets.split(",")
    seed = 20260804
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    assert torch.cuda.is_available(), "CUDA-capable GPU required"
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    sys.path.insert(0, str(a.source))
    d = pd.read_csv(a.labels, dtype={"scan_id": "string", "patient_id": "string"})
    guidance = pd.read_csv(a.roi_data / "qa.csv", dtype={"scan_id": "string"})
    if a.pool_mode == "segment":
        assert gate["provider"] == "liam_scct16"
        region_groups = (
            gate["four_family_groups"]
            if len(targets) == 8
            else gate["detailed14_groups"]
        )
        assert len(targets) in [8, 28]
        validity = guidance.region_valid.map(json.loads)
        for region, ids in region_groups.items():
            guidance["mask_" + region + "_valid"] = validity.map(
                lambda v: any((v[str(i)] for i in ids))
            )
        guidance["mask_union_valid"] = validity.map(lambda v: any(v.values()))
        guidance["mask_both_valid"] = False
        guidance = guidance[
            ["scan_id", "status", "mask_union_valid", "mask_both_valid"]
            + ["mask_" + g + "_valid" for g in region_groups]
        ].rename(columns={"status": "mask_qa_status"})
    else:
        validity = guidance.territory_valid.map(json.loads)
        guidance["mask_LCA_valid"] = validity.map(lambda v: bool(v[0]))
        guidance["mask_RCA_valid"] = validity.map(lambda v: bool(v[1]))
        guidance["mask_union_valid"] = validity.map(any)
        guidance["mask_both_valid"] = validity.map(all)
        guidance = guidance[
            [
                "scan_id",
                "status",
                "mask_LCA_valid",
                "mask_RCA_valid",
                "mask_union_valid",
                "mask_both_valid",
            ]
        ].rename(columns={"status": "mask_qa_status"})
    assert guidance.scan_id.is_unique and set(d.scan_id) == set(guidance.scan_id)
    assert d.scan_id.is_unique and d.groupby("patient_id").split.nunique().max() == 1
    assert set(d.split) == {"train", "val", "test"}
    for target in targets:
        assert set(d[target].dropna().unique()) <= {0, 1}
    sha = validate_manifest(d, a.labels, targets, a.dataset_contract)
    if a.dataset_contract:
        assert (
            a.architecture in ["cora", "resnet18_gn"]
            and a.input_mode == "cora4"
            or (a.architecture == "resnet18_gn" and a.input_mode == "raw1")
            or (
                a.architecture == "cardiac_clip"
                and a.input_mode in ["cardiac_raw", "cardiac_cora4"]
            )
        )
    frames = {
        s: d[d.split == s].reset_index(drop=True) for s in ["train", "val", "test"]
    }
    train = frames["train"][frames["train"][targets].notna().any(axis=1)].reset_index(
        drop=True
    )
    a.out.mkdir(parents=True, exist_ok=True)
    if a.architecture == "cora":
        from models.model import CORAClassifier

        model = CORAClassifier(num_input_channels=4, pretrained_path=None)
        if len(targets) != 2:
            model.cls_head2 = nn.Linear(512, len(targets))
        from coronary_pooling import CoronaryPoolingClassifier

        model = CoronaryPoolingClassifier(
            model,
            a.pool_mode,
            a.disable_guidance,
            targets,
            region_groups if a.pool_mode == "segment" else None,
        )
    elif a.architecture == "resnet18_gn":
        from monai.networks.nets import resnet18

        class ResNetClassifier(nn.Module):

            def __init__(self):
                super().__init__()
                self.network = resnet18(
                    spatial_dims=3,
                    n_input_channels=1 if a.input_mode == "raw1" else 4,
                    num_classes=len(targets),
                    pretrained=False,
                )

                def groupnorm(module):
                    for name, child in list(module.named_children()):
                        if isinstance(child, nn.BatchNorm3d):
                            setattr(
                                module,
                                name,
                                nn.GroupNorm(
                                    min(32, child.num_features), child.num_features
                                ),
                            )
                        else:
                            groupnorm(child)

                groupnorm(self.network)

            def forward(self, x):
                logits = self.network(x)
                return (None, logits)

        model = ResNetClassifier()
        assert a.pretrained is None, "Scratch baseline must not load pretrained weights"
    else:
        from cardiac_image_model import CardiacImageClassifier

        assert (
            a.input_mode in ["cardiac_raw", "cardiac_cora4"]
            and a.pretrained is not None
        )
        model = CardiacImageClassifier(
            a.source,
            a.pretrained,
            1 if a.input_mode == "cardiac_raw" else 4,
            len(targets),
        )
        (a.out / "pretrained_load_audit.json").write_text(
            json.dumps(model.load_audit, indent=2) + "\n"
        )
    if not a.eval_only and a.architecture == "cora":
        assert a.pretrained is not None
        state = torch.load(a.pretrained, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        enc = {
            k.removeprefix("unet.encoder."): v
            for k, v in state.items()
            if k.startswith("unet.encoder.")
        }
        assert enc, "Expected official full-model checkpoint encoder keys"
        model.encoder.load_state_dict(enc, strict=True)
    model.cuda()
    generator = torch.Generator().manual_seed(seed)

    def loader(frame, shuffle=False):
        return DataLoader(
            Crops(frame, a.data, targets, a.input_mode, a.roi_data),
            batch_size=1,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    train_loader, val_loader = (loader(train, True), loader(frames["val"]))
    positives = train[targets].eq(1).sum().to_numpy()
    negatives = train[targets].eq(0).sum().to_numpy()
    assert a.eval_only or ((positives > 0) & (negatives > 0)).all()
    weights = torch.tensor(
        negatives / np.maximum(positives, 1), dtype=torch.float32, device="cuda"
    ).clamp(0.5, 5)
    if a.architecture == "cardiac_clip":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.visual.parameters(), "lr": 1e-05},
                {"params": model.projection.parameters(), "lr": 5e-05},
                {"params": model.cls_head.parameters(), "lr": 5e-05},
            ],
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.0001, weight_decay=0.0001
        )
    amp_dtype = torch.bfloat16 if a.amp_dtype == "bfloat16" else torch.float16
    assert a.amp_dtype != "bfloat16" or torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp_dtype == "float16")
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights, reduction="none")
    if a.smoke_only:
        x, y, _ = next(iter(train_loader))
        x, y = (x.cuda(), y.cuda())
        if a.amp_dtype == "bfloat16":
            diagnostics = []
            for diagnostic_scale in [65536.0, 1.0]:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    _, diagnostic_logits = model(x)
                    diagnostic_known = torch.isfinite(y)
                    diagnostic_loss = (
                        criterion(diagnostic_logits, torch.nan_to_num(y))
                        * diagnostic_known
                    ).sum() / diagnostic_known.sum()
                (diagnostic_loss * diagnostic_scale).backward()
                present = [
                    (n, p.grad)
                    for n, p in model.named_parameters()
                    if p.grad is not None
                ]
                nonfinite = [n for n, g in present if not torch.isfinite(g).all()]
                diagnostics.append(
                    dict(
                        dtype="float16",
                        scale=diagnostic_scale,
                        finite_logits=bool(torch.isfinite(diagnostic_logits).all()),
                        loss=float(diagnostic_loss.detach()),
                        gradient_tensors=len(present),
                        nonfinite_gradient_tensors=len(nonfinite),
                        first_nonfinite_names=nonfinite[:8],
                    )
                )
            (a.out / "fp16_preflight_diagnostic.json").write_text(
                json.dumps(diagnostics, indent=2) + "\n"
            )
            print("FP16 PREFLIGHT DIAGNOSTIC", json.dumps(diagnostics), flush=True)
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            _, logits = model(x)
            known = torch.isfinite(y)
            loss = (criterion(logits, torch.nan_to_num(y)) * known).sum() / known.sum()
        assert logits.shape == (1, len(targets)) and torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        print(
            "GRADIENT CHECK",
            a.amp_dtype,
            "tensors=",
            len(gradients),
            "nonfinite=",
            sum((not bool(torch.isfinite(g).all()) for g in gradients)),
            flush=True,
        )
        assert gradients and all(
            (torch.isfinite(g).all() for g in gradients)
        ), "Missing or non-finite gradients; no training permitted"
        assert any((g.abs().max() > 0 for g in gradients))
        if a.architecture == "cardiac_clip":
            assert model.visual.conv1.weight.grad is not None
            assert model.visual.conv1.weight.grad.abs().max() > 0
        if a.architecture == "cora":
            assert any(
                (
                    p.grad is not None and p.grad.abs().max() > 0
                    for p in model.encoder.parameters()
                )
            )
        print(
            "SMOKE PASSED",
            a.architecture,
            "input=",
            tuple(x.shape),
            "loss=",
            float(loss),
            flush=True,
        )
        return

    def save(path, payload):
        temporary = str(path) + ".tmp"
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def phase(frame_loader, training, epoch):
        model.train(training)
        total = 0.0
        known_batches = 0
        rows = []
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        for i, (x, y, ids) in enumerate(frame_loader):
            x, y = (x.cuda(non_blocking=True), y.cuda(non_blocking=True))
            known = torch.isfinite(y)
            with torch.set_grad_enabled(training), torch.autocast(
                "cuda", dtype=amp_dtype
            ):
                _, logits = model(x)
                raw = criterion(logits, torch.nan_to_num(y))
                loss = (raw * known).sum() / known.sum().clamp_min(1)
            if training:
                scaler.scale(loss / 4).backward()
                if (i + 1) % 4 == 0 or i + 1 == len(frame_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            if known.any():
                total += float(loss.detach())
                known_batches += 1
            probabilities = logits.detach().sigmoid().float().cpu().numpy()
            truth = y.cpu().numpy()
            for j, sid in enumerate(ids):
                row = {"scan_id": sid}
                row.update({t: float(truth[j, k]) for k, t in enumerate(targets)})
                row.update(
                    {
                        "p_" + t: float(probabilities[j, k])
                        for k, t in enumerate(targets)
                    }
                )
                rows.append(row)
            if i % 100 == 0:
                print(
                    f"epoch={epoch} training={training} batch={i + 1}/{len(frame_loader)} loss={total / max(known_batches, 1):.4f}",
                    flush=True,
                )
        predictions = pd.DataFrame(rows).merge(
            d[["scan_id", "patient_id", "cohort"]], on="scan_id", validate="one_to_one"
        )
        predictions = predictions.merge(guidance, on="scan_id", validate="one_to_one")
        return (total / max(known_batches, 1), predictions, time.monotonic() - started)

    best_path, latest_path = (a.out / "best.pth", a.out / "latest.pth")
    epoch0, best, bad = (0, -float("inf"), 0)
    if not a.eval_only:
        recipe = dict(
            targets=targets,
            labels_sha256=sha,
            input=[1, 4, 224, 224, 224],
            seed=seed,
            batch=1,
            accumulation=4,
            lr=0.0001,
            weight_decay=0.0001,
            encoder_frozen=False,
            head=f"Linear(512,{len(targets)})",
            pos_weight=weights.cpu().tolist(),
            unknown_policy="masked; entirely unknown train rows excluded",
            split_rows={s: len(f) for s, f in frames.items()},
            train_used=len(train),
            test_policy="only finalized validation-selected best checkpoint",
            architecture=a.architecture,
            amp_dtype=a.amp_dtype,
            dataset_contract=(
                json.loads(a.dataset_contract.read_text())
                if a.dataset_contract
                else None
            ),
            pretrained_sha256=(
                hashlib.sha256(a.pretrained.read_bytes()).hexdigest()
                if a.pretrained
                else None
            ),
        )
        if a.architecture == "resnet18_gn":
            recipe.update(
                initialization="random; no pretrained weights",
                pooling="MONAI ResNet adaptive average pooling",
                normalization_layers="GroupNorm replaces BatchNorm3d",
                head=f"Linear(512,{len(targets)})",
                encoder_frozen=False,
            )
            recipe.update(
                input=[1, 1 if a.input_mode == "raw1" else 4, 224, 224, 224],
                input_mode=a.input_mode,
                normalization=(
                    "clip HU[-1000,2000], scale[0,1]"
                    if a.input_mode == "raw1"
                    else "official CORA four HU windows"
                ),
                geometry=(
                    "legacy LPS central0.5mm raw crop; pixel equivalence to Drive crop unverified"
                    if a.input_mode == "raw1"
                    else "existing Drive central224 crop; nominal0.5mm"
                ),
            )
        if a.architecture == "cardiac_clip":
            recipe.update(
                input=[1, model.channels, 128, 144, 144],
                input_mode=a.input_mode,
                lr=1e-05,
                head_projection_lr=5e-05,
                weight_decay=0.01,
                head=f"Linear→LayerNorm→GELU→Linear({len(targets)})",
                normalization=(
                    "per-volume minmax"
                    if model.channels == 1
                    else "CORA four HU windows"
                ),
                geometry="existing central224 crop, nominal112mm per provenance; resized grid NOT0.5mm; native raw equivalence unverified",
                adaptation="image-only supervised head; no text inputs or contrastive training",
                scheduler="constant learning rate; not official warmup/decay reproduction",
            )
        recipe.update(
            pool_mode=a.pool_mode,
            roi_guidance_enabled=not a.disable_guidance,
            input_packed=[1, 6, 224, 224, 224],
            encoder_initialization_path=str(a.pretrained),
            encoder_initialization=(
                "AP plaque-shape adapted CORA"
                if a.pretrained.parent.name == "CORA_AP_ENCODER_736_V1"
                else "CORA pretrained checkpoint"
            ),
            encoder_input=[1, 4, 224, 224, 224],
            mask_channels_auxiliary_not_encoder=True,
            head=(
                f"Linear(768,{len(targets)})"
                if a.pool_mode == "global"
                else (
                    "two territory-specific Linear(768,2) heads"
                    if a.pool_mode == "territory"
                    else f"{len(targets)} territory-routed Linear(768,1) heads"
                )
            ),
            roi_root=str(a.roi_data),
            roi_preparation=gate,
            pooling="global max+average (512) plus occupancy-weighted local average (256)",
            fallback="empty/invalid territory ROI uses global feature average; CT always intact",
            dataset_policy="same full patient-separated splits; no silent mask-based scan exclusion",
        )
        (a.out / "config.json").write_text(json.dumps(recipe, indent=2) + "\n")
        if a.pool_mode == "routed":
            from coronary_pooling import territory_for_target

            recipe["target_mask_routing"] = {
                t: territory_for_target(t) for t in targets
            }
            recipe["routing_limitation"] = (
                "LCA/RCA territory guidance, not per-family or per-segment masks; PDA/PLB use union because dominance unknown"
            )
            (a.out / "config.json").write_text(json.dumps(recipe, indent=2) + "\n")
        if a.pool_mode == "segment":
            recipe.update(
                input_packed=[1, 5, 224, 224, 224],
                target_region_label_ids={
                    t: region_groups[t.split("_")[0]] for t in targets
                },
                routing_limitation="Predicted Liam anatomical labels, not gold regional segmentation; missing regions global-mean fallback",
                region_margin_policy=gate["margin_policy"],
            )
            (a.out / "config.json").write_text(json.dumps(recipe, indent=2) + "\n")
        if latest_path.exists():
            c = torch.load(latest_path, map_location="cuda", weights_only=False)
            assert c["manifest_sha256"] == sha and c["targets"] == targets
            model.load_state_dict(c["model"], strict=True)
            optimizer.load_state_dict(c["optimizer"])
            scaler.load_state_dict(c["scaler"])
            epoch0, best, bad = (c["epoch"] + 1, c["best_val_auc"], c["bad_epochs"])
        for epoch in range(epoch0, a.epochs):
            if bad >= 8:
                break
            tl, tp, train_seconds = phase(train_loader, True, epoch + 1)
            summarize(tp, targets).to_csv(
                a.out / f"epoch_{epoch + 1}_training_metrics.csv", index=False
            )
            vl, vp, val_seconds = phase(val_loader, False, epoch + 1)
            metrics = summarize(vp, targets)
            score = float(metrics.AUROC.mean())
            vp.to_csv(
                a.out / f"epoch_{epoch + 1}_validation_predictions.csv", index=False
            )
            metrics.to_csv(
                a.out / f"epoch_{epoch + 1}_validation_metrics.csv", index=False
            )
            improved = np.isfinite(score) and score > best + 0.0001
            best, bad = (score, 0) if improved else (best, bad + 1)
            c = dict(
                epoch=epoch,
                best_val_auc=best,
                bad_epochs=bad,
                targets=targets,
                manifest_sha256=sha,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
            )
            if improved:
                save(best_path, c)
            save(latest_path, c)
            log = a.out / "training_log.csv"
            pd.DataFrame(
                [
                    dict(
                        epoch=epoch + 1,
                        train_loss=tl,
                        val_loss=vl,
                        val_AUROC=score,
                        best_val_AUROC=best,
                        bad_epochs=bad,
                        train_seconds=train_seconds,
                        val_seconds=val_seconds,
                    )
                ]
            ).to_csv(log, mode="a", header=not log.exists(), index=False)
            print(
                f"epoch={epoch + 1} val_AUROC={score:.4f} best={best:.4f} patience={bad}/8",
                flush=True,
            )
            if bad >= 8:
                break
    checkpoint = a.checkpoint if a.eval_only else best_path
    assert checkpoint is not None and checkpoint.is_file()
    c = torch.load(checkpoint, map_location="cuda", weights_only=False)
    assert c["manifest_sha256"] == sha
    if "targets" in c:
        assert c["targets"] == targets
    model.load_state_dict(c["model"], strict=True)
    _, validation, _ = phase(val_loader, False, c["epoch"] + 1)
    thresholds = validation_thresholds(validation, targets)
    (a.out / "validation_thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n"
    )
    _, test, seconds = phase(loader(frames["test"]), False, c["epoch"] + 1)
    test.to_csv(a.out / "test_predictions.csv", index=False)
    validation.to_csv(a.out / "selected_validation_predictions.csv", index=False)
    default, tuned = (summarize(test, targets), summarize(test, targets, thresholds))
    default.to_csv(a.out / "test_metrics_default_05.csv", index=False)
    tuned.to_csv(a.out / "test_metrics_validation_threshold.csv", index=False)
    groups = [
        summarize(g, targets).assign(mask_union_valid=str(flag))
        for flag, g in test.groupby("mask_union_valid")
    ]
    pd.concat(groups).to_csv(
        a.out / "test_metrics_by_guidance_availability.csv", index=False
    )
    if a.pool_mode == "territory":
        groups = [
            summarize(g, [target]).assign(territory_mask_valid=str(flag))
            for target in targets
            for flag, g in test.groupby(
                "mask_LCA_valid" if target.startswith("LCA_") else "mask_RCA_valid"
            )
        ]
        pd.concat(groups).to_csv(
            a.out / "test_metrics_by_territory_mask_availability.csv", index=False
        )
    if a.pool_mode == "routed":
        from coronary_pooling import territory_for_target

        groups = [
            summarize(g, [target]).assign(
                mask_route=territory_for_target(target), route_mask_valid=str(flag)
            )
            for target in targets
            for flag, g in test.groupby(
                {
                    "LCA": "mask_LCA_valid",
                    "RCA": "mask_RCA_valid",
                    "union": "mask_union_valid",
                }[territory_for_target(target)]
            )
        ]
        pd.concat(groups).to_csv(
            a.out / "test_metrics_by_territory_mask_availability.csv", index=False
        )
    if a.pool_mode == "segment":
        groups = [
            summarize(g, [target]).assign(region_mask_valid=str(flag))
            for target in targets
            for flag, g in test.groupby("mask_" + target.split("_")[0] + "_valid")
        ]
        pd.concat(groups).to_csv(
            a.out / "test_metrics_by_region_mask_availability.csv", index=False
        )
    pd.concat(
        [
            summarize(g, targets).assign(cohort=str(cohort))
            for cohort, g in test.groupby("cohort", dropna=False)
        ]
    ).to_csv(a.out / "test_metrics_by_cohort.csv", index=False)
    (a.out / "test_summary.json").write_text(
        json.dumps(
            dict(
                best_epoch=c["epoch"] + 1,
                labels_sha256=sha,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                targets=targets,
                test_scans=len(test),
                test_seconds=seconds,
                macro_test_AUROC=float(default.AUROC.mean()),
                macro_test_AP=float(default.AP.mean()),
                evaluation_limitation="Historical test set previously examined; this is not a new untouched external test.",
                threshold_policy="validation-only maximum balanced accuracy",
            ),
            indent=2,
        )
        + "\n"
    )
    print(default.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
