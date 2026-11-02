"""CORA features + occupancy-weighted coronary pooling; CT stays unmasked."""

import torch
from torch import nn
from torch.nn import functional as F


def weighted_pool(features, roi):
    # Average occupancy, not NN: thin arteries cannot vanish between grid samples.
    weights = F.adaptive_avg_pool3d(roi.float(), features.shape[-3:])
    denominator = weights.sum(dim=(2, 3, 4))
    local = (features.float() * weights).sum(dim=(2, 3, 4)) / denominator.clamp_min(
        1e-8
    )
    fallback = features.float().mean(dim=(2, 3, 4))
    return torch.where(denominator > 0, local, fallback)


def territory_for_target(target):
    region = target.split("_")[0]
    if region in {
        "LM",
        "LAD",
        "LCx",
        "pLAD",
        "mLAD",
        "dLAD",
        "diagonals",
        "pLCx",
        "mdLCx",
        "OM",
        "Ramus",
    }:
        return "LCA"
    if region in {"RCA", "pRCA", "mRCA", "dRCA"}:
        return "RCA"
    if region in {"PDA", "PLB"}:
        return "union"  # Parent depends on dominance.
    raise ValueError("No explicit anatomical routing for " + target)


class CoronaryPoolingClassifier(nn.Module):
    def __init__(
        self, base, mode, disable_guidance=False, targets=None, region_groups=None
    ):
        super().__init__()
        self.encoder = base.encoder
        self.adaptive_pool = base.adaptive_pool
        self.mode = mode
        self.disable_guidance = disable_guidance
        # Same 768->2 head per task; territorial heads only receive their own ROI.
        if mode == "global":
            self.cls_head2 = nn.Linear(768, len(targets) if targets else 2)
        elif mode == "territory":
            self.lca_head = nn.Linear(768, 2)
            self.rca_head = nn.Linear(768, 2)
        elif mode == "routed":
            assert targets
            self.routes = [territory_for_target(t) for t in targets]
            self.target_heads = nn.ModuleList(nn.Linear(768, 1) for _ in targets)
        elif mode == "segment":
            assert targets and region_groups
            self.routes = [t.split("_")[0] for t in targets]
            assert set(self.routes) <= set(region_groups)
            self.region_groups = region_groups
            self.target_heads = nn.ModuleList(nn.Linear(768, 1) for _ in targets)
        else:
            raise ValueError(mode)

    def forward(self, packed):
        assert packed.shape[1] == (5 if self.mode == "segment" else 6)
        features = self.encoder(packed[:, :4])[-1]
        assert features.shape[1] == 256
        global_features = self.adaptive_pool(features).flatten(1)
        maps = (
            torch.zeros_like(packed[:, 4:]) if self.disable_guidance else packed[:, 4:]
        )
        if self.mode == "global":
            union = maps.amax(dim=1, keepdim=True)
            local = weighted_pool(features, union)
            logits = self.cls_head2(
                torch.cat([global_features, local.to(global_features.dtype)], dim=1)
            )
        elif self.mode == "territory":
            left = weighted_pool(features, maps[:, :1])
            right = weighted_pool(features, maps[:, 1:2])
            logits = torch.cat(
                [
                    self.lca_head(
                        torch.cat(
                            [global_features, left.to(global_features.dtype)], dim=1
                        )
                    ),
                    self.rca_head(
                        torch.cat(
                            [global_features, right.to(global_features.dtype)], dim=1
                        )
                    ),
                ],
                dim=1,
            )
        elif self.mode == "routed":
            pools = {
                "LCA": weighted_pool(features, maps[:, :1]),
                "RCA": weighted_pool(features, maps[:, 1:2]),
                "union": weighted_pool(features, maps.amax(dim=1, keepdim=True)),
            }
            logits = torch.cat(
                [
                    head(
                        torch.cat(
                            [global_features, pools[route].to(global_features.dtype)],
                            dim=1,
                        )
                    )
                    for head, route in zip(self.target_heads, self.routes)
                ],
                dim=1,
            )
        else:
            pools = {}
            for route in set(self.routes):
                ids = torch.tensor(self.region_groups[route], device=maps.device)
                pools[route] = weighted_pool(features, torch.isin(maps, ids))
            logits = torch.cat(
                [
                    head(
                        torch.cat(
                            [global_features, pools[route].to(global_features.dtype)],
                            dim=1,
                        )
                    )
                    for head, route in zip(self.target_heads, self.routes)
                ],
                dim=1,
            )
        return None, logits
