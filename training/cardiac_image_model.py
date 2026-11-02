"""Official image tower + official-shaped head, without loading unused text tower.

This is image-only supervised adaptation, not an unmodified release reproduction.
Strictly load every visual/projection parameter; only the new2-logit head is random.
"""

import importlib.util
from pathlib import Path
import torch
from torch import nn


class CardiacImageClassifier(nn.Module):
    def __init__(self, source, pretrained, channels, outputs=2):
        super().__init__()
        spec = importlib.util.spec_from_file_location(
            "official_cardiac_vision", Path(source) / "clip_model.py"
        )
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        payload = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        visual = {
            k.removeprefix("vision_encoder.visual."): v
            for k, v in state.items()
            if k.startswith("vision_encoder.visual.")
        }
        assert visual, "No official pretrained image tower found"
        weight = visual["conv1.weight"]
        width = weight.shape[0]
        blocks = {
            int(k.split(".")[2])
            for k in visual
            if k.startswith("transformer.resblocks.")
        }
        assert blocks == set(range(len(blocks)))
        self.visual = official.VisualTransformer(
            input_resolution=(128, 144, 144),
            patch_size=8,
            width=width,
            layers=len(blocks),
            heads=width // 64,
            output_dim=width,
            resolution_after=(128, 144, 144),
        )
        self.visual.load_state_dict(visual, strict=True)
        proj = {
            k.removeprefix("multi_modal_vision_proj."): v
            for k, v in state.items()
            if k.startswith("multi_modal_vision_proj.")
        }
        hidden = proj["weight"].shape[0]
        self.projection = nn.Linear(proj["weight"].shape[1], hidden)
        self.projection.load_state_dict(proj, strict=True)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, outputs),
        )
        for module in self.cls_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        if channels == 4:
            assert weight.shape[1] == 1
            new = nn.Conv3d(4, width, kernel_size=8, stride=8, bias=False)
            with torch.no_grad():
                new.weight.copy_(weight.repeat(1, 4, 1, 1, 1) / 4)
            # The adapted stem must reproduce the single-channel stem when
            # all four channels are copies of the same image (up to roundoff).
            probe = torch.randn(1, 1, 16, 16, 16)
            torch.testing.assert_close(
                self.visual.conv1(probe),
                new(probe.repeat(1, 4, 1, 1, 1)),
                rtol=1e-4,
                atol=1e-4,
            )
            self.visual.conv1 = new
        else:
            assert channels == 1
        self.channels = channels
        self.load_audit = dict(
            visual_parameters=len(visual),
            visual_blocks=len(blocks),
            projection_parameters=len(proj),
            strict_loading=True,
            input_channels=channels,
            head_outputs=outputs,
            stem_adaptation="repeat pretrained weight /4" if channels == 4 else "none",
        )

    def forward(self, x):
        assert x.shape[1] == self.channels
        features = self.projection(self.visual(x, None))
        return features, self.cls_head(features)
