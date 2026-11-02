"""Convert existing 224³ HU crops; this does not reconstruct historical Drive crops."""

import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom

WINDOWS = [(-170, -30), (-150, 250), (0, 700), (-500, 1500)]


def transform(hu, mode):
    if hu.shape != (224, 224, 224) or not np.isfinite(hu).all():
        raise ValueError("Expected finite, unnormalised 224³ CT_HU")
    if mode == "cora4":
        return "STEP4_CROP", np.stack(
            [np.clip((hu - low) / (high - low), 0, 1) for low, high in WINDOWS]
        ).astype(np.float16)
    image = zoom(
        hu.astype(np.float32),
        np.array([128, 144, 144]) / 224,
        order=1,
        prefilter=False,
        grid_mode=False,
    )
    low, high = float(image.min()), float(image.max())
    if high <= low:
        raise ValueError("Constant volume cannot be min–max normalised")
    return "CARDIAC_RAW", ((image - low) / (high - low))[None].astype(np.float16)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["cora4", "cardiac_native"], required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for source in sorted(a.raw.glob("*.npz")):
        dest = a.out / source.name
        if dest.exists():
            raise FileExistsError(dest)
        with np.load(source, allow_pickle=False) as z:
            key, image = transform(z["CT_HU"].astype(np.float32), a.mode)
        np.savez_compressed(dest, **{key: image})


if __name__ == "__main__":
    main()
