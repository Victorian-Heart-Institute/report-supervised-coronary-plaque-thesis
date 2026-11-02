"""Construct a historical training command; dry-run unless --execute is supplied."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def build_command(config, args):
    root = Path(__file__).resolve().parent
    runner = "train_pooling.py" if config["pool_mode"] else "train_classifier.py"
    command = [
        sys.executable,
        str(root / "training" / runner),
        "--source",
        str(Path(args.source).resolve()),
        "--labels",
        str(Path(args.labels).resolve()),
        "--data",
        str(Path(args.data).resolve()),
        "--out",
        str(Path(args.out).resolve()),
        "--targets",
        ",".join(config["targets"]),
        "--architecture",
        config["architecture"],
        "--input-mode",
        config["input_mode"],
        "--epochs",
        str(config["epochs"]),
        "--amp-dtype",
        config["amp_dtype"],
        "--dataset-contract",
        str(Path(args.contract or root / config["contract"]).resolve()),
    ]
    if config["initialisation"] != "random":
        if not args.pretrained:
            raise ValueError(
                "Provide --pretrained; synthetic variants require the fixed epoch-5 adapted checkpoint"
            )
        command += ["--pretrained", str(Path(args.pretrained).resolve())]
    if config["pool_mode"]:
        if not args.roi_data:
            raise ValueError("Pooling needs --roi-data")
        command += [
            "--roi-data",
            str(Path(args.roi_data).resolve()),
            "--pool-mode",
            config["pool_mode"],
        ]
    if args.smoke_only:
        command += ["--smoke-only"]
    if args.eval_only:
        if not args.checkpoint:
            raise ValueError(
                "Evaluation needs --checkpoint pointing to the validation-selected model"
            )
        command += ["--eval-only", "--checkpoint", str(Path(args.checkpoint).resolve())]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "source", "labels", "data", "out"]:
        p.add_argument("--" + name, required=True)
    for name in ["contract", "pretrained", "roi-data", "checkpoint"]:
        p.add_argument("--" + name)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    command = build_command(config, a)
    print(shlex.join(command), flush=True)
    if a.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
