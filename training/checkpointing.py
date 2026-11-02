"""Atomic checkpoint writer extracted from the original training helper."""

import os
import torch


def checkpoint(path, payload):
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)
