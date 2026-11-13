"""Look at the augmentations before training on them (the cheapest bug-catcher there is).

    python -m donkey_seg_train.data.preview -n 8 --views 5
    python -m donkey_seg_train.data.preview --bench 2000     # also time the pipeline (samples/s)

Writes cache/preview_augment.png: one row per frame; the first column is the original, the other
columns are augmented views with the mask outlined (green = road, red = ignore).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .augment import IGNORE_INDEX, SegAugment, to_tensors
from .dataset import split_by_circuit
from .prepare import prepare
from .cache import read_rows


def overlay(img: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    x = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    road = (mask == 1).numpy()
    edge = road ^ np.roll(road, 1, 0) | road ^ np.roll(road, 1, 1)
    x[road] = (0.75 * x[road] + 0.25 * np.array([0, 255, 0])).astype(np.uint8)
    x[edge] = [0, 255, 0]
    x[(mask == IGNORE_INDEX).numpy()] = [255, 0, 0]
    return x


def main(argv=None) -> None:
    from ..config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train.yaml")
    ap.add_argument("-n", type=int, default=8, help="frames (rows)")
    ap.add_argument("--views", type=int, default=5, help="augmented views per frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0, help="time this many augmentations")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    paths = prepare(cfg)
    torch.manual_seed(a.seed)
    aug = SegAugment(cfg["augment"], paths["textures"] / "textures.npy")
    images = np.load(paths["train"] / "images.npy", mmap_mode="r")
    masks = np.load(paths["train"] / "masks.npy", mmap_mode="r")
    train_idx, _ = split_by_circuit(read_rows(paths["train"]), cfg["data"]["val_circuits"])
    pick = train_idx[torch.randperm(len(train_idx))[:a.n].numpy()]

    rows = []
    for j in pick:
        tiles = [overlay(*to_tensors(images[j], masks[j]))]
        tiles += [overlay(*aug(images[j], masks[j])[:2]) for _ in range(a.views)]
        rows.append(np.concatenate([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255) for t in tiles], 1))
    out = Path(cfg["data"]["cache_dir"]) / "preview_augment.png"
    Image.fromarray(np.concatenate(rows, 0)).resize(
        (rows[0].shape[1] * 2, len(rows) * rows[0].shape[0] * 2), Image.NEAREST).save(out)
    print(f"wrote {out}")

    if a.bench:
        torch.set_num_threads(1)   # like inside a DataLoader worker
        t0 = time.perf_counter()
        for k in range(a.bench):
            j = train_idx[k % len(train_idx)]
            aug(images[j], masks[j])
        dt = time.perf_counter() - t0
        print(f"augment: {a.bench / dt:.0f} samples/s per worker ({1e3 * dt / a.bench:.2f} ms/sample)")


if __name__ == "__main__":
    main()
