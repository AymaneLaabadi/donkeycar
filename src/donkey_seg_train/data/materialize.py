"""Step 3 (optional): write K augmented copies of the TRAIN frames to disk, then train on them as-is.

When to use it: only if the GPU waits on the CPU (data loading is the bottleneck), or if you need an
exactly reproducible augmented set to compare models on. The cost: the model sees only K fixed
variants of each frame instead of a fresh one every epoch, which regularises less. With online
augmentation (copies: 0, the default) epoch e of training shows views that epochs 1..e-1 never did.

Only the training circuits are materialised. Validation/test frames are never augmented (you want
to measure on real frames), and validation circuits must not leak into the augmented set.

    cache/train_aug<K>/images.npy  uint8 (K*N) x H x W x 3
    cache/train_aug<K>/masks.npy   uint8 (K*N) x H x W   ({0, 1, 255 = ignore})
    cache/train_aug<K>/meta.jsonl  source row + {"copy", "src_idx", "hflip"}; steering already flipped
    cache/train_aug<K>/info.json   everything it depends on; any change rebuilds it
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .augment import SegAugment
from .cache import publish_dir, read_json, read_rows, write_json, write_rows

MATERIALIZE_VERSION = 1


def materialize_train(train_dir: Path, train_idx: np.ndarray, out_dir: Path, augment_cfg: dict,
                      textures: Path | None, copies: int, seed: int, val_circuits: list[str],
                      force: bool = False, log=print) -> Path:
    src_info = read_json(train_dir / "info.json")
    tex_info = read_json(textures.parent / "info.json") if textures else None
    deps = {
        "version": MATERIALIZE_VERSION, "copies": copies, "seed": seed,
        "source_fingerprint": src_info["fingerprint"], "val_circuits": sorted(val_circuits),
        "augment": augment_cfg, "textures": tex_info and {k: tex_info[k] for k in ("count", "size", "seed")},
    }
    key = hashlib.sha1(json.dumps(deps, sort_keys=True).encode()).hexdigest()
    info = read_json(out_dir / "info.json")
    if not force and info and info.get("key") == key:
        log(f"[materialize] up to date ({info['n']} samples) -> {out_dir}")
        return out_dir

    n_src = len(train_idx)
    n = copies * n_src
    h, w = src_info["h"], src_info["w"]
    log(f"[materialize] writing {copies} x {n_src} = {n} augmented samples "
        f"(~{n * h * w * 4 / 2**30:.2f} GiB) ...")
    t0 = time.time()
    rows = read_rows(train_dir)
    images_src = np.load(train_dir / "images.npy", mmap_mode="r")
    masks_src = np.load(train_dir / "masks.npy", mmap_mode="r")
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    images = np.lib.format.open_memmap(tmp / "images.npy", mode="w+", dtype=np.uint8, shape=(n, h, w, 3))
    masks = np.lib.format.open_memmap(tmp / "masks.npy", mode="w+", dtype=np.uint8, shape=(n, h, w))

    torch.manual_seed(seed)   # the augmentation uses torch's RNG only -> reproducible set
    aug = SegAugment(augment_cfg, textures)
    out_rows = []
    k = 0
    for copy in range(copies):
        for j in train_idx:
            img, m, params = aug(images_src[j], masks_src[j])
            images[k] = (img * 255 + 0.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            masks[k] = m.to(torch.uint8).numpy()
            r = dict(rows[j], idx=k, src_idx=int(j), copy=copy, hflip=params["hflip"])
            if params["hflip"]:
                r["steering"] = -r["steering"]
            out_rows.append(r)
            k += 1
        log(f"[materialize] copy {copy + 1}/{copies} done")
    images.flush()
    masks.flush()
    del images, masks
    write_rows(tmp, out_rows)
    write_json(tmp / "info.json", {
        "kind": "materialized", "key": key, **deps, "n": n, "h": h, "w": w,
        "mean": src_info["mean"], "std": src_info["std"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    publish_dir(tmp, out_dir)
    log(f"[materialize] done in {time.time() - t0:.1f}s -> {out_dir}")
    return out_dir
