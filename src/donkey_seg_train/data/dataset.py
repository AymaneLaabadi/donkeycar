"""Dataset + DataLoader builders on top of the prepared cache.

    from donkey_seg_train.config import load_config
    from donkey_seg_train.data import build_dataloaders

    if __name__ == "__main__":            # REQUIRED on Windows when num_workers > 0 (spawn)
        loaders = build_dataloaders(load_config("train.yaml"))
        for img, mask in loaders["train"]:  # img float32 Bx3xHxW in [0,1]; mask int64 BxHxW {0,1,255}
            ...

Splits:
  train  every train circuit except data.val_circuits, augmented online (or the materialised copies)
  val    the held-out train circuits, never augmented. The split is BY CIRCUIT, because consecutive
         frames of one track are near-duplicates: a random frame split would put the "same" image on
         both sides and the validation score would mostly measure memorisation.
  test   the test split (other seeds; partly unseen tracks), never augmented
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augment import SegAugment, to_tensors
from .cache import read_json, read_rows


class CachedSegDataset(Dataset):
    """Reads one prepared directory (decoded split cache or materialised copies)."""

    def __init__(self, split_dir: str | Path, indices=None, transform: SegAugment | None = None,
                 return_meta: bool = False):
        self.dir = Path(split_dir)
        self.info = read_json(self.dir / "info.json")
        if self.info is None:
            raise FileNotFoundError(f"{self.dir} is not prepared; run: python -m donkey_seg_train.data.prepare")
        self.rows = read_rows(self.dir)
        self.indices = np.arange(len(self.rows)) if indices is None else np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.return_meta = return_meta
        self._images = self._masks = None

    # The memmaps are opened lazily inside each worker. Pickling a Dataset to a spawned worker
    # (Windows) then sends only paths and the index, never the arrays.
    def _open(self):
        self._images = np.load(self.dir / "images.npy", mmap_mode="r")
        self._masks = np.load(self.dir / "masks.npy", mmap_mode="r")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = state["_masks"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self._images is None:
            self._open()
        j = int(self.indices[i])
        img_u8, mask_u8 = self._images[j], self._masks[j]
        if self.transform is not None:
            img, mask, params = self.transform(img_u8, mask_u8)
        else:
            (img, mask), params = to_tensors(img_u8, mask_u8), {}
        if not self.return_meta:
            return img, mask
        meta = dict(self.rows[j])
        if params.get("hflip"):
            meta["steering"] = -meta["steering"]   # a mirrored frame turns the other way
            meta["hflip"] = True
        meta.setdefault("hflip", False)
        return img, mask, meta


def split_by_circuit(rows: list[dict], val_circuits) -> tuple[np.ndarray, np.ndarray]:
    val = set(val_circuits)
    present = {r["circuit_id"] for r in rows}
    missing = val - present
    if missing:
        raise ValueError(f"val_circuits {sorted(missing)} not in train split (have {sorted(present)})")
    is_val = np.array([r["circuit_id"] in val for r in rows])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def _worker_init(_):
    # Each worker is its own process; without this, 4 workers x N intra-op threads oversubscribe the CPU.
    torch.set_num_threads(1)


def build_datasets(cfg: dict, return_meta: bool = False) -> dict[str, Dataset]:
    from .prepare import prepare   # idempotent: a no-op when everything is up to date

    paths = prepare(cfg)
    rows = read_rows(paths["train"])
    train_idx, val_idx = split_by_circuit(rows, cfg["data"]["val_circuits"])
    if paths.get("materialized"):
        train = CachedSegDataset(paths["materialized"], return_meta=return_meta)
    else:
        aug = SegAugment(cfg["augment"], paths["textures"] / "textures.npy")
        train = CachedSegDataset(paths["train"], train_idx, aug, return_meta)
    out = {"train": train, "val": CachedSegDataset(paths["train"], val_idx, None, return_meta)}
    if paths.get("test"):
        out["test"] = CachedSegDataset(paths["test"], None, None, return_meta)
    return out


def build_dataloaders(cfg: dict, return_meta: bool = False) -> dict[str, DataLoader]:
    d = cfg["data"]
    workers = int(d["num_workers"])
    common = dict(
        batch_size=d["batch_size"],
        num_workers=workers,
        pin_memory=bool(d["pin_memory"]) and torch.cuda.is_available(),
        persistent_workers=bool(d["persistent_workers"]) and workers > 0,
        worker_init_fn=_worker_init if workers > 0 else None,
    )
    loaders = {}
    for name, ds in build_datasets(cfg, return_meta).items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            ds, shuffle=is_train, drop_last=is_train,   # drop_last: no tiny last batch -> stable BatchNorm
            generator=torch.Generator().manual_seed(int(d["seed"])) if is_train else None,
            **common,
        )
    return loaders
