"""Data pipeline: prepare once (cache.py, textures.py, materialize.py via prepare.py), augment online
(augment.py), load (dataset.py). See each module's docstring for the why.

`prepare` is imported from .prepare directly (not re-exported here, so `python -m
donkey_seg_train.data.prepare` runs without a double-import warning)."""

from .augment import IGNORE_INDEX, AugmentConfig, SegAugment
from .dataset import CachedSegDataset, build_dataloaders, build_datasets, split_by_circuit

__all__ = [
    "IGNORE_INDEX", "AugmentConfig", "SegAugment", "CachedSegDataset", "build_dataloaders",
    "build_datasets", "split_by_circuit",
]
