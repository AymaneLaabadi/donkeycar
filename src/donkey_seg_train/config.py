"""train.yaml loading.

The defaults here (and in AugmentConfig) are the single source of truth; the YAML only overrides them.
An unknown key is an error, because a typo like `hflip_P` would otherwise be silently ignored.
Relative paths are resolved against the YAML file's folder, so commands work from any cwd.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import yaml

from .data.augment import AugmentConfig

DEFAULTS: dict = {
    "data": {
        "root": "generated_data",          # output dir of donkey_seg_datagen
        "cache_dir": "cache",              # every prepared artefact lives here (safe to delete)
        "val_circuits": ["gen_s388044"],   # held out of train BY CIRCUIT (no near-duplicate leakage)
        "batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "seed": 0,
    },
    "textures": {                          # Step 2: bank of background noise textures
        "count": 128,
        "size": [240, 320],                # h, w; larger than a frame so random crops differ
        "seed": 1234,
    },
    "materialize": {                       # Step 3 (optional): augmented copies written to disk
        "copies": 0,                       # 0 = augment online (recommended)
        "seed": 0,
    },
    "augment": asdict(AugmentConfig()),
}

_PATH_KEYS = (("data", "root"), ("data", "cache_dir"))


def _merge(base: dict, over: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise KeyError(f"unknown config key '{where}{k}' (known: {', '.join(base)})")
        if isinstance(base[k], dict) and isinstance(v, dict):
            out[k] = _merge(base[k], v, f"{where}{k}.")
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = "train.yaml") -> dict:
    """Defaults merged with the YAML (if it exists). Paths are made absolute."""
    user, base_dir = {}, Path.cwd()
    if path is not None and Path(path).exists():
        path = Path(path).resolve()
        user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base_dir = path.parent
    cfg = _merge(DEFAULTS, user)
    for sec, key in _PATH_KEYS:
        p = Path(cfg[sec][key])
        cfg[sec][key] = str(p if p.is_absolute() else (base_dir / p).resolve())
    AugmentConfig.from_dict(cfg["augment"])  # validate early, not inside a worker
    return cfg
