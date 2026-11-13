"""Run every preparation step, in order. Each step is cached and skipped while its inputs are unchanged.

    python -m donkey_seg_train.data.prepare                 # uses ./train.yaml
    python -m donkey_seg_train.data.prepare --force         # rebuild everything
    python -m donkey_seg_train.data.prepare --only textures

    Step 1  cache         PNG -> uint8 .npy memmaps (train, test) + slim meta + mean/std
    Step 2  textures      procedural noise bank for the background augmentation
    Step 3  materialize   (only if materialize.copies > 0) K augmented copies of the train circuits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cache import build_split_cache, read_rows
from .materialize import materialize_train
from .textures import ensure_texture_bank

STEPS = ("cache", "textures", "materialize")


def prepare(cfg: dict, force: bool = False, only: str | None = None, log=print) -> dict[str, Path]:
    d = cfg["data"]
    root, cache_dir = Path(d["root"]), Path(d["cache_dir"])
    paths: dict[str, Path] = {}

    def run(step: str) -> bool:
        return only is None or only == step

    # Step 1
    paths["train"] = build_split_cache(root, "train", cache_dir, force and run("cache"), log=log)
    if (root / "test" / "plan.json").exists():
        paths["test"] = build_split_cache(root, "test", cache_dir, force and run("cache"), log=log)

    # Step 2
    t = cfg["textures"]
    paths["textures"] = ensure_texture_bank(cache_dir, t["count"], t["size"], t["seed"],
                                            force and run("textures"), log=log)

    # Step 3
    m = cfg["materialize"]
    if int(m["copies"]) > 0:
        from .dataset import split_by_circuit
        train_idx, _ = split_by_circuit(read_rows(paths["train"]), d["val_circuits"])
        paths["materialized"] = materialize_train(
            paths["train"], np.asarray(train_idx), cache_dir / f"train_aug{int(m['copies'])}",
            cfg["augment"], paths["textures"] / "textures.npy", int(m["copies"]), int(m["seed"]),
            d["val_circuits"], force and run("materialize"), log=log)
    return paths


def main(argv=None) -> None:
    from ..config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="train.yaml")
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    ap.add_argument("--only", choices=STEPS, help="with --force: rebuild only this step")
    a = ap.parse_args(argv)
    paths = prepare(load_config(a.config), force=a.force, only=a.only)
    for k, v in paths.items():
        print(f"  {k:<13} {v}")


if __name__ == "__main__":
    main()
