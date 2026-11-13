"""Train the road-segmentation model.

    python -m donkey_seg_train.train                                  # with augmentation
    python -m donkey_seg_train.train --no-augment                     # same run, no augmentation
    python -m donkey_seg_train.train --resume runs/<run>/last.pt      # continue an interrupted run
    python -m donkey_seg_train.train --resume runs/<run>/last.pt --epochs 50   # train it longer

A run folder runs/<name>/ holds config.json, log.csv, last.pt (full state, for resuming) and
best.pt (weights with the best validation IoU).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import IGNORE_INDEX, build_dataloaders
from .losses import BCEDiceLoss
from .model import SmallUNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     # CPU and every CUDA device
    torch.backends.cudnn.benchmark = False      # benchmark picks kernels at random-ish -> off
    torch.backends.cudnn.deterministic = True


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    """LR factor per optimizer step: linear 0 -> 1 during warmup, then cosine 1 -> 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device) -> float:
    model.train()
    total, n = 0.0, 0
    for img, mask in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        loss, _ = criterion(model(img), mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()                         # per step, so the warmup is smooth within epoch 1
        total += loss.item() * len(img)
        n += len(img)
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """Mean loss and road IoU over the whole set (ignored pixels excluded)."""
    model.eval()
    loss_sum, inter, union, n = 0.0, 0, 0, 0
    for img, mask in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        logits = model(img)
        loss_sum += criterion(logits, mask)[0].item() * len(img)
        n += len(img)
        pred, road, valid = logits[:, 0] > 0, mask == 1, mask != IGNORE_INDEX
        inter += (pred & road & valid).sum().item()
        union += ((pred | road) & valid).sum().item()
    return {"loss": loss_sum / n, "iou": inter / max(union, 1)}


def save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)                        # never leaves a half-written checkpoint


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="train.yaml")
    ap.add_argument("--no-augment", action="store_true", help="train on the raw frames")
    ap.add_argument("--epochs", type=int, help="total epochs (overrides the config; with --resume: extend)")
    ap.add_argument("--name", help="run folder name (default: <time>_aug / <time>_noaug)")
    ap.add_argument("--resume", type=Path, help="last.pt of the run to continue")
    a = ap.parse_args(argv)

    # --- config and run folder --------------------------------------------------------------
    ckpt = None
    if a.resume:
        if a.no_augment:
            ap.error("--no-augment cannot change a resumed run; it keeps its saved config")
        ckpt = torch.load(a.resume, map_location="cpu")
        cfg, run_dir = ckpt["cfg"], a.resume.parent
    else:
        cfg = load_config(a.config)
        if a.no_augment:
            cfg["augment"]["enabled"] = False
            cfg["materialize"]["copies"] = 0
        tag = "aug" if cfg["augment"]["enabled"] else "noaug"
        run_dir = Path(cfg["train"]["out_dir"]) / (a.name or f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}")
        run_dir.mkdir(parents=True, exist_ok=False)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    t, seed = cfg["train"], cfg["data"]["seed"]

    # --- everything that uses randomness is created after seeding -----------------------------
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(cfg)
    model = SmallUNet(base=t["model_base"]).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps = len(loaders["train"])
    scheduler = warmup_cosine(optimizer, t["warmup_epochs"] * steps, t["epochs"] * steps)

    start, best_iou = 1, -1.0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])   # keeps the step count; the cosine uses the new total
        start, best_iou = ckpt["epoch"] + 1, ckpt["best_iou"]

    print(f"run {run_dir} | device {device} | augment {cfg['augment']['enabled']} | "
          f"epochs {start}..{t['epochs']} | {steps} steps/epoch")
    log_path = run_dir / "log.csv"
    for epoch in range(start, t["epochs"] + 1):
        # Shuffle order depends only on (seed, epoch): identical whether the run was resumed or not.
        loaders["train"].generator.manual_seed(seed + epoch)
        t0, lr = time.time(), scheduler.get_last_lr()[0]   # LR this epoch starts with
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, device)
        val = evaluate(model, loaders["val"], criterion, device)
        row = {"epoch": epoch, "lr": f"{lr:.3e}", "train_loss": f"{train_loss:.4f}",
               "val_loss": f"{val['loss']:.4f}", "val_iou": f"{val['iou']:.4f}", "time_s": f"{time.time() - t0:.0f}"}
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        print("  ".join(f"{k} {v}" for k, v in row.items()))

        if val["iou"] > best_iou:
            best_iou = val["iou"]
            save(run_dir / "best.pt", {"model": model.state_dict(), "epoch": epoch, "val_iou": best_iou, "cfg": cfg})
        save(run_dir / "last.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_iou": best_iou, "cfg": cfg,
        })

    if "test" in loaders and (run_dir / "best.pt").exists():
        best = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(best["model"])
        test = evaluate(model, loaders["test"], criterion, device)
        print(f"test (best.pt, epoch {best['epoch']}): loss {test['loss']:.4f}  iou {test['iou']:.4f}")
        (run_dir / "test.json").write_text(json.dumps({"epoch": best["epoch"], **test}, indent=1), encoding="utf-8")


if __name__ == "__main__":                       # required on Windows (DataLoader workers re-import this file)
    main()
