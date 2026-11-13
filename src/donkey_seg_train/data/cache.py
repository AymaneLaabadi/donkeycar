"""Step 1: decode every PNG once into flat uint8 arrays that training reads through a memory map.

Why: decoding 10 000 PNGs again in every epoch is pure waste (the data never changes), and opening
20 000 small files per epoch costs far more than reading slices of two big ones. After this step a
sample is one slice of a big array:

    cache/<split>/images.npy    uint8 N x H x W x 3   (RGB, exactly the decoded sim pixels)
    cache/<split>/masks.npy     uint8 N x H x W       ({0,1}; materialised copies may also hold 255)
    cache/<split>/meta.jsonl    one slim row per sample (circuit_id, frame, steering, ...)
    cache/<split>/info.json     shape, source fingerprint, per-channel mean/std, ...

`.npy` + `np.load(mmap_mode="r")` gives O(1) random access with no decoding, and DataLoader workers
share the OS page cache instead of each holding a copy. The source fingerprint (sizes + mtimes of
plan.json and every meta.jsonl) makes the step idempotent: it is skipped while the data is unchanged
and rebuilt automatically when you collect more frames.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from donkey_seg_datagen.index import load_index

CACHE_VERSION = 1


# --------------------------------------------------------------------------- shared helpers

def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_rows(split_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (Path(split_dir) / "meta.jsonl").read_text(encoding="utf-8").splitlines() if l]


def write_rows(split_dir: Path, rows: list[dict]) -> None:
    (Path(split_dir) / "meta.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def publish_dir(tmp: Path, final: Path) -> None:
    """Swap a fully written temp dir into place, so a crash never leaves a half-written cache.

    On Windows a file that is memory-mapped by another process (a running training or preview)
    cannot be deleted; say so instead of failing with a bare PermissionError halfway through rmtree.
    """
    try:
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    except PermissionError as e:
        raise PermissionError(
            f"cannot replace {final}: a file in it is still open ({e.filename}). Stop any running "
            f"training/preview that uses the cache and run prepare again.") from e


def channel_stats(images: np.ndarray, chunk: int = 512) -> tuple[list[float], list[float]]:
    """Per-channel mean/std in [0,1] units, streamed in chunks (never loads everything as float)."""
    s = np.zeros(3)
    ss = np.zeros(3)
    n = 0
    for i in range(0, len(images), chunk):
        x = images[i:i + chunk].reshape(-1, 3).astype(np.float64) / 255.0
        s += x.sum(0)
        ss += (x * x).sum(0)
        n += len(x)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean**2, 0.0))
    return mean.round(6).tolist(), std.round(6).tolist()


# --------------------------------------------------------------------------- step 1

def source_fingerprint(root: Path, split: str) -> str:
    files = [root / split / "plan.json", *sorted((root / split).glob("*/meta.jsonl"))]
    h = hashlib.sha1()
    for f in files:
        st = f.stat()
        h.update(f"{f.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _slim(i: int, r: dict) -> dict:
    """Fields training may want later (steering for a control head, cte for analysis, ...)."""
    tel, ctl = r.get("telemetry") or {}, r.get("control") or {}
    nan = float("nan")
    return {
        "idx": i,
        "circuit_id": r["circuit_id"],
        "circuit_type": r.get("circuit_type", ""),
        "frame": r["frame"],
        "steering": float(ctl.get("steering", nan)),
        "throttle": float(ctl.get("throttle", nan)),
        "speed": float(tel.get("speed", nan)),
        "cte": float(tel.get("cte", nan)) if tel.get("cte") is not None else nan,
        "road_frac": float(r.get("road_frac", nan)),
        "seen_in_train": bool(r.get("seen_in_train", False)),
    }


def build_split_cache(root: str | Path, split: str, cache_dir: str | Path, force: bool = False,
                      threads: int = 8, log=print) -> Path:
    root, cache_dir = Path(root), Path(cache_dir)
    out = cache_dir / split
    rows = load_index(root, split)
    if not rows:
        raise FileNotFoundError(f"no frames for split '{split}' under {root} (missing plan.json?)")
    fp = source_fingerprint(root, split)
    info = read_json(out / "info.json")
    if not force and info and info.get("version") == CACHE_VERSION and info.get("fingerprint") == fp:
        log(f"[cache] {split}: up to date ({info['n']} frames) -> {out}")
        return out

    t0 = time.time()
    first = np.asarray(Image.open(rows[0]["image_path"]).convert("RGB"))
    h, w = first.shape[:2]
    n = len(rows)
    log(f"[cache] {split}: decoding {n} frames ({w}x{h}) ...")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{split}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    images = np.lib.format.open_memmap(tmp / "images.npy", mode="w+", dtype=np.uint8, shape=(n, h, w, 3))
    masks = np.lib.format.open_memmap(tmp / "masks.npy", mode="w+", dtype=np.uint8, shape=(n, h, w))

    def load(i: int):
        r = rows[i]
        img = np.asarray(Image.open(r["image_path"]).convert("RGB"))
        m = np.asarray(Image.open(r["mask_path"]))
        if img.shape != (h, w, 3) or m.shape != (h, w):
            raise ValueError(f"{r['image_path']}: shape {img.shape}/{m.shape}, expected {(h, w)}")
        if m.max() > 1:
            raise ValueError(f"{r['mask_path']}: mask values must be in {{0,1}}, found max {m.max()}")
        return i, img, m

    # PNG decoding releases the GIL, so threads are enough (no process start-up cost on Windows).
    with ThreadPoolExecutor(threads) as ex:
        for k, (i, img, m) in enumerate(ex.map(load, range(n)), 1):
            images[i] = img
            masks[i] = m
            if k % 2000 == 0:
                log(f"[cache] {split}: {k}/{n}")

    mean, std = channel_stats(images)
    road = float(masks.mean())
    circuits: dict[str, int] = {}
    for r in rows:
        circuits[r["circuit_id"]] = circuits.get(r["circuit_id"], 0) + 1
    images.flush()
    masks.flush()
    del images, masks  # close the memmaps; Windows cannot rename a dir with open files

    write_rows(tmp, [_slim(i, r) for i, r in enumerate(rows)])
    write_json(tmp / "info.json", {
        "version": CACHE_VERSION, "kind": "decoded", "split": split, "source_root": str(root),
        "fingerprint": fp, "n": n, "h": h, "w": w, "mean": mean, "std": std,
        "road_fraction": round(road, 6), "circuits": circuits,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    publish_dir(tmp, out)
    log(f"[cache] {split}: done in {time.time() - t0:.1f}s -> {out}")
    return out
