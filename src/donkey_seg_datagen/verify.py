"""Dataset checks for every split under --root (or --split ...).

    python -m donkey_seg_datagen.verify [--root out] [--split train]

Errors (-> FAIL): missing files, mask not single-channel or values outside {0,1}, image/mask size
mismatch, image size != camera config, road_frac in meta != mask, files without meta rows,
a gen_track seed used in two splits, wrong seen_in_train flag.
Reported: frames per circuit vs target, generated/existing ratio, road-fraction histogram.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .index import load_plan, splits
from .io_utils import read_meta


def check_split(root: Path, split: str, errors: list[str]) -> list[float]:
    plan = load_plan(root, split)
    man_p = root / split / "manifest.json"
    cam = json.loads(man_p.read_text(encoding="utf-8"))["config"]["camera"] if man_p.exists() else None
    fracs: list[float] = []
    by_type = defaultdict(lambda: [0, 0])          # type -> [circuits, frames]
    print(f"\n=== {split} ({len(plan['circuits'])} circuits, plan_seed {plan['plan_seed']}) ===")
    print(f"  {'#':>2} {'circuit':24s} {'type':10s} {'frames':>12s}  status")
    for c in plan["circuits"]:
        d = root / split / c["id"]
        rows = read_meta(d)
        listed = set()
        for r in rows:
            name = f"{split}/{c['id']}/{r['frame']:06d}"
            ip, mp = d / r["image_file"], d / r["mask_file"]
            listed |= {r["image_file"], r["mask_file"]}
            if not ip.exists() or not mp.exists():
                errors.append(f"{name}: missing image or mask")
                continue
            m_im, im = Image.open(mp), Image.open(ip)
            m = np.asarray(m_im)
            if m_im.mode != "L":
                errors.append(f"{name}: mask mode {m_im.mode}, expected L")
            if not set(np.unique(m).tolist()) <= {0, 1}:
                errors.append(f"{name}: mask values {np.unique(m).tolist()} not in {{0,1}}")
            if im.size != (m.shape[1], m.shape[0]):
                errors.append(f"{name}: image {im.size} != mask {m.shape[1]}x{m.shape[0]}")
            if cam and im.size != (int(cam["img_w"]), int(cam["img_h"])):
                errors.append(f"{name}: image {im.size} != camera {cam['img_w']}x{cam['img_h']}")
            f = float(m.mean())
            if abs(f - r["road_frac"]) > 1e-5:
                errors.append(f"{name}: meta road_frac {r['road_frac']} != mask {f:.6f}")
            if "seen_in_train" in c and r.get("seen_in_train") != c["seen_in_train"]:
                errors.append(f"{name}: seen_in_train {r.get('seen_in_train')} != plan {c['seen_in_train']}")
            fracs.append(f)
        if d.exists():
            on_disk = {f"{s}/{p.name}" for s in ("images", "masks") for p in (d / s).glob("*")}
            if on_disk - listed:
                errors.append(f"{split}/{c['id']}: {len(on_disk - listed)} files without meta rows")
        status = "skipped" if (d / "SKIPPED.json").exists() else \
            "done" if len(rows) >= c["frames"] else "partial" if rows else "todo"
        seen = f"  seen_in_train={c['seen_in_train']}" if "seen_in_train" in c else ""
        print(f"  {c['index'] + 1:2d} {c['id']:24s} {c['type']:10s} {len(rows):5d} / {c['frames']:<5d}  {status}{seen}")
        by_type[c["type"]][0] += 1
        by_type[c["type"]][1] += len(rows)
    n_c, n_f = len(plan["circuits"]), max(sum(v[1] for v in by_type.values()), 1)
    print("  ratio: " + ", ".join(f"{t} {v[0]}/{n_c} circuits ({100 * v[0] / n_c:.0f}%), {v[1]} frames "
                                  f"({100 * v[1] / n_f:.0f}%)" for t, v in sorted(by_type.items())))
    print(f"  frames: {sum(v[1] for v in by_type.values())} / {sum(c['frames'] for c in plan['circuits'])} target")
    return fracs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out")
    ap.add_argument("--split", action="append", help="default: every split with a plan.json")
    args = ap.parse_args(argv)
    root = Path(args.root)
    names = args.split or splits(root)
    if not names:
        print(f"no plan.json under {root}")
        return 2
    print(f"dataset: {root.resolve()}")
    errors: list[str] = []
    fracs: list[float] = []
    for s in names:
        fracs += check_split(root, s, errors)

    seed_owner = defaultdict(set)                   # seeds are never reused across splits
    for s in splits(root):
        for c in load_plan(root, s)["circuits"]:
            if c["type"] == "generated":
                seed_owner[c["seed"]].add(s)
    for seed, owners in seed_owner.items():
        if len(owners) > 1:
            errors.append(f"seed {seed} used in splits {sorted(owners)}")
    print(f"\ngen_track seeds: {len(seed_owner)} across {splits(root)}; "
          f"overlap between splits: {sum(len(o) > 1 for o in seed_owner.values())}")

    if fracs:
        f = np.array(fracs)
        print(f"\nroad fraction ({len(f)} frames): min {f.min():.3f}  median {np.median(f):.3f}  "
              f"mean {f.mean():.3f}  max {f.max():.3f}")
        hist, edges = np.histogram(f, bins=10, range=(0, 1))
        for h, a, b in zip(hist, edges[:-1], edges[1:]):
            print(f"  {a:.1f}-{b:.1f} {h:6d} {'#' * round(40 * h / max(hist.max(), 1))}")

    print(f"\nERRORS: {len(errors)}")
    for e in errors[:30]:
        print("  ", e)
    if len(errors) > 30:
        print(f"   ... {len(errors) - 30} more")
    print("RESULT:", "PASS" if not errors else "FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
