"""Preview grid of random samples: RGB | overlay (green road, yellow outermost road pixels).

    python -m donkey_seg_datagen.preview --split train -n 16 --out preview_train.png
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

from .index import load_index
from .masks import overlay


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="out")
    ap.add_argument("--split", required=True)
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: preview_<split>.png")
    args = ap.parse_args(argv)

    idx = load_index(args.root, args.split)
    if not idx:
        print(f"no frames in {args.root}/{args.split}")
        return 2
    pick = random.Random(args.seed).sample(idx, min(args.n, len(idx)))
    tiles = []
    for r in pick:
        rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
        m = np.asarray(Image.open(r["mask_path"]))
        pair = np.concatenate([rgb, np.full((rgb.shape[0], 2, 3), 255, np.uint8), overlay(rgb, m)], axis=1)
        im = Image.fromarray(pair).resize((pair.shape[1] * args.scale, pair.shape[0] * args.scale), Image.NEAREST)
        tile = Image.new("RGB", (im.width, im.height + 18), (20, 20, 20))
        ImageDraw.Draw(tile).text((4, 3), f"{r['circuit_id']} #{r['frame']}  road={r['road_frac']:.2f}",
                                  fill=(230, 230, 230))
        tile.paste(im, (0, 18))
        tiles.append(tile)

    cols = min(args.cols, len(tiles))
    rows = -(-len(tiles) // cols)
    tw, th, pad = tiles[0].width, tiles[0].height, 6
    grid = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + pad) + pad), (60, 60, 60))
    for i, t in enumerate(tiles):
        grid.paste(t, (pad + (i % cols) * (tw + pad), pad + (i // cols) * (th + pad)))
    out = args.out or f"preview_{args.split}.png"
    grid.save(out)
    print(f"wrote {out}: {len(tiles)} samples from {len({r['circuit_id'] for r in pick})} circuits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
