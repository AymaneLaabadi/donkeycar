"""Step 2: a bank of procedural noise textures for the background augmentation.

Generating good multi-octave noise per sample is the one slow augmentation ingredient, so it is
done once here and saved. At training time the background step only random-crops/flips a texture
from the bank and blends it in, which costs microseconds. With a 2x-larger-than-frame texture,
random crops, flips and the colour jitter applied afterwards, 128 textures give far more distinct
backgrounds than the model can memorise.

    cache/textures/textures.npy   uint8 K x Ht x Wt x 3
    cache/textures/info.json      the parameters; a change rebuilds the bank
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cache import publish_dir, read_json, write_json

TEXTURE_VERSION = 1
KINDS = ("fractal_rgb", "fractal_palette", "patches", "grain")


def _u(g: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand((), generator=g).item()


def fractal_noise(g: torch.Generator, h: int, w: int, channels: int = 1, octaves: int = 5,
                  persistence: float = 0.5, base: int = 4) -> torch.Tensor:
    """Value noise summed over octaves (coarse blobs + finer detail), normalised to [0,1]. C x H x W."""
    out = torch.zeros(1, channels, h, w)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh, gw = max(2, int(base * 2**o * h / max(h, w))), max(2, int(base * 2**o * w / max(h, w)))
        grid = torch.rand(1, channels, gh, gw, generator=g)
        out += amp * F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
        total += amp
        amp *= persistence
    out = out[0] / total
    lo = out.amin(dim=(1, 2), keepdim=True)
    hi = out.amax(dim=(1, 2), keepdim=True)
    return (out - lo) / (hi - lo).clamp_min(1e-6)


def make_texture(g: torch.Generator, h: int, w: int, kind: str) -> torch.Tensor:
    """3 x H x W float in [0,1]."""
    octaves = int(torch.randint(3, 7, (), generator=g))
    pers = _u(g, 0.35, 0.7)
    base = int(torch.randint(2, 9, (), generator=g))
    if kind == "fractal_rgb":          # independent colour noise per channel
        return fractal_noise(g, h, w, 3, octaves, pers, base)
    if kind == "fractal_palette":      # one noise field mapped between two random colours (grass, dirt...)
        t = fractal_noise(g, h, w, 1, octaves, pers, base)
        c0, c1 = torch.rand(3, 1, 1, generator=g), torch.rand(3, 1, 1, generator=g)
        return c0 + (c1 - c0) * t
    if kind == "patches":              # posterised noise: hard-edged patches of 3-6 colours
        t = fractal_noise(g, h, w, 1, octaves, pers, base)
        k = int(torch.randint(3, 7, (), generator=g))
        palette = torch.rand(k, 3, generator=g)
        idx = (t[0] * k).long().clamp_(max=k - 1)
        return palette[idx].permute(2, 0, 1)
    if kind == "grain":                # fine high-frequency grain around a random base colour
        base_c = torch.rand(3, 1, 1, generator=g)
        return (base_c + _u(g, 0.1, 0.35) * (torch.rand(3, h, w, generator=g) - 0.5)).clamp(0, 1)
    raise ValueError(kind)


def build_texture_bank(count: int, h: int, w: int, seed: int) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    bank = np.empty((count, h, w, 3), np.uint8)
    for i in range(count):
        tex = make_texture(g, h, w, KINDS[i % len(KINDS)])
        bank[i] = (tex.clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).numpy()
    return bank


def ensure_texture_bank(cache_dir: str | Path, count: int, size, seed: int, force: bool = False,
                        log=print) -> Path:
    out = Path(cache_dir) / "textures"
    params = {"version": TEXTURE_VERSION, "count": int(count), "size": [int(s) for s in size], "seed": int(seed)}
    info = read_json(out / "info.json")
    if not force and info and all(info.get(k) == v for k, v in params.items()):
        log(f"[textures] up to date ({count} x {size[0]}x{size[1]}) -> {out}")
        return out
    t0 = time.time()
    bank = build_texture_bank(count, size[0], size[1], seed)
    tmp = Path(cache_dir) / "textures.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / "textures.npy", bank)
    write_json(tmp / "info.json", {**params, "kinds": list(KINDS), "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    publish_dir(tmp, out)
    log(f"[textures] built {count} textures in {time.time() - t0:.1f}s -> {out}")
    return out
