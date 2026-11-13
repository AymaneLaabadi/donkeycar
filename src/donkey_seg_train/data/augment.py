"""Online augmentation: Geometry (image + mask together, mild) then Appearance (image only, strong).

Everything is plain torch on one CHW float sample, so it runs inside DataLoader workers and uses
only torch's RNG. That matters: DataLoader seeds torch (and `random`) differently in every worker and
every epoch, but NOT numpy's global RNG. Mixing numpy randomness into worker code is the classic
bug where all workers produce identical "random" augmentations.

Pipeline order and why:
  1. geometry      hflip, then one affine (shift/rotate/scale). Image bilinear, mask NEAREST
                   (never interpolate labels). Pixels pulled in from outside the frame get
                   IGNORE_INDEX in the mask: we don't know what is there, so the loss must skip them.
  2. background    noise texture blended into mask==0 pixels (needs the post-geometry mask).
  3. shadows       darken random soft polygons over road and background alike.
  4. colour        brightness / contrast / gamma / hue / saturation. Applied AFTER the background
                   texture and shadows, so those get the same colour cast as the rest of the frame
                   (otherwise "different colour statistics" would itself reveal the background).
  5. gray/shuffle  rare; forces shape cues over colour cues.
  6. sensor noise  mild gaussian noise over the WHOLE image (see the background note below).
  7. cutout        occluding holes on the image only; the mask is unchanged, so the model learns to
                   infer road it can't fully see.

Background-noise caveat (shortcut learning): if noise only ever appears on background, "noisy =
not road" becomes a free cue. That cue never exists at test time, so it would not break inference,
but it wastes capacity and can make the model over-trust "clean = road". Mitigations used: it is
applied to only ~30% of samples, the mask edge is feathered (no pixel-exact hint of the boundary),
and mild sensor noise is also added to the whole image in other samples.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

IGNORE_INDEX = 255


@dataclass
class AugmentConfig:
    enabled: bool = True
    # --- geometry (image + mask) ------------------------------------------------------------
    hflip_p: float = 0.5
    affine_p: float = 0.7
    shift: float = 0.05                      # fraction of width/height, +-
    rotate_deg: float = 5.0                  # +-
    scale: tuple = (0.9, 1.1)
    # --- appearance (image only) ------------------------------------------------------------
    color_p: float = 0.8                     # probability of the brightness/contrast/gamma/hue/sat block
    brightness: float = 0.25                 # multiply by U(1-b, 1+b)
    contrast: float = 0.25                   # blend around the mean gray, factor U(1-c, 1+c)
    gamma: tuple = (0.7, 1.5)                # sampled log-uniformly (as many darkening as brightening)
    hue_deg: float = 20.0                    # +- degrees on the 360 deg hue circle
    saturation: float = 0.3                  # multiply S by U(1-s, 1+s)
    grayscale_p: float = 0.05
    channel_shuffle_p: float = 0.05
    shadow_p: float = 0.3
    shadow_count: tuple = (1, 2)
    shadow_strength: tuple = (0.3, 0.6)      # fraction of light removed inside the shadow
    background_p: float = 0.3
    background_alpha: tuple = (0.3, 1.0)     # texture opacity on background pixels
    background_feather: int = 5              # px, odd; softens the mask edge of the blend
    sensor_noise_p: float = 0.2
    sensor_noise_std: tuple = (0.005, 0.03)
    cutout_p: float = 0.5
    cutout_holes: tuple = (1, 4)
    cutout_size: tuple = (0.05, 0.15)        # hole side as a fraction of the image side

    @classmethod
    def from_dict(cls, d: dict | None) -> "AugmentConfig":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown augment keys: {sorted(unknown)}")
        cfg = cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
        if cfg.background_feather % 2 == 0:
            raise ValueError("background_feather must be odd")
        return cfg

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


# --------------------------------------------------------------------------- random helpers (torch RNG only)

def _rand() -> float:
    return torch.rand(()).item()


def _u(lo: float, hi: float) -> float:
    return lo + (hi - lo) * _rand()


def _randint(lo: int, hi: int) -> int:
    """Inclusive on both ends."""
    return int(torch.randint(lo, hi + 1, ()).item())


def _box_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """Box blur of a HxW or CxHxW map, edges handled by replicate padding."""
    if k <= 1:
        return x
    squeeze = x.dim() == 2
    x = x[None, None] if squeeze else x[None]
    x = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="replicate"), k, stride=1)
    return x[0, 0] if squeeze else x[0]


# --------------------------------------------------------------------------- colour space

def rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img
    maxc, _ = img.max(0)
    minc, _ = img.min(0)
    delta = maxc - minc
    s = delta / maxc.clamp_min(1e-8)
    d = delta.clamp_min(1e-8)
    rc, gc, bc = (maxc - r) / d, (maxc - g) / d, (maxc - b) / d
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(delta > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    return torch.stack([h, s, maxc])


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.long() % 6
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    idx = i[None]
    r = torch.stack([v, q, p, p, t, v]).gather(0, idx)[0]
    g = torch.stack([t, v, v, q, p, p]).gather(0, idx)[0]
    b = torch.stack([p, p, t, v, v, q]).gather(0, idx)[0]
    return torch.stack([r, g, b])


def gray(img: torch.Tensor) -> torch.Tensor:
    """ITU-R 601 luma, 1xHxW."""
    return (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])[None]


# --------------------------------------------------------------------------- geometry (image + mask)

def hflip(img: torch.Tensor, mask: torch.Tensor):
    return img.flip(-1), mask.flip(-1)


def affine(img: torch.Tensor, mask: torch.Tensor, shift_xy, angle_deg: float, scale: float):
    """One combined resample (resampling twice would blur twice). Mask: nearest + ignore outside."""
    _, h, w = img.shape
    a = math.radians(angle_deg)
    # Forward map in pixel units about the centre: p' = s * R p + t.  affine_grid works in normalised
    # [-1,1] coords and needs the INVERSE map (output -> input). Converting through pixel units keeps
    # the rotation a true rotation on a non-square (160x120) image.
    S = torch.tensor([[w / 2, 0.0], [0.0, h / 2]])
    R = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * scale
    M = torch.linalg.inv(S) @ R @ S
    t = torch.tensor([2.0 * shift_xy[0], 2.0 * shift_xy[1]])
    Minv = torch.linalg.inv(M)
    theta = torch.cat([Minv, (-Minv @ t)[:, None]], 1)[None]
    grid = F.affine_grid(theta, (1, 3, h, w), align_corners=False)
    img = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    m = F.grid_sample(mask[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]
    valid = F.grid_sample(torch.ones(1, 1, h, w), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]
    m = m.long()
    m[valid < 0.5] = IGNORE_INDEX
    return img.contiguous(), m


# --------------------------------------------------------------------------- appearance (image only)

def adjust_colour(img: torch.Tensor, c: AugmentConfig) -> torch.Tensor:
    img = img * _u(1 - c.brightness, 1 + c.brightness)
    m = gray(img).mean()
    img = ((img - m) * _u(1 - c.contrast, 1 + c.contrast) + m).clamp(0, 1)
    g = math.exp(_u(math.log(c.gamma[0]), math.log(c.gamma[1])))
    img = img.clamp_min(1e-6) ** g
    if c.hue_deg > 0 or c.saturation > 0:
        hsv = rgb_to_hsv(img)
        hsv[0] = (hsv[0] + _u(-c.hue_deg, c.hue_deg) / 360.0) % 1.0
        hsv[1] = (hsv[1] * _u(1 - c.saturation, 1 + c.saturation)).clamp(0, 1)
        img = hsv_to_rgb(hsv)
    return img.clamp(0, 1)


def random_shadow(img: torch.Tensor, c: AugmentConfig) -> torch.Tensor:
    _, h, w = img.shape
    shade = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(shade)
    for _ in range(_randint(*c.shadow_count)):
        # A random star-shaped polygon, large enough to cross the road like a tree/building shadow.
        cx, cy = _u(0, w), _u(0, h)
        n = _randint(3, 6)
        angles = sorted(_u(0, 2 * math.pi) for _ in range(n))
        rad = [_u(0.2, 0.9) * w for _ in range(n)]
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for r, a in zip(rad, angles)]
        draw.polygon(pts, fill=int(255 * _u(*c.shadow_strength)))
    s = torch.from_numpy(np.asarray(shade, dtype=np.float32) / 255.0)
    s = _box_blur(s, 2 * _randint(1, 4) + 1)   # soft penumbra
    return img * (1.0 - s)[None]


class TextureBank:
    """Lazily memory-maps cache/textures/textures.npy (cheap to pickle into spawned workers)."""

    def __init__(self, path: str | Path | None):
        self.path = None if path is None else str(path)
        self._arr = None

    def __bool__(self) -> bool:
        return self.path is not None

    def sample(self, h: int, w: int) -> torch.Tensor:
        if self._arr is None:
            self._arr = np.load(self.path, mmap_mode="r")
        k = _randint(0, len(self._arr) - 1)
        th, tw = self._arr.shape[1:3]
        if th < h or tw < w:
            raise ValueError(f"texture bank is {th}x{tw} but frames are {h}x{w}; "
                             f"set textures.size >= [{h}, {w}] (2x recommended) in train.yaml")
        y, x = _randint(0, th - h), _randint(0, tw - w)
        tex = torch.from_numpy(np.array(self._arr[k, y:y + h, x:x + w])).permute(2, 0, 1).float() / 255.0
        if _rand() < 0.5:
            tex = tex.flip(-1)
        if _rand() < 0.5:
            tex = tex.flip(-2)
        return tex


def background_noise(img: torch.Tensor, mask: torch.Tensor, bank: TextureBank, c: AugmentConfig) -> torch.Tensor:
    _, h, w = img.shape
    bg = (mask != 1).float()                  # background AND ignore pixels
    bg = _box_blur(bg, c.background_feather)  # feathered edge, no pixel-exact boundary hint
    a = _u(*c.background_alpha) * bg[None]
    tex = bank.sample(h, w)
    if _rand() < 0.5:   # replace: background looks like a different ground material
        out = img * (1 - a) + tex * a
    else:               # perturb: keeps the scene's structure but adds texture on top
        out = img + (tex - tex.mean(dim=(1, 2), keepdim=True)) * a
    return out.clamp(0, 1)


def cutout(img: torch.Tensor, c: AugmentConfig) -> torch.Tensor:
    img = img.clone()
    _, h, w = img.shape
    for _ in range(_randint(*c.cutout_holes)):
        hh = max(1, int(_u(*c.cutout_size) * h))
        ww = max(1, int(_u(*c.cutout_size) * w))
        y, x = _randint(0, h - hh), _randint(0, w - ww)
        r = _rand()
        if r < 0.33:
            fill = torch.zeros(3, 1, 1)
        elif r < 0.66:
            fill = torch.rand(3, 1, 1)              # flat random colour
        else:
            fill = torch.rand(3, hh, ww)            # noise
        img[:, y:y + hh, x:x + ww] = fill
    return img


# --------------------------------------------------------------------------- the pipeline

def to_tensors(img_u8: np.ndarray, mask_u8: np.ndarray):
    """HxWx3 uint8 + HxW uint8 -> float32 3xHxW in [0,1] + int64 HxW.

    np.array() copies out of the read-only memmap (torch refuses to wrap non-writable memory safely).
    """
    img = torch.from_numpy(np.array(img_u8)).permute(2, 0, 1).float().div_(255.0)
    mask = torch.from_numpy(np.array(mask_u8)).long()
    return img, mask


class SegAugment:
    """callable(img_u8 HxWx3, mask_u8 HxW) -> (img float 3xHxW [0,1], mask int64 HxW, params dict).

    `params` records what was done (e.g. hflip) so the dataset can keep side labels consistent:
    a flipped frame has steering -> -steering.
    """

    def __init__(self, cfg: AugmentConfig | dict | None = None, textures: str | Path | None = None):
        self.cfg = cfg if isinstance(cfg, AugmentConfig) else AugmentConfig.from_dict(cfg)
        self.bank = TextureBank(textures)

    def __call__(self, img_u8: np.ndarray, mask_u8: np.ndarray):
        img, mask = to_tensors(img_u8, mask_u8)
        c = self.cfg
        params = {"hflip": False}
        if not c.enabled:
            return img, mask, params
        _, h, w = img.shape

        # 1. geometry --------------------------------------------------------------------
        if _rand() < c.hflip_p:
            img, mask = hflip(img, mask)
            params["hflip"] = True
        if _rand() < c.affine_p:
            img, mask = affine(img, mask, (_u(-c.shift, c.shift), _u(-c.shift, c.shift)),
                               _u(-c.rotate_deg, c.rotate_deg), _u(*c.scale))

        # 2-7. appearance -------------------------------------------------------------------
        if self.bank and _rand() < c.background_p:
            img = background_noise(img, mask, self.bank, c)
        if _rand() < c.shadow_p:
            img = random_shadow(img, c)
        if _rand() < c.color_p:
            img = adjust_colour(img, c)
        r = _rand()
        if r < c.grayscale_p:
            img = gray(img).expand(3, h, w).contiguous()
        elif r < c.grayscale_p + c.channel_shuffle_p:
            img = img[torch.randperm(3)]
        if _rand() < c.sensor_noise_p:
            img = (img + torch.randn_like(img) * _u(*c.sensor_noise_std)).clamp(0, 1)
        if _rand() < c.cutout_p:
            img = cutout(img, c)
        return img.contiguous(), mask, params
