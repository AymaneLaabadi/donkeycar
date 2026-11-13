"""Mask decoding / validation and overlay drawing (shared by collect, verify, preview)."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

EXT = {"JPG": "jpg", "PNG": "png"}
MAGIC = {"PNG": b"\x89PNG", "JPG": b"\xff\xd8"}


class MaskError(Exception):
    pass


def decode_seg(b64: str) -> np.ndarray:
    """Decode the fork's image_seg (RGB PNG, R = class id, G = B = 0) -> HxW uint8 in {0,1}."""
    arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.any(arr[:, :, 1:3]):
            raise MaskError("image_seg G/B channels are not all 0")
        cls = arr[:, :, 0]
    elif arr.ndim == 2:
        cls = arr
    else:
        raise MaskError(f"unexpected image_seg shape {arr.shape}")
    u = np.unique(cls)
    if not set(u.tolist()) <= {0, 1}:
        raise MaskError(f"image_seg values {u.tolist()} not a subset of {{0,1}}")
    return np.ascontiguousarray(cls, dtype=np.uint8)


def road_edge(cls: np.ndarray) -> np.ndarray:
    """Outermost road pixels: road pixels with a 4-neighbour that is background (as check_seg.py)."""
    road = cls == 1
    edge = np.zeros_like(road)
    dv = road[:-1, :] != road[1:, :]
    dh = road[:, :-1] != road[:, 1:]
    edge[:-1, :] |= dv
    edge[1:, :] |= dv
    edge[:, :-1] |= dh
    edge[:, 1:] |= dh
    return edge & road


def overlay(rgb: np.ndarray, cls: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    road = cls == 1
    out = rgb[:, :, :3].astype(np.float32).copy()
    out[road] = (1 - alpha) * out[road] + alpha * np.array([0, 255, 0], np.float32)
    out[road_edge(cls)] = (255, 255, 0)
    return out.clip(0, 255).astype(np.uint8)
