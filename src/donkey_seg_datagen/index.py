"""Read the dataset layout: out/<split>/plan.json + out/<split>/<circuit_id>/meta.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from .io_utils import read_meta


def load_plan(root: str | Path, split: str) -> dict | None:
    p = Path(root) / split / "plan.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def splits(root: str | Path) -> list[str]:
    return sorted(p.parent.name for p in Path(root).glob("*/plan.json"))


def load_index(root: str | Path, split: str) -> list[dict]:
    """All saved frames of a split, plan order, with absolute 'image_path' / 'mask_path'."""
    plan = load_plan(root, split)
    if plan is None:
        return []
    out = []
    for c in plan["circuits"]:
        d = Path(root) / split / c["id"]
        for r in read_meta(d):
            out.append(dict(r, image_path=str(d / r["image_file"]), mask_path=str(d / r["mask_file"])))
    return out
