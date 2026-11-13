"""Atomic file writes and meta.jsonl reading."""

from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1).encode("utf-8"))


def read_meta(circuit_dir: str | Path) -> list[dict]:
    """Complete rows of meta.jsonl (a torn last line is ignored)."""
    p = Path(circuit_dir) / "meta.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                break
    return rows


def write_meta(circuit_dir: str | Path, rows: list[dict]) -> None:
    data = "".join(json.dumps(r) + "\n" for r in rows)
    atomic_write_bytes(Path(circuit_dir) / "meta.jsonl", data.encode("utf-8"))
