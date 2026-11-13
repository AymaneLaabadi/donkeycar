"""Circuit plan of a split: drawn once from plan_seed, saved as out/<split>/plan.json.

For each circuit a coin with p = p_generated decides:
  generated -> new gen_track seed from the split's seed_range (never reused)
  existing  -> a random confirmed scene not yet used in this split (none left -> generated)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .io_utils import atomic_write_json


def draw_plan(cfg: dict, split: str) -> dict:
    s = cfg["splits"][split]
    rng = random.Random(int(s["plan_seed"]))
    lo, hi = (int(v) for v in s["seed_range"])
    unused = list(cfg["existing_scenes"])
    seeds: set[int] = set()
    circuits = []
    for k in range(int(s["circuits"])):
        if rng.random() < float(s["p_generated"]) or not unused:
            seed = rng.randrange(lo, hi)
            while seed in seeds:
                seed = rng.randrange(lo, hi)
            seeds.add(seed)
            c = {"type": "generated", "id": f"gen_s{seed}", "seed": seed, "scene": cfg["generated_scene"]}
        else:
            scene = unused.pop(rng.randrange(len(unused)))
            c = {"type": "existing", "id": scene, "seed": None, "scene": scene}
        circuits.append({"index": k, **c, "frames": int(s["frames"]), "frame_step": int(s["frame_step"])})
    return {"split": split, "plan_seed": int(s["plan_seed"]), "p_generated": float(s["p_generated"]),
            "seed_range": [lo, hi], "existing_scenes": list(cfg["existing_scenes"]), "circuits": circuits}


def train_scenes(cfg: dict) -> set[str]:
    """Existing scenes used in train (from out/train/plan.json, else the deterministic draw)."""
    p = Path(cfg["output"]["dir"]) / "train" / "plan.json"
    plan = json.loads(p.read_text(encoding="utf-8")) if p.exists() else draw_plan(cfg, "train")
    return {c["scene"] for c in plan["circuits"] if c["type"] == "existing"}


def load_or_create_plan(cfg: dict, split: str) -> dict:
    p = Path(cfg["output"]["dir"]) / split / "plan.json"
    if p.exists():
        plan = json.loads(p.read_text(encoding="utf-8"))
        saved = [{k: v for k, v in c.items() if k != "seen_in_train"} for c in plan["circuits"]]
        if saved != draw_plan(cfg, split)["circuits"]:
            print(f"note: {p} differs from what data_generation_config.yaml would draw now; using the saved plan "
                  "(delete it to redraw)")
        return plan
    plan = draw_plan(cfg, split)
    if split != "train":
        seen = train_scenes(cfg)
        for c in plan["circuits"]:
            c["seen_in_train"] = c["type"] == "existing" and c["scene"] in seen
    atomic_write_json(p, plan)
    return plan
