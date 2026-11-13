"""Load and check data_generation_config.yaml.

Relative paths (output.dir, sim.exe_path) are resolved against the YAML file's folder, so moving the
whole project folder does not break them (e.g. exe_path: ../Builds/Segmentation/donkey_sim.exe).
"""

from __future__ import annotations

from pathlib import Path

import yaml

CAMERA_KEYS = ("img_w", "img_h", "img_d", "img_enc", "fov", "offset_x", "offset_y", "offset_z",
               "rot_x", "rot_y", "rot_z", "fish_eye_x", "fish_eye_y")


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg["output"]["dir"])
    cfg["output"]["dir"] = str(out if out.is_absolute() else path.parent / out)
    exe = (cfg.get("sim") or {}).get("exe_path")
    if exe:
        exe = Path(exe).expanduser()
        cfg["sim"]["exe_path"] = str(exe if exe.is_absolute() else (path.parent / exe).resolve())
    cfg["_config_path"] = str(path)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    cam = cfg["camera"]
    missing = [k for k in CAMERA_KEYS if k not in cam]
    if missing:
        raise SystemExit(f"config: camera is missing {missing}")
    if float(cam["fish_eye_x"]) != 0 or float(cam["fish_eye_y"]) != 0:
        raise SystemExit("config: fish_eye_x/y must be 0 (the segmentation mask is rendered without fisheye)")
    if str(cam["img_enc"]).upper() not in ("PNG", "JPG"):
        raise SystemExit("config: camera.img_enc must be PNG or JPG")

    splits = cfg["splits"]
    for name, s in splits.items():
        lo, hi = s["seed_range"]
        if not (0 <= lo < hi):
            raise SystemExit(f"config: splits.{name}.seed_range must be [lo, hi) with 0 <= lo < hi")
        if not 0 <= float(s["p_generated"]) <= 1:
            raise SystemExit(f"config: splits.{name}.p_generated must be in [0, 1]")
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (alo, ahi), (blo, bhi) = splits[a]["seed_range"], splits[b]["seed_range"]
            if alo < bhi and blo < ahi:
                raise SystemExit(f"config: seed_range of {a} and {b} overlap (seeds must never be reused)")
