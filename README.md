# DonkeyCar road segmentation

Two packages in one project:

* **`donkey_seg_datagen`**: manual-driving data collector for the DonkeyCar Unity sim fork with the
  segmentation camera (see `sdsandbox/docs/SEGMENTATION.md`). You drive; every saved sample is
  the RGB **exactly as the car camera sends it** (the received `image` bytes, written verbatim) and
  the **drivable-road mask of the same frame** (single-channel uint8 PNG, 0 background / 1 road).
* **`donkey_seg_train`**: training side. Cached data pipeline with online augmentation and PyTorch
  DataLoaders (done); model, losses and training loop (placeholders, next step).

## Layout

```
DonkeyCar Project/
  Builds/Segmentation/donkey_sim.exe     the sim build (sim.exe_path points here)
  gym-donkeycar/                         local clone of gym-donkeycar 1.3.1 (optional)
  sdsandbox/                             sim source
  DonkeyCar/                             this repo
    data_generation_config.yaml          collector config
    train.yaml                           training config (overrides only)
    generated_data/                      collected frames (git-ignored)
    cache/                               prepared training data (git-ignored, safe to delete)
    src/donkey_seg_datagen/              collector
    src/donkey_seg_train/                training
```

## Install

```
uv sync --extra torch    # everything (torch is needed for donkey_seg_train)
uv sync                  # collector only: numpy, pillow, pyyaml, pygame-ce, gym-donkeycar 1.3.1
```

gym-donkeycar 1.3.x is not on PyPI, so `pyproject.toml` pulls it from GitHub (pinned commit,
see `[tool.uv.sources]`). Plain `pip install -e .` ignores that section; with pip, install
gym-donkeycar from the local clone first (`pip install -e ../gym-donkeycar`).
On Windows the PyPI torch wheel is CPU-only; for a GPU, point torch at the CUDA wheel index.

Prefix the commands below with `uv run` (or activate `.venv`).

## 1. Collect data

Extended telemetry must be ON in the sim (default). If the collector says `cte`/`pos_x` are missing:
sim main menu, settings panel, tick `extendedTelemetry`, press Save.

```
python -m donkey_seg_datagen.collect --split train      # drive; rerun the same command to resume
python -m donkey_seg_datagen.collect --split test       # later, independently
python -m donkey_seg_datagen.verify                     # all splits (or --split train)
python -m donkey_seg_datagen.preview --split train -n 16
```

(`dsg-collect`, `dsg-verify`, `dsg-preview` do the same.) The config defaults to
`data_generation_config.yaml`; pass `--config other.yaml` for another one. Relative paths in it
(`output.dir`, `sim.exe_path`) are resolved against the YAML's folder, so moving the project folder
does not break them. With `sim.launch: auto` the collector uses a sim already listening on
host:port, otherwise it starts `sim.exe_path` and closes it at the end.

### Keys

| key | action |
|---|---|
| arrows / WASD | steer, throttle / reverse (smoothed) |
| gamepad | left stick X steer, RT throttle, LT brake |
| SPACE | start / pause recording (each circuit starts **paused**) |
| R | `reset_car`, then discard frames for `discard_s` |
| BACKSPACE | delete the frames saved in the last 3 s (after a crash) |
| M | mask overlay on / off (40 % green) |
| N | skip this circuit (Y to confirm); it is marked `SKIPPED.json` and not revisited |
| ESC | save and quit; rerun to continue |

The panel shows circuit k/N, track (seed or scene), saved/target frames, speed and recording state
(REC / PAUSED / DISCARD). When a circuit reaches its target the car brakes, "done" is shown, and the
next circuit is loaded.

### How the plan is drawn

At the first run of a split the full plan is drawn from `splits.<split>.plan_seed` and saved as
`generated_data/<split>/plan.json`; later runs reuse that file (delete it to redraw). For each circuit:

* with probability `p_generated`: a **generated track**, `gen_track` with a new seed drawn from the
  split's `seed_range` (train and test ranges must not overlap, checked at load time and by verify);
* otherwise an **existing scene** from `existing_scenes`, not yet used in this split. If all are used,
  it falls back to a generated track.

`existing_scenes` is currently `generated_track, mini_monaco, mountain_track` (checked with
`tools/check_seg.py`; add `warren` once it passes). Change the list *before* a split's first run,
because a saved plan is not redrawn.

Test circuits get `seen_in_train`: true for an existing scene that train also used. Generated
test tracks are never seen.

### Recording rules

* Frames are discarded for `sim.discard_s` (1.0 s) after `load_scene`, `gen_track` and `reset_car`.
* A frame is saved only while recording is on, at least `frame_step` telemetry frames after the
  previous save, with `hit == "none"` and `speed >= 0.3`.
* `image_seg` is checked for every frame (G = B = 0, values within {0,1}); a violation stops the collector.
* The camera (`camera:` in data_generation_config.yaml) is sent with `cam_config` on every circuit;
  `fish_eye_*` must be 0. Size and encoding of the received image are checked against it.
* Asynchronous mode only; `step_mode synchronous` is never sent (it freezes this fork).

### Output

```
generated_data/<split>/plan.json
generated_data/<split>/manifest.json                   config, sim exe, git hash, runs, frame counts
generated_data/<split>/<circuit_id>/images/000123.png  verbatim sim bytes (.jpg if img_enc: JPG)
                                   /masks/000123.png   L-mode {0,1}
                                   /meta.jsonl         one row per saved frame
                                   /track_info.json    generated tracks only (the sim's reply, as-is)
                                   /SKIPPED.json       only if skipped with N
generated_data/_logs/                                  Unity log if the collector started the sim
```

`circuit_id` is `gen_s<seed>` or the scene name. A meta row has `frame`, file names, `split`,
`circuit_id/index/type`, `seed`, `scene`, `telemetry_seq`, `sim_time`, `wall_time`, all telemetry
except images, the `control` in effect when the frame was produced, `road_frac`, and
`seen_in_train` (test only).

**Resume.** A frame exists if and only if its meta row exists. On restart, image/mask files without a
row are deleted and numbering continues. A generated circuit is rebuilt with the same seed, and its
`track_info.hash` must match the saved one, otherwise the collector stops.

## 2. Prepare training data

```
python -m donkey_seg_train.data.prepare                # Steps 1-3, cached: reruns are no-ops (dsg-prepare)
python -m donkey_seg_train.data.prepare --force        # rebuild everything (--only cache|textures|materialize)
python -m donkey_seg_train.data.preview --bench 2000   # cache/preview_augment.png + samples/s (dsg-preview-aug)
```

| step | module | output | what |
|---|---|---|---|
| 1 | `data/cache.py` | `cache/<split>/images.npy, masks.npy, meta.jsonl, info.json` | decode every PNG once into uint8 arrays read through a memory map; rebuilt automatically when more frames are collected |
| 2 | `data/textures.py` | `cache/textures/` | 128 procedural noise textures for the background augmentation |
| 3 | `data/materialize.py` | `cache/train_aug<K>/` | optional (`materialize.copies > 0`): K fixed augmented copies; online augmentation is the default |

Settings live in `train.yaml` (overrides only; defaults are in `src/donkey_seg_train/config.py` and
`AugmentConfig` in `data/augment.py`; an unknown key is an error).

**Splits.** `train` = train circuits minus `data.val_circuits`, augmented online every epoch.
`val` = the held-out circuits (currently `gen_s388044`), never augmented. The split is **by circuit**:
consecutive frames are near-duplicates, so a random frame split would measure memorisation.
`test` = the test split, never augmented. Current data: 8000 train / 2000 val / 1000 test.

**Augmentation** (`data/augment.py`, torch RNG only so every worker differs): geometry on image + mask
(hflip, small shift/rotate/scale; out-of-frame pixels get mask 255 = ignore), then appearance on the
image only (background noise texture, shadows, colour/gamma/hue, rare gray/channel shuffle, sensor
noise, cutout). A flipped frame has `steering` negated in its meta.

## 3. Train (PyTorch)

```python
from donkey_seg_train.config import load_config
from donkey_seg_train.data import build_dataloaders

if __name__ == "__main__":            # required on Windows when num_workers > 0
    loaders = build_dataloaders(load_config("train.yaml"))
    for img, mask in loaders["train"]:  # img float32 Bx3xHxW in [0,1]; mask int64 BxHxW {0,1,255}
        ...                              # use ignore_index=255 in the loss
```

`model.py`, `losses.py` and `train.py` are placeholders.

## Label caveats (SEGMENTATION.md)

* Fisheye is not supported (the mask is undistorted).
* Start lines are labelled as their whole quad.
* `imageb` (second camera) has no mask; `img_d = 1` does not affect the mask.
