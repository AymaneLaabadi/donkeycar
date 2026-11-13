"""Manual-driving data collector (pygame).

    python -m donkey_seg_datagen.collect --split train [--config data_generation_config.yaml]

Keys: arrows / WASD drive, SPACE pause/resume recording, R reset_car, BACKSPACE delete the last
3 s of saved frames, M mask overlay on/off, N skip circuit (asks Y to confirm), ESC save and quit.
Gamepad: left stick X steers, RT throttle, LT brake (or left stick Y if there are no triggers).
Rerunning the same command resumes where it stopped.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_config
from .io_utils import atomic_write_bytes, atomic_write_json, read_meta, write_meta
from .masks import EXT, MAGIC, decode_seg, overlay
from .plan import load_or_create_plan
from .simlink import SimDisconnected, SimLink, SimSetupError

log = logging.getLogger("dsg")

GUIDANCE = ["Drive normally, sometimes closer to the edges,",
            "take some curves wide and some tight.",
            "SPACE: start / pause recording"]
DROP_KEYS = {"image", "image_seg", "imageb", "msg_type"}


# ---------------------------------------------------------------------------- per-circuit files
class CircuitWriter:
    """out/<split>/<circuit_id>/{images,masks}/NNNNNN.*, meta.jsonl. A frame exists iff its meta row does."""

    def __init__(self, path: Path, img_ext: str):
        self.dir = path
        self.ext = img_ext
        (path / "images").mkdir(parents=True, exist_ok=True)
        (path / "masks").mkdir(parents=True, exist_ok=True)
        self.rows = read_meta(path)
        write_meta(path, self.rows)                 # drops a torn last line
        keep = {r["image_file"] for r in self.rows} | {r["mask_file"] for r in self.rows}
        for sub in ("images", "masks"):             # files written before a crash, without a meta row
            for f in (path / sub).iterdir():
                if f"{sub}/{f.name}" not in keep:
                    f.unlink()

    @property
    def n(self) -> int:
        return len(self.rows)

    def save(self, img_bytes: bytes, mask: np.ndarray, row: dict) -> None:
        i = self.rows[-1]["frame"] + 1 if self.rows else 0
        row = {"frame": i, "image_file": f"images/{i:06d}.{self.ext}", "mask_file": f"masks/{i:06d}.png", **row}
        atomic_write_bytes(self.dir / row["image_file"], img_bytes)            # verbatim sim bytes
        buf = io.BytesIO()
        Image.fromarray(mask, mode="L").save(buf, format="PNG")
        atomic_write_bytes(self.dir / row["mask_file"], buf.getvalue())
        with open(self.dir / "meta.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)

    def delete_since(self, wall_t: float) -> int:
        k = len(self.rows)
        while k and self.rows[k - 1]["wall_time"] >= wall_t:
            k -= 1
        gone, self.rows = self.rows[k:], self.rows[:k]
        write_meta(self.dir, self.rows)
        for r in gone:
            (self.dir / r["image_file"]).unlink(missing_ok=True)
            (self.dir / r["mask_file"]).unlink(missing_ok=True)
        return len(gone)


# ---------------------------------------------------------------------------- input + window
class Controls:
    """Keyboard / gamepad -> smoothed (steering, throttle, brake)."""

    def __init__(self, pg, m: dict):
        self.pg, self.m = pg, m
        pg.joystick.init()
        self.js = pg.joystick.Joystick(0) if pg.joystick.get_count() else None
        if self.js:
            self.js.init()
            print(f"gamepad: {self.js.get_name()}")
        self.steer = self.throttle = 0.0

    def update(self, dt: float) -> tuple[float, float, float]:
        pg, m = self.pg, self.m
        k = pg.key.get_pressed()
        steer_in = (k[pg.K_RIGHT] or k[pg.K_d]) - (k[pg.K_LEFT] or k[pg.K_a])
        thr_target = m["throttle"] * ((k[pg.K_UP] or k[pg.K_w]) - (k[pg.K_DOWN] or k[pg.K_s]))
        brake = 0.0
        if steer_in:
            self.steer += steer_in * m["steer_rate"] * dt
        else:
            self.steer -= np.sign(self.steer) * min(abs(self.steer), m["steer_return"] * dt)
        if self.js is not None:
            dz = m["deadzone"]
            ax = self.js.get_axis(0)
            if abs(ax) > dz:
                self.steer = ax
            if self.js.get_numaxes() >= 6:          # XInput: axis 4 = LT, 5 = RT, range [-1, 1]
                rt, lt = (self.js.get_axis(5) + 1) / 2, (self.js.get_axis(4) + 1) / 2
                thr_target = rt if rt > dz else thr_target
                brake = lt if lt > dz else 0.0
            elif abs(self.js.get_axis(1)) > dz:
                thr_target = -self.js.get_axis(1)
        self.steer = float(np.clip(self.steer, -1, 1))
        step = m["throttle_rate"] * dt
        self.throttle = float(np.clip(thr_target, self.throttle - step, self.throttle + step))
        return self.steer, self.throttle, float(brake)


class Screen:
    PANEL = 330

    def __init__(self, pg, w: int, h: int, scale: int):
        self.pg, self.scale = pg, scale
        self.img_size = (w * scale, h * scale)
        self.win = pg.display.set_mode((self.img_size[0] + self.PANEL, max(self.img_size[1], 420)))
        pg.display.set_caption("donkey-seg-datagen collector")
        self.font = pg.font.SysFont("consolas", 17)
        self.big = pg.font.SysFont("consolas", 26, bold=True)

    def draw(self, rgb: np.ndarray | None, lines: list[tuple[str, tuple]], banner: list[str] | None = None) -> None:
        pg = self.pg
        self.win.fill((25, 25, 25))
        if rgb is not None:
            surf = pg.transform.scale(pg.surfarray.make_surface(rgb.swapaxes(0, 1)), self.img_size)
            self.win.blit(surf, (0, 0))
        y = 10
        for text, color in lines:
            self.win.blit(self.font.render(text, True, color), (self.img_size[0] + 12, y))
            y += 22
        if banner:
            box = pg.Surface((self.img_size[0], 30 * len(banner) + 20), pg.SRCALPHA)
            box.fill((0, 0, 0, 170))
            self.win.blit(box, (0, self.img_size[1] // 2 - box.get_height() // 2))
            for i, t in enumerate(banner):
                s = self.big.render(t, True, (255, 230, 120))
                self.win.blit(s, ((self.img_size[0] - s.get_width()) // 2,
                                  self.img_size[1] // 2 - box.get_height() // 2 + 10 + 30 * i))
        pg.display.flip()


# ---------------------------------------------------------------------------- collector
class Collector:
    def __init__(self, cfg: dict, split: str):
        import pygame as pg
        self.pg = pg
        self.cfg, self.split = cfg, split
        self.cam = cfg["camera"]
        self.enc = str(self.cam["img_enc"]).upper()
        self.root = Path(cfg["output"]["dir"])
        self.split_dir = self.root / split
        self.plan = load_or_create_plan(cfg, split)
        self.link = SimLink(cfg, str(self.root / "_logs"))
        pg.init()
        self.screen = Screen(pg, int(self.cam["img_w"]), int(self.cam["img_h"]), int(cfg["manual"]["window_scale"]))
        self.controls = Controls(pg, cfg["manual"])
        self.show_mask = False
        self.quit = False
        self.run_record = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "end": None, "saved": 0}

    # ---- status of circuits on disk
    def circuit_dir(self, c: dict) -> Path:
        return self.split_dir / c["id"]

    def status(self, c: dict) -> tuple[int, str]:
        d = self.circuit_dir(c)
        n = len(read_meta(d))
        if (d / "SKIPPED.json").exists():
            return n, "skipped"
        return n, "done" if n >= c["frames"] else ("partial" if n else "todo")

    def write_manifest(self) -> None:
        p = self.split_dir / "manifest.json"
        man = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
            "split": self.split, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
        runs = [r for r in man["runs"] if r["start"] != self.run_record["start"]]
        try:
            git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 cwd=Path(__file__).parent).stdout.strip() or None
        except OSError:
            git = None
        circuits = {}
        for c in self.plan["circuits"]:
            n, st = self.status(c)
            circuits[c["id"]] = {"index": c["index"], "type": c["type"], "frames": n, "target": c["frames"], "status": st}
        man.update({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "config_path": self.cfg["_config_path"],
                    "config": {k: v for k, v in self.cfg.items() if not k.startswith("_")},
                    "sim_exe": self.cfg["sim"].get("exe_path"), "git": git,
                    "runs": runs + [dict(self.run_record, end=time.strftime("%Y-%m-%d %H:%M:%S"))],
                    "frames_total": sum(v["frames"] for v in circuits.values()), "circuits": circuits})
        atomic_write_json(p, man)

    # ---- main loop
    def run(self) -> None:
        self.link.start()
        try:
            circuits = self.plan["circuits"]
            for c in circuits:
                n, st = self.status(c)
                print(f"  circuit {c['index'] + 1}/{len(circuits)} {c['id']:24s} {n:5d}/{c['frames']}  {st}")
            for c in circuits:
                if self.quit:
                    break
                if self.status(c)[1] in ("done", "skipped"):
                    continue
                for attempt in range(3):
                    try:
                        self.run_circuit(c)
                        break
                    except SimDisconnected as e:
                        if attempt == 2:
                            raise
                        print(f"sim connection lost ({e}); reconnecting")
                        self.link.reconnect()
                self.write_manifest()
            if not self.quit:
                print("all circuits of this split are done")
        finally:
            self.write_manifest()
            try:
                self.link.send({"msg_type": "seg_config", "enabled": "0"})
            except Exception:
                pass
            self.link.shutdown()
            self.pg.quit()

    def setup_circuit(self, c: dict, d: Path) -> dict:
        k, total = c["index"] + 1, len(self.plan["circuits"])
        self.screen.draw(None, [(f"circuit {k}/{total}: {c['id']}", (230, 230, 230))], ["loading ..."])
        self.pg.event.pump()
        self.link.ensure_scene(c["scene"])
        self.link.configure(self.cfg["car"], self.cam)
        if c["type"] == "generated":
            info = self.link.gen_track(c["seed"])
            log.info("track_info: %s", json.dumps(info))
            p = d / "track_info.json"
            if p.exists():
                old = json.loads(p.read_text(encoding="utf-8"))
                if old.get("hash") != info.get("hash"):
                    raise SystemExit(f"gen_track seed {c['seed']} built a different track than the saved "
                                     f"{p} (hash {old.get('hash')} -> {info.get('hash')}); sim changed? "
                                     f"Move {d} away to re-record this circuit.")
            atomic_write_json(p, info)
        return self.check_stream()

    def check_stream(self) -> dict:
        t0 = time.time()
        while True:
            _, msg = self.link.next_frame()
            if "image_seg" in msg or time.time() - t0 > 5:
                break
        if "image_seg" not in msg:
            raise SystemExit("telemetry has no 'image_seg': is this the segmentation build of the fork?")
        if "cte" not in msg or "pos_x" not in msg:
            raise SystemExit("telemetry has no 'cte'/'pos_x': extended telemetry is OFF. In the sim's main menu "
                             "open the settings panel, tick 'extendedTelemetry', press Save, then rerun.")
        raw = base64.b64decode(msg["image"])
        im = Image.open(io.BytesIO(raw))
        want = (int(self.cam["img_w"]), int(self.cam["img_h"]))
        if im.size != want or not raw.startswith(MAGIC[self.enc]):
            raise SystemExit(f"camera mismatch: got {im.size} {im.format}, configured {want} {self.enc}")
        return msg

    def run_circuit(self, c: dict) -> None:
        pg, m, flt = self.pg, self.cfg["manual"], self.cfg["filters"]
        d = self.circuit_dir(c)
        k, total = c["index"] + 1, len(self.plan["circuits"])
        self.setup_circuit(c, d)
        w = CircuitWriter(d, EXT[self.enc])
        seen = c.get("seen_in_train")
        discard_s = float(self.cfg["sim"]["discard_s"])
        discard_until = time.time() + discard_s
        t_start = time.time()
        recording, confirm_skip, done_at = False, False, None
        since, prev_seq, t_prev = 0, None, time.time()
        note, note_until = "", 0.0
        ctrl = (0.0, 0.0, 0.0)

        while True:
            # ---- input
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    e = pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE)
                if e.type != pg.KEYDOWN:
                    continue
                if confirm_skip:
                    confirm_skip = False
                    if e.key == pg.K_y:
                        atomic_write_json(d / "SKIPPED.json", {"frames": w.n, "target": c["frames"],
                                                                "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                        print(f"  skipped {c['id']} at {w.n}/{c['frames']} frames")
                        return
                    note, note_until = "skip cancelled", time.time() + 2
                elif e.key == pg.K_ESCAPE:
                    self.quit = True
                    self.link.control(0, 0, 1)
                    print(f"  quit: {c['id']} at {w.n}/{c['frames']} frames (rerun to resume)")
                    return
                elif e.key == pg.K_SPACE:
                    recording = not recording
                elif e.key == pg.K_r:
                    self.link.reset_car()
                    discard_until = time.time() + discard_s
                elif e.key == pg.K_BACKSPACE:
                    n = w.delete_since(time.time() - float(m["backspace_s"]))
                    note, note_until = f"deleted {n} frames", time.time() + 3
                elif e.key == pg.K_m:
                    self.show_mask = not self.show_mask
                elif e.key == pg.K_n:
                    confirm_skip = True

            # ---- frame
            seq, msg = self.link.next_frame()
            gap = 1 if prev_seq is None else seq - prev_seq
            prev_seq = seq
            now = time.time()
            dt, t_prev = min(now - t_prev, 0.2), now
            img_bytes = base64.b64decode(msg["image"])
            mask = decode_seg(msg["image_seg"])     # raises MaskError if G/B != 0 or values not in {0,1}
            rgb = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

            if done_at is not None:
                ctrl = (0.0, 0.0, 1.0)
                self.link.control(*ctrl)
                self.screen.draw(rgb, self.panel(c, k, total, w, msg, "DONE"),
                                 [f"circuit {k}/{total} done", "loading next ..." if k < total else "split complete"])
                if now - done_at > 2.0:
                    return
                continue

            # ---- record (the frame was produced under the last control sent)
            discarding = now < discard_until
            if recording and not discarding:
                since += gap
                speed = float(msg.get("speed", 0.0))
                hit_ok = not flt["skip_hit"] or msg.get("hit", "none") in ("none", None, "")
                if since >= c["frame_step"] and hit_ok and speed >= float(flt["min_speed"]):
                    tel = {kk: v for kk, v in msg.items() if kk not in DROP_KEYS}
                    row = {"split": self.split, "circuit_id": c["id"], "circuit_index": c["index"],
                           "circuit_type": c["type"], "seed": c["seed"], "scene": c["scene"],
                           "telemetry_seq": seq, "sim_time": msg.get("time"), "wall_time": now,
                           "telemetry": tel, "control": dict(zip(("steering", "throttle", "brake"), ctrl)),
                           "road_frac": round(float(mask.mean()), 6)}
                    if seen is not None:
                        row["seen_in_train"] = seen
                    w.save(img_bytes, mask, row)
                    self.run_record["saved"] += 1
                    since = 0
                    if w.n >= c["frames"]:
                        done_at = now
                        print(f"  circuit {k}/{total} {c['id']} done ({w.n} frames)")

            # ---- drive + draw
            ctrl = self.drive(dt, mask)
            self.link.control(*ctrl)
            state = "DISCARD" if discarding else ("REC" if recording else "PAUSED")
            banner = None
            if confirm_skip:
                banner = [f"Skip circuit {c['id']}?", "Y = skip, any other key = cancel"]
            elif not recording and (w.n == 0 or now - t_start < 6):
                banner = [f"circuit {k}/{total}: {c['id']}"] + GUIDANCE
            if note and now > note_until:
                note = ""
            lines = self.panel(c, k, total, w, msg, state) + ([("", (0, 0, 0)), (note, (255, 200, 80))] if note else [])
            self.screen.draw(overlay(rgb, mask, float(m["overlay_alpha"])) if self.show_mask else rgb, lines, banner)

    def drive(self, dt: float, mask: np.ndarray) -> tuple[float, float, float]:
        return self.controls.update(dt)

    def panel(self, c, k, total, w, msg, state) -> list[tuple[str, tuple]]:
        col = {"REC": (255, 80, 80), "PAUSED": (200, 200, 200), "DISCARD": (255, 200, 80), "DONE": (120, 230, 120)}
        name = f"seed {c['seed']}" if c["type"] == "generated" else c["scene"]
        return [(f"split     {self.split}", (230, 230, 230)),
                (f"circuit   {k}/{total}", (230, 230, 230)),
                (f"          {c['type']}: {name}", (230, 230, 230)),
                (f"saved     {w.n}/{c['frames']}", (230, 230, 230)),
                (f"step      every {c['frame_step']} frames", (160, 160, 160)),
                (f"speed     {float(msg.get('speed', 0)):.2f}", (230, 230, 230)),
                (f"hit       {msg.get('hit', '?')}", (160, 160, 160)),
                (f"state     {state}", col[state]),
                ("", (0, 0, 0)),
                ("SPACE rec on/off   R reset", (140, 140, 140)),
                ("BKSP  delete last 3 s", (140, 140, 140)),
                ("M mask overlay  N skip", (140, 140, 140)),
                ("ESC   save + quit", (140, 140, 140))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default="data_generation_config.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.split not in cfg["splits"]:
        raise SystemExit(f"unknown split {args.split!r}; config has {list(cfg['splits'])}")
    try:
        import pygame  # noqa: F401
    except ImportError:
        raise SystemExit("the collector needs pygame: pip install pygame-ce")
    try:
        Collector(cfg, args.split).run()
    except SimSetupError as e:
        raise SystemExit(f"sim setup failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
