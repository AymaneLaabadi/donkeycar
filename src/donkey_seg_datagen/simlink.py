"""Link to the sim: gym_donkeycar's SimClient (v1.3.x) + our own message handler.

The stock gym env has no senders for seg_config / gen_track and does not route track_info,
so we reuse only its socket client. Every value is sent as a string (the sim parses strings).
Asynchronous mode only: step_mode synchronous freezes this fork and is never sent.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
import time

from gym_donkeycar.core.message import IMesgHandler
from gym_donkeycar.core.sim_client import SimClient

log = logging.getLogger("dsg.sim")

CAM_INT = ("img_w", "img_h", "img_d")
CAM_FLOAT = ("fov", "fish_eye_x", "fish_eye_y", "offset_x", "offset_y", "offset_z", "rot_x", "rot_y", "rot_z")


class SimDisconnected(Exception):
    pass


class SimSetupError(Exception):
    pass


class Handler(IMesgHandler):
    """Thread-safe store of what the sim sends (called from the socket thread)."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.seq = 0                       # telemetry messages received so far
        self.latest: dict | None = None
        self.track_info: dict | None = None
        self.counts = {"car_loaded": 0, "scene_selection_ready": 0, "scene_names": 0, "track_info": 0}
        self.scene_names: list[str] = []
        self.aborted = False

    def on_connect(self, client) -> None:
        pass

    def on_recv_message(self, msg: dict) -> None:
        t = msg.get("msg_type")
        with self.cond:
            if t == "telemetry":
                self.seq += 1
                self.latest = msg
            elif t == "track_info":
                self.track_info = msg
            elif t == "scene_names":
                self.scene_names = list(msg.get("scene_names", []))
            elif t == "aborted":
                self.aborted = True
            if t in self.counts:
                self.counts[t] += 1
            self.cond.notify_all()

    def on_close(self) -> None:
        with self.cond:
            self.aborted = True
            self.cond.notify_all()

    on_disconnect = on_close


class SimLink:
    def __init__(self, cfg: dict, log_dir: str):
        self.sc = cfg["sim"]
        self.host, self.port = self.sc["host"], int(self.sc["port"])
        self.log_dir = log_dir
        self.proc: subprocess.Popen | None = None
        self.client: SimClient | None = None
        self.h = Handler()
        self.last_seq = 0
        self.scene: str | None = None

    # ------------------------------------------------------------ process / connection
    def _port_open(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.5):
                return True
        except OSError:
            return False

    def launch(self) -> None:
        exe = self.sc.get("exe_path")
        if not exe or not os.path.exists(exe):
            raise SimSetupError(f"sim exe not found: {exe!r}. Start the sim yourself or fix sim.exe_path.")
        os.makedirs(self.log_dir, exist_ok=True)
        logfile = os.path.join(self.log_dir, time.strftime("unity_%Y%m%d_%H%M%S.log"))
        log.info("launching sim %s", exe)
        self.proc = subprocess.Popen([exe, "--host", self.host, "--port", str(self.port), "-logFile", logfile])

    def start(self) -> None:
        mode = self.sc["launch"]
        if mode == "always" or (mode == "auto" and not self._port_open()):
            self.launch()
        self._connect(float(self.sc["startup_timeout_s"]))

    def _connect(self, timeout: float) -> None:
        t0, last = time.time(), None
        while time.time() - t0 < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                raise SimSetupError(f"sim process exited with code {self.proc.returncode}")
            try:
                self.h = Handler()
                self.client = SimClient((self.host, self.port), self.h)
                self.last_seq, self.scene = 0, None
                log.info("connected to sim %s:%d", self.host, self.port)
                return
            except Exception as e:        # SDClient raises a plain Exception while Unity boots
                last = e
                time.sleep(1.0)
        raise SimSetupError(f"could not connect to sim at {self.host}:{self.port} within {timeout:.0f}s ({last})")

    def reconnect(self) -> None:
        log.warning("reconnecting to the sim ...")
        self.close_client()
        if self.proc is not None and self.proc.poll() is not None:
            log.warning("sim process died, relaunching")
            self.launch()
        self._connect(float(self.sc["startup_timeout_s"]))

    def close_client(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None

    def shutdown(self) -> None:
        self.close_client()
        if self.proc is not None:
            self.proc.kill()
            self.proc = None

    # ------------------------------------------------------------ sending
    def send(self, msg: dict) -> None:
        if self.client is None or self.h.aborted or self.client.aborted:
            raise SimDisconnected("socket closed")
        msg = {k: (v if k == "msg_type" else (repr(v) if isinstance(v, float) else str(v))) for k, v in msg.items()}
        try:
            self.client.send_now(msg)
        except OSError as e:
            raise SimDisconnected(str(e)) from e

    def control(self, steering: float, throttle: float, brake: float = 0.0) -> None:
        self.send({"msg_type": "control", "steering": float(steering), "throttle": float(throttle),
                   "brake": float(brake)})

    def reset_car(self) -> None:
        self.send({"msg_type": "reset_car"})

    def configure(self, car: dict, cam: dict) -> None:
        """car_config + cam_config (inference camera) + seg_config enabled=1."""
        r, g, b = car["body_rgb"]
        self.send({"msg_type": "car_config", "body_style": car["body_style"], "body_r": int(r), "body_g": int(g),
                   "body_b": int(b), "car_name": car["car_name"], "font_size": int(car["font_size"])})
        time.sleep(0.1)
        msg = {"msg_type": "cam_config", "img_enc": str(cam["img_enc"]).upper()}
        msg.update({k: int(cam[k]) for k in CAM_INT})
        msg.update({k: float(cam[k]) for k in CAM_FLOAT})
        self.send(msg)
        time.sleep(0.1)
        self.send({"msg_type": "seg_config", "enabled": "1"})
        time.sleep(0.2)

    def gen_track(self, seed: int, timeout: float = 30.0) -> dict:
        """gen_track in generated_road; returns the track_info reply (raises if not ok)."""
        n0 = self.h.counts["track_info"]
        self.send({"msg_type": "gen_track", "seed": int(seed)})
        if not self._wait(lambda: self.h.counts["track_info"] > n0, timeout, "track_info"):
            raise SimSetupError(f"no track_info within {timeout:.0f}s after gen_track seed {seed}")
        info = dict(self.h.track_info)
        if not info.get("ok", True) or "error" in info:
            raise SimSetupError(f"gen_track seed {seed} failed: {info}")
        return info

    # ------------------------------------------------------------ receiving
    def _wait(self, pred, timeout: float, what: str):
        t_end = time.time() + timeout
        with self.h.cond:
            while True:
                r = pred()
                if r:
                    return r
                if self.h.aborted:
                    raise SimDisconnected(f"connection lost while waiting for {what}")
                rem = t_end - time.time()
                if rem <= 0:
                    return None
                self.h.cond.wait(min(rem, 0.25))

    def next_frame(self, timeout: float | None = None) -> tuple[int, dict]:
        """Newest telemetry message after the last one returned -> (telemetry seq, msg).
        Older unread messages are skipped; the seq gap tells how many."""
        timeout = float(self.sc["telemetry_timeout_s"]) if timeout is None else timeout
        h = self.h
        r = self._wait(lambda: h.seq > self.last_seq and (h.seq, h.latest), timeout, "telemetry")
        if not r:
            raise SimDisconnected(f"no telemetry for {timeout:.0f}s")
        self.last_seq = r[0]
        return r

    # ------------------------------------------------------------ scenes
    def ensure_scene(self, name: str) -> bool:
        """Load scene `name` (via the menu) unless already in it. Returns True if it was (re)loaded."""
        if self.scene == name:
            return False
        h, to = self.h, float(self.sc["scene_load_timeout_s"])
        if self.scene is not None or self._wait(lambda: h.latest or h.counts["car_loaded"], 3.0, "first message"):
            self.send({"msg_type": "exit_scene"})       # we are in a track scene -> back to the menu
        self.scene = None
        n0, t0 = h.counts["scene_names"], time.time()
        while True:                                     # only the menu answers get_scene_names
            self.send({"msg_type": "get_scene_names"})
            if self._wait(lambda: h.counts["scene_names"] > n0, 1.0, "scene_names"):
                break
            if time.time() - t0 > to:
                raise SimSetupError("the sim menu did not answer get_scene_names")
        if name not in h.scene_names:
            raise SimSetupError(f"scene '{name}' not in the sim's scene list {h.scene_names}")
        n_car = h.counts["car_loaded"]
        log.info("loading scene %s", name)
        self.send({"msg_type": "load_scene", "scene_name": name})
        if not self._wait(lambda: h.counts["car_loaded"] > n_car, to, "car_loaded"):
            raise SimSetupError(f"no car_loaded after load_scene {name}")
        self.scene = name
        time.sleep(0.5)
        return True
