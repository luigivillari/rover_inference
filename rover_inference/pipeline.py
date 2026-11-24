"""
Inference pipeline without ROS: frames + Vicon poses + goals/command in, predicted trajectory out.

It rebuilds, for one camera frame taken at time t, exactly the sample the model was trained on
(extract_samples.py v3):
  frames    the frame at t and the frame closest to t - 0.5 s (within 50 ms)
  status    command one-hot (intention: NAVSIM rule on the chain of goals, L 1.5 m, ratio 0.1),
            velocity and acceleration in the rover frame (centred differences as in training, or backward)
  output    8 poses (x, y, heading) at t + 0.5 ... 4.0 s in the rover frame at t

The output carries the Vicon pose at t (pose_world), so the controller can anchor the trajectory in the
world and follow it from the rover's CURRENT pose: inference takes a few hundred ms on the Jetson and the
rover keeps moving meanwhile.

The ROS node (node.py) and the bag replay (tools/replay_bag.py) both use this class, so what is tested on a
recorded bag is the code that runs on the rover.
"""
import math
import threading
import time

import numpy as np

from .commands import CMD_NAMES, goals_command, onehot, self_command
from .ego import PoseBuffer
from .frames import FrameBuffer
from .timing import ClockAligner

DEFAULTS = {
    "camera_latency": 0.017,          # s, image stamp - time the image was taken (training: 17 ms)
    "frame_dt": 0.5,                  # s between the two frames (NAVSIM double image)
    "max_frame_offset": 0.05,         # s, training: frame at t - 0.5 s within 50 ms
    "time_mode": "aligned",           # aligned (as training) | receive
    "dynamics": "central",            # central (as training, waits ~0.3 s for later poses) | backward
    "h_vel": 0.1, "h_acc": 0.25, "smooth": 0.05, "max_pose_gap": 0.1,
    "command_source": "goals",        # goals | topic | fixed | self
    "command_fixed": "straight",
    "command_timeout": 1.0,           # s, topic: older command -> unknown
    "route_lookahead": 1.5, "route_threshold": 0.1,
}


class InferencePipeline:
    def __init__(self, runner, **cfg):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown pipeline options: {sorted(unknown)}")
        self.cfg = dict(DEFAULTS, **cfg)
        c = self.cfg
        if c["command_source"] not in ("goals", "topic", "fixed", "self"):
            raise ValueError(f"unknown command source {c['command_source']}")
        self.runner = runner
        self.clock = ClockAligner(c["time_mode"])
        self.poses = PoseBuffer(keep_s=3.0, smooth_s=c["smooth"], max_gap_s=c["max_pose_gap"])
        self.frames = FrameBuffer(keep_s=2.0, latency_s=c["camera_latency"])
        self.lock = threading.Lock()
        self.goals = None                       # list of world (x, y): current goal first
        self.command_msg = (None, -math.inf)    # (name, receive time)
        self.prev = None                        # (pose_world, trajectory) of the last prediction (command 'self')
        self.last_stamp = None

    # ------------------------------------------------------------------ inputs (called from ROS callbacks)
    def on_image(self, stamp, recv, jpeg, stamp_ns=None):
        """jpeg None: only the clock is updated (bag replay keeps just the frames it needs)."""
        with self.lock:
            self.clock.add_image(stamp, recv)
            if jpeg is not None:
                self.frames.add(self.clock.image_time(stamp, recv),
                                stamp_ns if stamp_ns is not None else round(stamp * 1e9), jpeg)

    def on_vicon(self, stamp, recv, x, y, yaw):
        with self.lock:
            self.clock.add_vicon(stamp, recv)
            self.poses.add(self.clock.vicon_key(stamp, recv), x, y, yaw)   # stored on the Vicon clock

    def on_goals(self, goals):
        with self.lock:
            self.goals = [np.asarray(g[:2], float) for g in goals] if goals else None

    def on_command(self, name, recv):
        with self.lock:
            self.command_msg = (name if name in CMD_NAMES else "unknown", recv)

    # ------------------------------------------------------------------ one prediction
    def _command(self, origin, now):
        c = self.cfg
        src = c["command_source"]
        if src == "goals":
            idx, _ = goals_command(origin, self.goals, c["route_lookahead"], c["route_threshold"])
        elif src == "topic":
            name, t = self.command_msg
            idx = CMD_NAMES.index(name) if name is not None and now - t <= c["command_timeout"] else 3
        elif src == "fixed":
            idx = CMD_NAMES.index(c["command_fixed"])
        else:
            idx, _ = self_command(origin, self.prev, c["route_lookahead"], c["route_threshold"])
        return idx

    def latest_pose_time(self):
        """Newest Vicon pose on the camera time base (None while the clock offset is not known)."""
        off = self.clock.offset()
        return None if off is None or self.poses.latest is None else self.poses.latest + off

    def _select_frame(self):
        """Newest frame whose status can be computed now (central dynamics need poses after t)."""
        c = self.cfg
        latest = self.latest_pose_time()
        if latest is None:
            return None, "no Vicon pose" if self.clock.ready else "clock offset still being estimated"
        before, after = self.poses.needs(c["dynamics"], c["h_vel"], c["h_acc"])
        f = self.frames.newest(not_after=latest - after)
        if f is None:
            return None, "no camera frame"
        return f, None

    def build(self, now, frame=None):
        """Inputs of the model for one frame (newest usable one if None). Returns (sample, reason)."""
        c = self.cfg
        with self.lock:
            if frame is None:
                frame, why = self._select_frame()
                if frame is None:
                    return None, why
            t = frame[0]
            prev = self.frames.nearest(t - c["frame_dt"], c["max_frame_offset"])
            if prev is None:
                return None, "no frame 0.5 s before"
            off = self.clock.offset()
            if off is None:
                return None, "clock offset still being estimated"
            tv = t - off                                     # frame time on the Vicon clock
            before, after = self.poses.needs(c["dynamics"], c["h_vel"], c["h_acc"])
            if not self.poses.covered(tv - before, tv + after):
                return None, "Vicon gap around the frame"
            origin, v, a = self.poses.dynamics(tv, c["h_vel"], c["h_acc"], c["dynamics"])
            cmd = self._command(origin, now)
            frames = (prev, frame)
        img_prev, img_now = self.frames.image(frames[0]), self.frames.image(frames[1])
        status = np.array(onehot(cmd) + [v[0], v[1], a[0], a[1]], np.float32)
        return {"t": t, "frame_stamp_ns": frames[1][1], "prev_stamp_ns": frames[0][1], "pose_world": origin,
                "status": status, "command": CMD_NAMES[cmd], "images": (img_prev, img_now)}, None

    def step(self, now):
        """Predict on the newest usable frame (not twice the same). Returns (result, reason)."""
        s, why = self.build(now)
        if s is None:
            return None, why
        if s["frame_stamp_ns"] == self.last_stamp:
            return None, "no new frame"
        return self.run(s, now), None

    def run(self, s, now=None):
        t0 = time.perf_counter()
        traj = np.asarray(self.runner(*s["images"], s["status"]), float)
        infer_ms = (time.perf_counter() - t0) * 1000
        with self.lock:
            self.prev = (s["pose_world"], traj)
            self.last_stamp = s["frame_stamp_ns"]
        out = {k: v for k, v in s.items() if k != "images"}
        out.update(trajectory=traj, infer_ms=infer_ms)
        if now is not None:
            out["age_ms"] = (now - s["t"]) * 1000     # frame taken -> prediction ready
        return out
