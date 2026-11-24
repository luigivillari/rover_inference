"""
Trajectory follower (no ROS): the newest Drive-JEPA prediction -> (v, w) for the rover.

Each prediction is 8 poses in the rover frame at the time t its frame was taken, plus the Vicon pose at t
(pose_world). It is anchored in the world with that pose, so a prediction that arrives a few hundred ms late
is still followed from where the rover IS now: the controller projects the current position on the
predicted path and pure-pursues it (same pure pursuit as the data-collection driver: lookahead = speed x 1 s
in 0.15-0.35 m, curvature limited to 1 / 0.20 m). Only the near part of the 4 s prediction is used (the
part where the open-loop test found no contacts); a new prediction replaces it every few hundred ms.

Speed: 'model' = the speed of the predicted trajectory over its first second (the model also predicts how
fast to go), capped at v_max; 'fixed' = v_fixed.

Safety, the same rules as rover_collector Collector.update:
  no fresh Vicon pose (> pose_timeout)         -> zero at once, resume when it is back
  outline outside the geofence                 -> STOP (latched)
  outline closer than stop_gap to an obstacle  -> STOP (latched)
  battery below min_battery_v                  -> STOP (latched)
and for the model (brake with the acceleration limit, resume with the next good prediction):
  no prediction / newest prediction older than prediction_timeout
  rover farther than max_cross_track from the predicted path
  less than min_path_ahead of predicted path left, or predicted speed ~0
Every latched stop and every brake is counted (stats): in closed loop they are the interventions.
"""
import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .ego import to_world
from .safety import Outline, pose_gaps


@dataclass
class FollowerParams:
    pose_timeout: float = 0.3          # s, as the collector
    prediction_timeout: float = 1.0    # s between the frame of the newest prediction and now
    speed: str = "model"               # model | fixed
    v_fixed: float = 0.20
    v_max: float = 0.25                # m/s cap (first closed-loop tests; training data 0.20-0.35)
    v_stop: float = 0.03               # predicted speed below this = the model wants to stop
    speed_horizon: float = 1.0         # s of prediction used for the speed
    accel_max: float = 0.5             # m/s^2, as config/collector.yaml
    w_max: float = 1.5                 # rad/s
    w_accel_max: float = 4.0           # rad/s^2
    r_min_ctrl: float = 0.20           # m, tightest curvature commanded
    lookahead_gain: float = 1.0        # s
    lookahead_min: float = 0.15
    lookahead_max: float = 0.35
    max_cross_track: float = 0.20      # m from the predicted path
    min_path_ahead: float = 0.10       # m of predicted path ahead of the rover
    half_length: float = 0.10          # rover outline from the Vicon origin
    half_width: float = 0.10
    stop_gap: float = 0.01             # m, as the collector
    min_battery_v: float = 10.5        # 0 = off

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]


@dataclass
class Output:
    v: float
    w: float                           # REAL yaw rate; the node multiplies it by cmd_w_scale for the motors
    state: str
    gap: float = math.inf


@dataclass
class Prediction:
    t: float                           # time the frame was taken (camera clock)
    path: np.ndarray                   # (9, 2) world: pose at t, then the 8 predicted poses
    s: np.ndarray                      # (9,) arc length
    times: np.ndarray = field(default_factory=lambda: 0.5 * np.arange(9))
    seq: int = 0

    @classmethod
    def from_msg(cls, d, dt=0.5):
        anchor = np.asarray(d["pose_world"], float)
        traj = np.asarray(d["trajectory"], float)
        path = np.vstack([anchor[None, :2], to_world(anchor, traj[:, :2])])
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
        return cls(float(d["t"]), path, s, dt * np.arange(len(path)), int(d.get("seq", 0)))

    def project(self, xy):
        """(arc length of the closest point, distance to the path)."""
        seg = np.diff(self.path, axis=0)
        L2 = np.maximum((seg ** 2).sum(1), 1e-12)
        f = np.clip(((xy - self.path[:-1]) * seg).sum(1) / L2, 0, 1)
        proj = self.path[:-1] + f[:, None] * seg
        d = np.hypot(*(proj - xy).T)
        k = int(np.argmin(d))
        return float(self.s[k] + f[k] * math.sqrt(L2[k])), float(d[k])

    def point_at(self, s):
        return np.array([np.interp(s, self.s, self.path[:, 0]), np.interp(s, self.s, self.path[:, 1])])

    def speed_after(self, s, horizon):
        """Mean predicted speed over `horizon` seconds after arc length s."""
        if self.s[-1] < 1e-6:
            return 0.0
        t0 = float(np.interp(s, self.s, self.times))
        s1 = float(np.interp(min(t0 + horizon, self.times[-1]), self.times, self.s))
        dt = min(t0 + horizon, self.times[-1]) - t0
        return (s1 - s) / dt if dt > 1e-3 else 0.0


class Follower:
    def __init__(self, params=None, safety_map=None):
        self.p = params or FollowerParams()
        if self.p.speed not in ("model", "fixed"):
            raise ValueError(f"unknown speed mode {self.p.speed}")
        self.map = safety_map
        self.outline = Outline(self.p.half_length, self.p.half_width)
        self.pred = None
        self.v = self.w = 0.0
        self.t_prev = None
        self.latched = None
        self._last_xy = None
        self._holding = None                # reason of the current brake (a new reason counts once)
        self._last_seq = None
        self.stats = {"time_running": 0.0, "distance": 0.0, "predictions_used": 0, "min_gap": math.inf,
                      "max_cross_track": 0.0, "holds": {}, "hold_time": {}, "latched": None,
                      "prediction_age_sum": 0.0, "control_steps": 0}

    def set_prediction(self, msg):
        p = Prediction.from_msg(msg)
        if self.pred is None or p.t > self.pred.t:
            self.pred = p

    def _zero(self, state, gap=math.inf):
        self.v = self.w = 0.0
        return Output(0.0, 0.0, state, gap)

    def _latch(self, reason, gap=math.inf):
        self.latched = reason
        self.stats["latched"] = reason
        return self._zero(f"stopped: {reason}", gap)

    def _hold(self, reason, dt, gap):
        """Brake with the acceleration limit, no steering once stopped (as the collector's waits)."""
        h, ht = self.stats["holds"], self.stats["hold_time"]
        if self._holding != reason:
            h[reason] = h.get(reason, 0) + 1
        self._holding = reason
        ht[reason] = ht.get(reason, 0.0) + dt
        self.v = max(0.0, self.v - self.p.accel_max * 2 * dt)
        self.w = 0.0 if self.v == 0 else self.w
        return Output(self.v, self.w, f"waiting: {reason}", gap)

    def update(self, t, pose, battery=None):
        """t: now (camera clock); pose: latest Vicon (x, y, yaw) or None if older than pose_timeout."""
        p = self.p
        dt = 0.0 if self.t_prev is None else min(max(t - self.t_prev, 0.0), 0.2)
        self.t_prev = t
        if self.latched:
            return self._zero(f"stopped: {self.latched}")
        if pose is None:
            return self._zero("no_pose")
        if battery is not None and p.min_battery_v > 0 and battery < p.min_battery_v:
            return self._latch(f"battery low ({battery:.2f} V)")
        pose = np.asarray(pose, float)
        gap = math.inf
        if self.map is not None:
            gap_fence, gap = pose_gaps(self.map, self.outline, pose)
            self.stats["min_gap"] = min(self.stats["min_gap"], gap)
            if gap_fence < 0:
                return self._latch("geofence", gap)
            if gap < p.stop_gap:
                return self._latch(f"too close to an obstacle (gap {gap:.3f} m)", gap)

        pr = self.pred
        if pr is None:
            return self._hold("no prediction yet", dt, gap)
        age = t - pr.t
        if age > p.prediction_timeout:
            return self._hold("prediction too old", dt, gap)
        s0, cross = pr.project(pose[:2])
        self.stats["max_cross_track"] = max(self.stats["max_cross_track"], cross)
        if cross > p.max_cross_track:
            return self._hold("off the predicted path", dt, gap)
        v_des = min(pr.speed_after(s0, p.speed_horizon) if p.speed == "model" else p.v_fixed, p.v_max)
        if v_des < p.v_stop:                               # before the path check: a stop is a zero-length path
            return self._hold("model predicts a stop", dt, gap)
        if pr.s[-1] - s0 < p.min_path_ahead:
            return self._hold("end of the predicted path", dt, gap)
        self._holding = None

        # pure pursuit on the predicted path (as rover_collector controller.PurePursuit)
        L = float(np.clip(p.lookahead_gain * max(v_des, 0.05), p.lookahead_min, p.lookahead_max))
        tx, ty = pr.point_at(min(s0 + L, pr.s[-1]))
        dx, dy = tx - pose[0], ty - pose[1]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        ld = max(math.hypot(lx, ly), 1e-3)
        kmax = 1.0 / p.r_min_ctrl
        kappa = max(-kmax, min(kmax, 2.0 * ly / (ld * ld)))
        v, w = v_des, v_des * kappa

        # limits: speed, yaw rate, accelerations (as the collector)
        v = min(max(v, 0.0), p.v_max)
        w = max(-p.w_max, min(p.w_max, w))
        if dt > 0:
            v = self.v + max(-p.accel_max * dt, min(p.accel_max * dt, v - self.v))
            w = self.w + max(-p.w_accel_max * dt, min(p.w_accel_max * dt, w - self.w))
        self.v, self.w = v, w

        st = self.stats
        st["time_running"] += dt
        st["control_steps"] += 1
        st["prediction_age_sum"] += age
        if self._last_xy is not None:
            st["distance"] += float(np.hypot(*(pose[:2] - self._last_xy)))
        self._last_xy = pose[:2].copy()
        if pr.seq != self._last_seq:
            st["predictions_used"] += 1
            self._last_seq = pr.seq
        return Output(v, w, "running", gap)

    def summary(self):
        st = dict(self.stats)
        n = max(st.pop("control_steps"), 1)
        st["mean_prediction_age_ms"] = round(1000 * st.pop("prediction_age_sum") / n, 1)
        st["params"] = asdict(self.p)
        return st
