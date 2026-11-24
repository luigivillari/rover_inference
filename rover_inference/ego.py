"""
Ego state from the Vicon stream: pose at a given time, velocity and acceleration in the rover frame.

Same computation as extract_samples.py (PoseTrack + ego_dynamics) so that the status vector the model
receives on the rover matches the one it was trained on:
  - poses smoothed with a centred 50 ms moving average, then linearly interpolated;
  - velocity  = central difference over +-h_v (0.1 s);
  - acceleration = second difference over +-h_a (0.25 s);
  - both rotated into the rover frame at time t.

The training values are CENTRED: they need Vicon poses up to t + 0.25 s. On the rover there are two ways:
  'central'   wait until the poses after t have arrived (the frame used is ~0.25-0.3 s old) -> exactly the
              training status, more latency;
  'backward'  only poses up to t: v = (p(t) - p(t - 2 h_v)) / 2 h_v, a = (p(t) - 2 p(t - h_a) + p(t - 2 h_a)) / h_a^2
              -> no extra latency, but these are the velocity / acceleration of 0.1 / 0.25 s earlier.
Which one is better is measured with tools/replay_bag.py (both modes on a recorded bag).
"""
import bisect
import math

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def to_local(origin, pts):
    """World (x, y, yaw) -> frame of origin (x forward, y left). Same as extract_samples.to_local."""
    pts = np.atleast_2d(np.asarray(pts, float))
    c, s = math.cos(origin[2]), math.sin(origin[2])
    d = pts[:, :2] - np.asarray(origin[:2], float)
    x = c * d[:, 0] + s * d[:, 1]
    y = -s * d[:, 0] + c * d[:, 1]
    if pts.shape[1] >= 3:
        return np.stack([x, y, wrap(pts[:, 2] - origin[2])], axis=1)
    return np.stack([x, y], axis=1)


def to_world(origin, pts):
    """Frame of origin -> world, (x, y[, heading])."""
    pts = np.atleast_2d(np.asarray(pts, float))
    c, s = math.cos(origin[2]), math.sin(origin[2])
    x = origin[0] + c * pts[:, 0] - s * pts[:, 1]
    y = origin[1] + s * pts[:, 0] + c * pts[:, 1]
    if pts.shape[1] >= 3:
        return np.stack([x, y, origin[2] + pts[:, 2]], axis=1)
    return np.stack([x, y], axis=1)


class PoseBuffer:
    """The last keep_s seconds of Vicon poses on the camera time base."""

    def __init__(self, keep_s=3.0, smooth_s=0.05, max_gap_s=0.1):
        self.keep_s, self.smooth_s, self.max_gap_s = keep_s, smooth_s, max_gap_s
        self.t, self.p = [], []                              # sorted by stamp
        self._cache = None

    def add(self, t, x, y, yaw):
        """Vicon over Wi-Fi arrives out of order (~12 % of the messages in the synthetic bag, jitter > 10 ms
        period): a late pose is inserted at its place, not dropped (dropping it changed the status)."""
        t = float(t)
        i = bisect.bisect_left(self.t, t)
        if i < len(self.t) and self.t[i] == t:               # duplicate stamp
            return
        ref = self.p[i - 1][2] if i > 0 else (self.p[0][2] if self.p else yaw)
        yaw = ref + wrap(yaw - ref)                          # continuous heading (no jump at +-pi)
        self.t.insert(i, t)
        self.p.insert(i, (float(x), float(y), float(yaw)))
        if self.t[0] < self.t[-1] - self.keep_s - 0.5:       # trim in chunks
            k = bisect.bisect_left(self.t, self.t[-1] - self.keep_s)
            del self.t[:k], self.p[:k]
        self._cache = None

    @property
    def latest(self):
        return self.t[-1] if self.t else None

    def _arrays(self):
        if self._cache is None:
            t = np.array(self.t)
            p = np.array(self.p)
            if self.smooth_s > 0 and len(t) > 2:             # centred moving average (truncated at the ends)
                half = self.smooth_s / 2
                csum = np.vstack([np.zeros((1, 3)), np.cumsum(p, axis=0)])
                lo = np.searchsorted(t, t - half, side="left")
                hi = np.searchsorted(t, t + half, side="right")
                p = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
            self._cache = (t, p)
        return self._cache

    def covered(self, t0, t1):
        """Poses span [t0, t1] without a gap longer than max_gap_s."""
        t, _ = self._arrays()
        if len(t) < 2 or t[0] > t0 or t[-1] < t1:
            return False
        i0 = max(int(np.searchsorted(t, t0, side="right")) - 1, 0)
        i1 = min(int(np.searchsorted(t, t1, side="left")), len(t) - 1)
        return bool(np.all(np.diff(t[i0:i1 + 1]) <= self.max_gap_s))

    def at(self, times):
        t, p = self._arrays()
        times = np.atleast_1d(np.asarray(times, float))
        return np.stack([np.interp(times, t, p[:, k]) for k in range(3)], axis=1)

    def needs(self, mode, h_v, h_a):
        """(seconds before t, seconds after t) of poses needed by dynamics()."""
        if mode == "central":
            return max(h_v, h_a) + self.smooth_s, max(h_v, h_a) + self.smooth_s / 2
        # backward: only the 25 ms of the centred smoothing after t (a truncated window lags: -0.01 m/s)
        return 2 * max(h_v, h_a) + self.smooth_s, self.smooth_s / 2

    def dynamics(self, t, h_v=0.1, h_a=0.25, mode="central"):
        """(pose at t, velocity (2,), acceleration (2,)) in the rover frame at t."""
        if mode == "central":
            pm, p0, pp = self.at([t - h_v, t, t + h_v])
            v_w = (pp[:2] - pm[:2]) / (2 * h_v)
            am, ap = self.at([t - h_a, t + h_a])
            a_w = (ap[:2] - 2 * p0[:2] + am[:2]) / (h_a * h_a)
        elif mode == "backward":
            p0, pv = self.at([t, t - 2 * h_v])
            v_w = (p0[:2] - pv[:2]) / (2 * h_v)
            a1, a2 = self.at([t - h_a, t - 2 * h_a])
            a_w = (p0[:2] - 2 * a1[:2] + a2[:2]) / (h_a * h_a)
        else:
            raise ValueError(f"unknown dynamics mode {mode}")
        c, s = math.cos(p0[2]), math.sin(p0[2])
        rot = np.array([[c, s], [-s, c]])                    # world -> rover
        return np.array([p0[0], p0[1], wrap(p0[2])]), rot @ v_w, rot @ a_w
