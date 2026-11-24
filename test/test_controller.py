"""Tests of the controller (follower + safety), no ROS: python3 -m pytest -q test/"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rover_inference.ego import to_local  # noqa: E402
from rover_inference.follower import Follower, FollowerParams  # noqa: E402
from rover_inference.safety import Outline, SafetyMap, pose_gaps  # noqa: E402

ARENA = {"bounds": {"x_min": -1.0, "x_max": 6.0, "y_min": -2.0, "y_max": 2.0}}


class Unicycle:
    """Same model as rover_collector kinematics.Unicycle (motor response tau, acceleration limits)."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, tau=0.12, a_max=1.0, alpha_max=6.0):
        self.x, self.y, self.yaw, self.v, self.w = x, y, yaw, 0.0, 0.0
        self.tau, self.a_max, self.alpha_max = tau, a_max, alpha_max

    def step(self, v_cmd, w_cmd, dt):
        k = min(1.0, dt / self.tau)
        self.v += np.clip((v_cmd - self.v) * k, -self.a_max * dt, self.a_max * dt)
        self.w += np.clip((w_cmd - self.w) * k, -self.alpha_max * dt, self.alpha_max * dt)
        mid = self.yaw + 0.5 * self.w * dt
        self.x += self.v * math.cos(mid) * dt
        self.y += self.v * math.sin(mid) * dt
        self.yaw = (self.yaw + self.w * dt + math.pi) % (2 * math.pi) - math.pi
        return np.array([self.x, self.y, self.yaw])


def reference(n=2000):
    """1 m straight, left arc of radius 0.8 m over 90 deg, then straight up: (n, 2)."""
    pts = [(x, 0.0) for x in np.linspace(0, 1, 100)]
    pts += [(1 + 0.8 * math.sin(a), 0.8 - 0.8 * math.cos(a)) for a in np.linspace(0, math.pi / 2, 200)[1:]]
    pts += [(1.8, 0.8 + y) for y in np.linspace(0, 2, 200)[1:]]
    return np.array(pts)


def ideal_prediction(ref, pose, t, v=0.22, seq=0):
    """What a perfect model would predict at the frame pose: 8 poses along the reference at speed v."""
    s_ref = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ref, axis=0).T))])
    k = int(np.argmin(np.hypot(*(ref - pose[:2]).T)))
    s = s_ref[k] + v * 0.5 * np.arange(1, 9)
    pts = np.stack([np.interp(s, s_ref, ref[:, 0]), np.interp(s, s_ref, ref[:, 1])], 1)
    head = np.arctan2(*np.diff(np.vstack([ref[k], pts]), axis=0)[:, ::-1].T)
    traj = to_local(pose, np.column_stack([pts, head]))
    return {"t": t, "pose_world": pose.tolist(), "trajectory": traj.tolist(), "seq": seq}


def cross_track(ref, xy):
    return float(np.min(np.hypot(*(ref - xy).T)))


def test_closed_loop_with_latency_follows_the_path():
    """Predictions every 0.3 s computed on the frame pose of 0.3 s earlier (Jetson inference)."""
    ref = reference()
    f = Follower(FollowerParams(v_max=0.25), SafetyMap(ARENA))
    rover = Unicycle()
    dt, history, errs, seq = 0.05, [], [], 0
    pose = np.zeros(3)
    for k in range(int(14 / dt)):
        t = k * dt
        history.append((t, pose.copy()))
        if k % 6 == 0 and t >= 0.3:                       # a new prediction every 0.3 s ...
            t_frame, pose_frame = history[-7]            # ... made on the frame taken 0.3 s ago
            seq += 1
            f.set_prediction(ideal_prediction(ref, pose_frame, t_frame, seq=seq))
        out = f.update(t, pose)
        pose = rover.step(out.v, out.w, dt)
        if t > 2.0:
            errs.append(cross_track(ref, pose[:2]))
    assert f.latched is None
    assert max(errs) < 0.03, f"max distance from the reference {max(errs):.3f} m"
    assert pose[1] > 1.5                                 # went round the curve and up the last straight
    assert abs(f.stats["distance"] / f.stats["time_running"] - 0.22) < 0.03   # model speed used


def test_brakes_when_predictions_stop():
    ref = reference()
    f = Follower(FollowerParams(), SafetyMap(ARENA))
    rover, pose, dt = Unicycle(), np.zeros(3), 0.05
    for k in range(int(6 / dt)):
        t = k * dt
        if k % 6 == 0 and t < 3.0:                       # inference stops after 3 s
            f.set_prediction(ideal_prediction(ref, pose, t, seq=k))
        out = f.update(t, pose)
        pose = rover.step(out.v, out.w, dt)
    assert out.v == 0 and out.state == "waiting: prediction too old"
    assert f.stats["holds"]["prediction too old"] == 1


def test_no_pose_gives_zero_at_once():
    f = Follower(FollowerParams(), SafetyMap(ARENA))
    f.set_prediction(ideal_prediction(reference(), np.zeros(3), 0.0))
    for k in range(20):
        f.update(k * 0.05, np.array([0.02 * k, 0, 0]))
    out = f.update(1.0, None)
    assert (out.v, out.w, out.state) == (0.0, 0.0, "no_pose")


def test_geofence_and_obstacle_stops_are_latched():
    smap = SafetyMap({"bounds": ARENA["bounds"],
                      "obstacles": [{"type": "box", "center": [1.0, 0.0], "size": [0.2, 0.2], "yaw_deg": 0}]})
    f = Follower(FollowerParams(), smap)
    f.set_prediction(ideal_prediction(reference(), np.zeros(3), 0.0))
    out = f.update(0.0, np.array([0.795, 0.0, 0.0]))     # front of the outline 5 mm from the box
    assert out.state.startswith("stopped: too close") and f.update(0.1, np.zeros(3)).v == 0.0
    g = Follower(FollowerParams(), smap)
    assert g.update(0.0, np.array([5.95, 0.0, 0.0])).state == "stopped: geofence"


def test_off_path_and_end_of_path_brake():
    f = Follower(FollowerParams(), SafetyMap(ARENA))
    f.set_prediction(ideal_prediction(reference(), np.zeros(3), 0.0))
    assert f.update(0.1, np.array([0.3, 0.5, 0.0])).state == "waiting: off the predicted path"
    # prediction at 0.22 m/s from the origin: 8 x 0.5 s -> path ends at x = 0.88 m
    assert f.update(0.2, np.array([0.85, 0.0, 0.0])).state == "waiting: end of the predicted path"


def test_speed_from_the_model_is_capped():
    f = Follower(FollowerParams(v_max=0.25, accel_max=100), SafetyMap(ARENA))
    f.set_prediction(ideal_prediction(reference(), np.zeros(3), 0.0, v=0.15))
    assert abs(f.update(0.05, np.zeros(3)).v - 0.15) < 0.01
    f.set_prediction(ideal_prediction(reference(), np.zeros(3), 0.1, v=0.40))
    assert abs(f.update(0.15, np.zeros(3)).v - 0.25) < 1e-9
    f.set_prediction(ideal_prediction(reference(), np.zeros(3), 0.2, v=0.0))
    assert f.update(0.25, np.zeros(3)).state == "waiting: model predicts a stop"


def test_battery_low_latches():
    f = Follower(FollowerParams(), SafetyMap(ARENA))
    assert f.update(0.0, np.zeros(3), battery=10.2).state.startswith("stopped: battery low")


def test_walls_and_rotated_boxes_as_in_the_collector():
    smap = SafetyMap({"bounds": ARENA["bounds"], "obstacles": [
        {"type": "wall", "points": [[0, 1], [2, 1]], "thickness": 0.04},
        {"type": "box", "center": [3, 0], "size": [0.4, 0.2], "yaw_deg": 90}]})
    o = Outline()
    assert abs(pose_gaps(smap, o, np.array([1.0, 0.75, 0.0]))[1] - (0.25 - 0.10 - 0.02)) < 1e-9
    # box rotated 90 deg: its 0.4 m side along y -> half width 0.1 along x
    assert abs(pose_gaps(smap, o, np.array([2.5, 0.0, 0.0]))[1] - (0.5 - 0.10 - 0.10)) < 1e-9
