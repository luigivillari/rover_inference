"""
Tests of the inference pipeline without ROS and without the network (python3 -m pytest -q test/).

The equivalence tests compare with the training code when it is found next to this repository
(rover_collector/tools/dataset or ROVER_DATASET_TOOLS): same command rule, same image preprocessing.
"""
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rover_inference.commands import CMD_NAMES, goals_command, route_command, self_command  # noqa: E402
from rover_inference.ego import PoseBuffer, to_local  # noqa: E402
from rover_inference.frames import preprocess_image, decode_jpeg_rgb  # noqa: E402
from rover_inference.model_runner import ConstantVelocityRunner  # noqa: E402
from rover_inference.pipeline import InferencePipeline  # noqa: E402
from rover_inference.timing import ClockAligner  # noqa: E402


def dataset_tools():
    cands = [os.environ.get("ROVER_DATASET_TOOLS", ""),
             ROOT.parent / "rover_collector_separated" / "rover_collector" / "tools" / "dataset",
             ROOT.parent / "rover_collector" / "tools" / "dataset"]
    for c in cands:
        if c and (Path(c) / "driving_commands.py").exists():
            return Path(c)
    return None


# ------------------------------------------------------------------------------------------- clock
def test_clock_offset_from_lower_envelope():
    rng = np.random.default_rng(0)
    ca = ClockAligner("aligned")
    for k in range(3000):                                   # 30 s of Vicon at 100 Hz, lab clock 15.3 s ahead
        T = 100.0 + k * 0.01
        ca.add_vicon(T + 15.3, T + 0.006 + rng.uniform(0, 0.02))
        if k % 3 == 0:                                      # images: 2 ms from stamp to reader
            ca.add_image(T, T + 0.002 + rng.uniform(0, 0.003))
    # aligned = stamp - 15.3 + (min network delay 6 ms) - (min image delay 2 ms)
    assert abs(ca.offset() - (-15.3 + 0.006 - 0.002)) < 0.001


# ------------------------------------------------------------------------------------------- ego
def circle(t, R=1.0, w=0.3):
    return R * math.sin(w * t), R * (1 - math.cos(w * t)), w * t


def test_dynamics_on_a_circle():
    pb = PoseBuffer(keep_s=10, smooth_s=0.05)
    for k in range(1000):
        pb.add(k * 0.01, *circle(k * 0.01))
    _, v, a = pb.dynamics(5.0, mode="central")
    assert abs(v[0] - 0.3) < 0.002 and abs(v[1]) < 0.002          # forward speed R w
    assert abs(a[0]) < 0.002 and abs(a[1] - 0.09) < 0.002          # centripetal R w^2, to the left
    _, vb, ab = pb.dynamics(5.0, mode="backward")
    assert abs(math.hypot(*vb) - 0.3) < 0.003                      # same speed, direction of 0.1 s earlier
    assert vb[1] < -0.005                                         # chord of the last 0.2 s: right of the heading (-0.3 sin 0.03)


def test_out_of_order_poses_are_kept():
    a, b = PoseBuffer(keep_s=10), PoseBuffer(keep_s=10)
    ts = np.arange(0, 3, 0.01)
    rng = np.random.default_rng(1)
    order = np.argsort(ts + rng.uniform(0, 0.025, len(ts)))      # arrival order with Wi-Fi jitter
    for i in range(len(ts)):
        a.add(ts[i], *circle(ts[i]))
    for i in order:
        b.add(ts[i], *circle(ts[i]))
    assert len(b.t) == len(a.t)
    np.testing.assert_allclose(b.dynamics(1.5)[1], a.dynamics(1.5)[1], atol=1e-12)


def test_heading_wraps_continuously():
    pb = PoseBuffer(keep_s=10)
    for k in range(200):                                          # turning through +-pi
        yaw = math.pi - 0.5 + k * 0.005
        pb.add(k * 0.01, 0.0, 0.0, math.atan2(math.sin(yaw), math.cos(yaw)))
    pose = pb.at([1.0])[0]
    assert abs(math.cos(pose[2]) - math.cos(math.pi - 0.5 + 0.5)) < 1e-6


# ------------------------------------------------------------------------------------------- command
def test_goals_command_matches_training_rule():
    tools = dataset_tools()
    if tools is None:
        pytest.skip("rover_collector tools/dataset not found")
    sys.path.insert(0, str(tools))
    import driving_commands as dc
    rng = np.random.default_rng(2)
    for _ in range(500):
        origin = np.array([rng.uniform(-5, 5), rng.uniform(-2, 2), rng.uniform(-math.pi, math.pi)])
        goals = [rng.uniform(-6, 6, 2) for _ in range(rng.integers(1, 4))]
        idx, loc = goals_command(origin, goals, 1.5, 0.1)
        # extraction: route_lookahead point (rounded to 0.1 mm) then driving_commands.route_command
        ref = dc.route_command([round(float(loc[0]), 4), round(float(loc[1]), 4)], 1.5, 0.1)
        assert idx == ref


def test_goals_command_cases():
    o = np.array([0.0, 0.0, 0.0])
    assert CMD_NAMES[goals_command(o, [np.array([3.0, 0.05])])[0]] == "straight"
    assert CMD_NAMES[goals_command(o, [np.array([1.0, 0.0]), np.array([1.0, 2.0])])[0]] == "left"   # next goal to the left
    assert CMD_NAMES[goals_command(o, [np.array([-2.0, -0.3])])[0]] == "right"                     # goal behind, right side
    assert CMD_NAMES[goals_command(o, None)[0]] == "unknown"


def test_self_command():
    o = np.array([1.0, 0.0, 0.0])
    straight = (np.array([0.0, 0.0, 0.0]), np.array([[0.25 * k, 0, 0] for k in range(1, 9)], float))
    assert CMD_NAMES[self_command(o, straight)[0]] == "straight"
    left = (np.array([0.0, 0.0, 0.0]), np.array([[0.25 * k, 0.04 * k * k, 0] for k in range(1, 9)], float))
    assert CMD_NAMES[self_command(np.array([0.5, 0.04 * 4, 0.3]), left)[0]] == "left"
    assert CMD_NAMES[self_command(o, None)[0]] == "straight"


# ------------------------------------------------------------------------------------------- images
def test_preprocess_matches_training():
    tools = dataset_tools()
    if tools is None:
        pytest.skip("rover_collector tools/dataset not found")
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(tools))
    import rover_dataset as rd
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8), (9, 9), 3)
    ok, jpg = cv2.imencode(".jpg", img)
    path = ROOT / "test" / "_tmp_frame.jpg"
    path.write_bytes(jpg.tobytes())
    try:
        ref = rd.preprocess_image(rd.load_jpeg_rgb(path)).numpy()
    finally:
        path.unlink()
    ours = preprocess_image(decode_jpeg_rgb(jpg.tobytes()))
    assert ours.shape == (3, 256, 512)
    np.testing.assert_array_equal(ours, ref)


# ------------------------------------------------------------------------------------------- pipeline
def tiny_jpeg(k):
    img = np.full((36, 64, 3), k % 255, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.mark.parametrize("mode", ["central", "backward"])
def test_pipeline_real_time(mode):
    """Rover driving straight at 0.25 m/s; camera 30 fps (Jetson clock), Vicon 100 Hz (lab clock +15.3 s)."""
    p = InferencePipeline(ConstantVelocityRunner(), dynamics=mode, command_source="goals")
    p.on_goals([[10.0, 0.0]])
    rng = np.random.default_rng(4)
    events = []
    for k in range(int(4 / 0.01)):
        T = k * 0.01
        events.append((T + 0.01 + rng.uniform(0, 0.02), "v", T))
    for k in range(int(4 * 30)):
        T = k / 30
        events.append((T + 0.017 + 0.003, "i", T + 0.017))   # stamp 17 ms after the exposure
    events.sort()
    outs, reasons = [], set()
    for recv, kind, T in events:
        if kind == "v":
            p.on_vicon(T + 15.3, recv, 0.25 * T, 0.0, 0.0)
        else:
            p.on_image(T, recv, tiny_jpeg(int(T * 30)))
        res, why = p.step(recv)
        if res is not None:
            outs.append((recv, res))
        elif why:
            reasons.add(why)
    assert len(outs) > 50
    for recv, r in outs[10:]:
        assert r["command"] == "straight"
        assert abs(r["status"][4] - 0.25) < 0.01 and abs(r["status"][5]) < 0.01
        assert abs((r["frame_stamp_ns"] - r["prev_stamp_ns"]) * 1e-9 - 0.5) <= 0.034   # frame at t - 0.5 s
        assert abs(r["pose_world"][0] - 0.25 * r["t"]) < 0.01                        # Vicon pose at the frame time
    ages = [r["age_ms"] for _, r in outs[10:]]
    if mode == "central":
        assert min(ages) > 250        # waits for the poses after t (h_acc 0.25 s + smoothing)
    else:
        assert max(ages) < 150        # no wait: only the Vicon delay
