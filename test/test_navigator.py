"""Navigator without ROS: list route (advance, loop, end), random goals (free, ahead, in range), and a
closed loop where an ideal rover drives towards the look-ahead point of the chain (training command rule)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rover_inference.commands import goals_command, point_along  # noqa: E402
from rover_inference.navigator import Navigator, NavigatorParams  # noqa: E402
from rover_inference.safety import SafetyMap  # noqa: E402

BOUNDS = {"x_min": -1.0, "x_max": 6.0, "y_min": -2.0, "y_max": 2.0}
SMAP = SafetyMap({"bounds": BOUNDS, "obstacles": [
    {"type": "circle", "center": [2.5, 0.0], "radius": 0.3},
    {"type": "box", "center": [4.5, 1.0], "size": [0.6, 0.4], "yaw_deg": 30.0}]})


def test_list_advance_loop_and_end():
    goals = [[1, 0], [3, 1], [3, -1]]
    nav = Navigator(NavigatorParams(mode="list", chain_length=2, loop=True), goals)
    assert nav.update(0, (0, 0, 0)) == [[1, 0], [3, 1]]
    assert nav.update(1, (0.8, 0.1, 0)) == [[3, 1], [3, -1]]        # within 0.30 m of goal 0
    assert nav.update(2, (3, -0.9, 0)) == [[3, 1], [3, -1]]          # far from goal 1: not skipped
    nav.update(3, (3, 1, 0))
    assert nav.update(4, (3, -1, 0)) == [[1, 0], [3, 1]]             # last goal -> loop
    assert nav.reached == 3 and [g for _, g in nav.log] == [[1, 0], [3, 1], [3, -1]]

    nav = Navigator(NavigatorParams(mode="list", chain_length=3, loop=False), goals)
    assert nav.update(0, (1, 0, 0)) == [[3, 1], [3, -1]]             # no wrap-around without loop
    nav.update(1, (3, 1, 0))
    assert nav.update(2, (3, -1, 0)) == []                           # route finished
    assert nav.update(3, (0, 0, 0)) == []


def test_list_goal_on_obstacle_refused():
    with pytest.raises(ValueError):
        Navigator(NavigatorParams(mode="list"), [[1, 0], [2.5, 0.1]], SMAP)
    with pytest.raises(ValueError):
        Navigator(NavigatorParams(mode="list"), [[5.9, 0]], SMAP)     # inside bounds, but not by 0.40 m
    Navigator(NavigatorParams(mode="list"), [[1, 0], [4, -1]], SMAP)
    with pytest.raises(ValueError):
        Navigator(NavigatorParams(mode="random"))                     # random needs the map


def test_random_goals_free_ahead_in_range():
    p = NavigatorParams(mode="random", chain_length=3, seed=3)
    nav = Navigator(p, safety_map=SMAP)
    rng = np.random.default_rng(0)
    pose = (0.0, 0.0, 0.0)
    for k in range(60):
        chain = [np.array(g) for g in nav.update(k, pose)]
        assert len(chain) == 3
        for g in chain:
            assert nav.fits(g)
        d = np.hypot(*(chain[0] - pose[:2]))
        assert d <= p.goal_max_dist + 1e-9
        pose = (chain[0][0], chain[0][1], rng.uniform(-math.pi, math.pi))   # jump onto the goal
    assert nav.reached >= 59
    # consecutive goals of the log: distance 2-5 m, mostly within the 75 deg cone of the arrival direction
    pts = np.array([g for _, g in nav.log])
    seg = np.diff(pts, axis=0)
    dist = np.hypot(*seg.T)
    assert dist.min() >= p.goal_min_dist - 1e-9 and dist.max() <= p.goal_max_dist + 1e-9
    turn = np.degrees(np.abs((np.diff(np.arctan2(seg[:, 1], seg[:, 0])) + np.pi) % (2 * np.pi) - np.pi))
    # 7 x 4 m test arena: from a goal near a wall the 75 deg cone is often empty -> widened to 120 / 180 deg
    assert np.mean(turn <= p.goal_cone_deg + 1e-6) > 0.3 and np.mean(turn <= 120 + 1e-6) > 0.8
    # same seed, same start -> same goals (repeatable runs)
    a = Navigator(p, safety_map=SMAP).update(0, (0, 0, 0))
    b = Navigator(p, safety_map=SMAP).update(0, (0, 0, 0))
    assert a == b


def test_closed_loop_ideal_rover_reaches_goals():
    """An ideal rover moving towards the look-ahead point (1.5 m along the chain) at 0.25 m/s reaches every
    goal in order; the commands it gets are all valid (never 'unknown' while the chain is not empty)."""
    for mode in ("list", "random"):
        nav = Navigator(NavigatorParams(mode=mode, seed=1),
                        [[1.0, 0.0], [4.0, -1.2], [4.0, 1.5], [0.5, 1.2]] if mode == "list" else None, SMAP)
        pose, dt, cmds = np.array([0.0, 0.0, 0.0]), 0.1, []
        for k in range(int(240 / dt)):
            chain = nav.update(k * dt, pose)
            cmd, _ = goals_command(pose, chain)
            cmds.append(cmd)
            p = point_along([pose[:2]] + [np.array(g) for g in chain], 0.4)
            head = math.atan2(p[1] - pose[1], p[0] - pose[0])
            pose = np.array([pose[0] + 0.25 * dt * math.cos(head), pose[1] + 0.25 * dt * math.sin(head), head])
        assert nav.reached >= 8, (mode, nav.reached)
        assert 3 not in cmds
        if mode == "list":
            assert [g for _, g in nav.log[:4]] == [[1.0, 0.0], [4.0, -1.2], [4.0, 1.5], [0.5, 1.2]]
