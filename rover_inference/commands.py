"""
Driving command of the status vector (one-hot left / straight / right / unknown).

The model was trained with the 'intention' command of extract_samples.py v3: the NAVSIM rule on the chain
of goals (rover -> current goal -> next goals): walk L = 1.5 m along it, left/right if that point is more
than 0.1 * L to the side, else straight; unknown without goals. route_command and the look-ahead walk are
copies of tools/dataset/driving_commands.py and extract_samples.route_lookahead (rover_collector), kept
here so this service does not depend on the collector code. Same rule = same command semantics as in training.

Sources (config `command.source`):
  goals  chain of goals from the navigator (/drive_jepa/goals): exactly the training rule (default)
  topic  command name from a topic (/drive_jepa/command), e.g. a person or another navigator
  fixed  always the same command (e.g. straight): the rover goes forward and avoids what it sees
  self   from the model's previous prediction (walk L along it): no external reference, experimental
         (the rover wanders; a curve it starts reinforces itself)
"""
import numpy as np

from .ego import to_local, to_world

CMD_NAMES = ["left", "straight", "right", "unknown"]


def route_command(point, lookahead, ratio):
    """Index in CMD_NAMES for a look-ahead point (x, y) in the rover frame (x forward, y left):
    left/right if |y| > ratio * look-ahead, else straight. A point behind the rover is a turn towards its side.
    (copy of rover_collector tools/dataset/driving_commands.py)"""
    x, y = point
    if x < 0:
        return 0 if y >= 0 else 2
    return 0 if y > ratio * lookahead else 2 if y < -ratio * lookahead else 1


def point_along(polyline, L):
    """Point at arc length L along a polyline (N, 2); beyond its end, along the last segment
    (as extract_samples.route_lookahead). None if the polyline has no length."""
    pts = [np.asarray(polyline[0], float)]
    for q in polyline[1:]:
        if np.hypot(*(np.asarray(q, float) - pts[-1])) > 1e-6:
            pts.append(np.asarray(q, float))
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    j = min(int(np.searchsorted(cum, L, side="right")) - 1, len(seg) - 1)
    return pts[j] + seg[j] / seglen[j] * (L - cum[j])


def goals_command(origin, goals, lookahead=1.5, ratio=0.1):
    """Training rule: look-ahead point along rover -> goals[0] -> goals[1] ... (world frame)."""
    if not goals:
        return 3, None
    p = point_along([origin[:2]] + [g[:2] for g in goals], lookahead)
    if p is None:                                         # rover exactly on the only goal
        return 3, None
    loc = to_local(origin, [p])[0]
    return route_command(loc, lookahead, ratio), loc


def self_command(origin, prev, lookahead=1.5, ratio=0.1):
    """From the previous prediction: prev = (pose_world at its t, trajectory (8, 3) in that frame).
    The look-ahead walk starts at the projection of the current position on the predicted path."""
    if prev is None:
        return 1, None
    anchor, traj = prev
    path = np.vstack([np.asarray(anchor[:2], float)[None], to_world(anchor, traj[:, :2])])
    seg = np.diff(path, axis=0)
    seglen2 = np.maximum((seg ** 2).sum(1), 1e-12)
    rel = np.asarray(origin[:2], float) - path[:-1]
    f = np.clip((rel * seg).sum(1) / seglen2, 0, 1)
    proj = path[:-1] + f[:, None] * seg
    k = int(np.argmin(np.hypot(*(proj - origin[:2]).T)))
    p = point_along(np.vstack([proj[k][None], path[k + 1:]]), lookahead)
    if p is None:
        return 1, None
    loc = to_local(origin, [p])[0]
    return route_command(loc, lookahead, ratio), loc


def onehot(idx):
    v = [0.0] * 4
    v[idx] = 1.0
    return v
