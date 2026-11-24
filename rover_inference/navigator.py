"""
Navigator (no ROS): gives the model its intention, i.e. the chain of goals from which the inference node
computes the command (NAVSIM rule, as in training). It never drives and never sees the images.

  list    fixed goals in order (repeatable test routes); `loop` starts again from the first one
  random  "roam without a destination": the next goal is a free point 2-5 m away within +-75 deg of the
          direction of arrival, with the same parameters as the data-collection expert (collector.yaml
          goal_min_dist / goal_max_dist / goal_cone_deg, direct variant; re-implemented here, the
          collector sampler is not imported), so the commands look like the training ones and the
          rover stays inside the arena (the bounds are invisible to the model)

The chain sent is the current goal + the next ones (training: the chain of goals of the session;
with look-ahead 1.5 m the current and the next goal are what matters). A goal is reached when the
rover's reference point is within reach_radius of it.
"""
import math
from dataclasses import dataclass, fields

import numpy as np

from .safety import Outline


@dataclass
class NavigatorParams:
    mode: str = "list"                 # list | random
    reach_radius: float = 0.30         # m
    chain_length: int = 3              # goals sent: current + next ones
    loop: bool = True                  # list: start again after the last goal
    # random (as rover_collector direct variant)
    goal_min_dist: float = 2.0
    goal_max_dist: float = 5.0
    goal_cone_deg: float = 75.0
    goal_clearance: float = 0.15       # free space around the rover outline at the goal [m]
    bounds_margin: float = 0.40        # goals at least this far inside the bounds [m]
    seed: int = 0

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]


class Navigator:
    def __init__(self, params=None, goals=None, safety_map=None):
        self.p = params or NavigatorParams()
        if self.p.mode not in ("list", "random"):
            raise ValueError(f"unknown navigator mode {self.p.mode}")
        self.map = safety_map
        self.rng = np.random.default_rng(self.p.seed)
        self.outline = Outline()
        self.reached = 0
        self.log = []                                   # (time, goal) of every goal reached
        if self.p.mode == "list":
            if not goals:
                raise ValueError("mode list needs goals")
            self.route = [np.asarray(g[:2], float) for g in goals]
            if self.map is not None:
                bad = [i for i, g in enumerate(self.route) if not self.fits(g)]
                if bad:
                    raise ValueError(f"goals {bad} are inside / too close to an obstacle or outside the arena")
            self.idx = 0
        else:
            if self.map is None:
                raise ValueError("mode random needs the map (free space, bounds)")
            self.chain = []

    def fits(self, g):
        """The rover fits on g in at least one of 8 headings with goal_clearance of free space."""
        best = -math.inf
        for h in np.arange(8) * math.pi / 4:
            best = max(best, float(self.map.clearance(self.outline.world((g[0], g[1], h))).min()))
        x0, x1, y0, y1 = self.map.bounds
        m = self.p.bounds_margin
        return best >= self.p.goal_clearance and x0 + m <= g[0] <= x1 - m and y0 + m <= g[1] <= y1 - m

    # ------------------------------------------------------------------ random goals
    def _sample(self, origin, heading, tries=400):
        p = self.p
        for cone in (p.goal_cone_deg, 120.0, 180.0):      # cone empty (short side of the arena): widen it
            for _ in range(tries):
                d = self.rng.uniform(p.goal_min_dist, p.goal_max_dist)
                a = heading + math.radians(self.rng.uniform(-cone, cone))
                g = np.asarray(origin, float) + d * np.array([math.cos(a), math.sin(a)])
                if self.fits(g):
                    return g
        raise RuntimeError("no free goal found around the rover (map too crowded?)")

    def _fill(self, pose):
        while len(self.chain) < self.p.chain_length:
            if self.chain:
                prev = self.chain[-2] if len(self.chain) > 1 else np.asarray(pose[:2], float)
                last = self.chain[-1]
                heading = math.atan2(last[1] - prev[1], last[0] - prev[0])   # direction of arrival
                self.chain.append(self._sample(last, heading))
            else:
                self.chain.append(self._sample(pose[:2], pose[2]))

    # ------------------------------------------------------------------ update
    def update(self, t, pose):
        """Returns the chain of goals (list of [x, y]) for the current pose; advances reached goals."""
        xy = np.asarray(pose[:2], float)
        if self.p.mode == "list":
            if self.idx >= len(self.route):
                return []                                 # route finished (no loop): command unknown
            while self.idx < len(self.route) and np.hypot(*(self.route[self.idx] - xy)) < self.p.reach_radius:
                self._reached(t, self.route[self.idx])
                self.idx += 1
                if self.idx >= len(self.route) and self.p.loop:
                    self.idx = 0
            if self.idx >= len(self.route):
                return []
            n = len(self.route)
            k = [(self.idx + i) % n if self.p.loop else self.idx + i for i in range(self.p.chain_length)]
            return [self.route[i].tolist() for i in k if i < n]
        self._fill(pose)
        while self.chain and np.hypot(*(self.chain[0] - xy)) < self.p.reach_radius:
            self._reached(t, self.chain.pop(0))
            self._fill(pose)
        return [g.tolist() for g in self.chain]

    def _reached(self, t, g):
        self.reached += 1
        self.log.append((round(float(t), 2), [round(float(v), 3) for v in g]))
