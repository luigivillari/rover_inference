"""
Safety geometry of the controller: arena map (geofence + static obstacles) and the rover outline.

Copy of the parts of rover_collector (arena.py: Obstacle, shape_from_cfg, walls_to_boxes, clearance;
footprint.py: rectangle outline) used by the data-collection driver for its stops, so the controller stops
the rover with the SAME rules on the SAME map file (maps/lab_latest.yaml of rover_collector):
  - geofence: any point of the outline outside the arena bounds
  - obstacle: outline closer than stop_gap (1 cm) to an obstacle or to the bounds
The model never sees the map: this is only the safety net (and every stop is counted as an intervention).
"""
import math

import numpy as np
import yaml


class Obstacle:
    def __init__(self, kind, center, half_size=None, yaw=0.0, radius=0.0):
        self.kind, self.center = kind, np.asarray(center, float)
        self.half_size = None if half_size is None else np.asarray(half_size, float)
        self.yaw, self.radius = float(yaw), float(radius)

    def signed_distance(self, pts):
        """Distance from points (N, 2) to the obstacle boundary: > 0 outside, < 0 inside."""
        d = np.atleast_2d(pts) - self.center
        if self.kind == "circle":
            return np.hypot(d[:, 0], d[:, 1]) - self.radius
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local = np.abs(np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1))
        q = local - self.half_size
        outside = np.hypot(np.maximum(q[:, 0], 0), np.maximum(q[:, 1], 0))
        inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0)
        return outside + inside


def walls_to_boxes(points, thickness):
    pts = np.asarray(points, float)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        length = float(np.hypot(*d))
        out.append(Obstacle("box", (a + b) / 2, np.array([length + thickness, thickness]) / 2, math.atan2(d[1], d[0])))
    return out


class SafetyMap:
    """Bounds + static obstacles of an arena YAML (rover_collector format; moving obstacles ignored)."""

    def __init__(self, cfg, name=""):
        b = cfg["bounds"]
        self.bounds = (float(b["x_min"]), float(b["x_max"]), float(b["y_min"]), float(b["y_max"]))
        self.name = cfg.get("name", name)
        self.obstacles = []
        for o in cfg.get("obstacles", []) or []:
            if o["type"] == "wall":
                self.obstacles += walls_to_boxes(o["points"], float(o.get("thickness", 0.04)))
            elif o["type"] == "box":
                self.obstacles.append(Obstacle("box", o["center"], np.array(o["size"], float) / 2,
                                               math.radians(o.get("yaw_deg", 0.0))))
            elif o["type"] == "circle":
                self.obstacles.append(Obstacle("circle", o["center"], radius=float(o["radius"])))
            else:
                raise ValueError(f"unknown obstacle type {o['type']}")

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls(yaml.safe_load(f), name=str(path))

    def clearance(self, pts, obstacles=True, bounds=True):
        pts = np.atleast_2d(pts)
        d = np.full(len(pts), np.inf)
        if bounds:
            x0, x1, y0, y1 = self.bounds
            d = np.min(np.stack([pts[:, 0] - x0, x1 - pts[:, 0], pts[:, 1] - y0, y1 - pts[:, 1]]), axis=0)
        if obstacles:
            for o in self.obstacles:
                d = np.minimum(d, o.signed_distance(pts))
        return d


class Outline:
    """Rectangle outline of the rover (samples every ds + centre), rover frame, Vicon origin = (0, 0)."""

    def __init__(self, half_length=0.10, half_width=0.10, ds=0.02):
        corners = np.array([[half_length, half_width], [-half_length, half_width], [-half_length, -half_width],
                            [half_length, -half_width], [half_length, half_width]])
        pts = [np.zeros((1, 2))]
        for a, b in zip(corners[:-1], corners[1:]):
            k = max(1, int(math.ceil(np.hypot(*(b - a)) / ds)))
            pts.append(a + np.linspace(0, 1, k, endpoint=False)[:, None] * (b - a))
        self.local = np.vstack(pts)

    def world(self, pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return np.stack([pose[0] + c * self.local[:, 0] - s * self.local[:, 1],
                         pose[1] + s * self.local[:, 0] + c * self.local[:, 1]], axis=1)


def pose_gaps(smap, outline, pose):
    """(gap to the geofence only, gap to obstacles or geofence) of the rover outline [m]; < 0 = overlap."""
    pts = outline.world(pose)
    return float(smap.clearance(pts, obstacles=False).min()), float(smap.clearance(pts).min())
