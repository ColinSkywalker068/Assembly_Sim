"""How far each YAM tip reaches towards the other, over the plane across its closing axis, from the tip meshes: what
the grasp planner (sim/pick.py) knows of where the fingers meet a fragment and what they sweep on the way down.

Coordinates are the TCP's with both finger joints at 0: h, height above the fingertips (along -approach); u, across
(the TCP's z: approach x closing); s, along the closing axis (the TCP's y). A cell's reach is the furthest the tip's
mesh comes along its inward direction (towards the other tip) within it, -inf where it has no material: the pads
reach 0.52 mm past the middle, the window between the rails not at all. At joint q a tip sits q further out, so it
meets a point at (h, u, s) when q = side * s + reach(h, u), side being +1 for the tip on the +s side; closing, each
tip stops at the largest such q over the fragment. Coming straight down open, a tip passes every section below a
point's height before reaching it, so clearing the fragment needs q above side * s + swept(h, u), the most any lower
section reaches. CELL_M bounds how far a reach is from the exact mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np

CELL_M = 0.0005


@dataclass(frozen=True)
class FingerReach:
    gripper: object
    cell_m: float
    lateral_min_m: float
    side: Mapping[str, float]
    reach: Mapping[str, np.ndarray]  # tip -> (heights, laterals) inward reach at joint 0, -inf where it has nothing
    swept: Mapping[str, np.ndarray]  # tip -> the most any section at or below each height reaches

    def at(self, table, heights, laterals):
        """`table`'s value at each (height, lateral); -inf off the finger."""
        rows = np.floor(np.asarray(heights, dtype=float) / self.cell_m).astype(int)
        cols = np.floor((np.asarray(laterals, dtype=float) - self.lateral_min_m) / self.cell_m).astype(int)
        inside = (rows >= 0) & (rows < table.shape[0]) & (cols >= 0) & (cols < table.shape[1])
        values = np.full(rows.shape, -np.inf)
        values[inside] = table[rows[inside], cols[inside]]
        return values

    def _joints(self, tables, heights, laterals, along):
        return {tip: float(np.max(self.side[tip] * np.asarray(along, dtype=float)
                                  + self.at(tables[tip], heights, laterals), initial=-np.inf)) for tip in self.side}

    def stop_joints(self, heights, laterals, along):
        """tip -> the joint at which it first meets one of the points, closing; -inf if it meets none."""
        return self._joints(self.reach, heights, laterals, along)

    def clear_joints(self, heights, laterals, along):
        """tip -> the joint above which it comes straight down past every point without touching one."""
        return self._joints(self.swept, heights, laterals, along)

    def stop_width(self, heights, laterals, along):
        """The width between the pads when the coupled fingers, closing together, stop on the points: at the joint
        where the first tip meets one (at the pads' contact if neither meets any)."""
        joints = self.stop_joints(heights, laterals, along)
        low, high = self.gripper.contact_joint_m, self.gripper.travel_m
        return self.gripper.width(float(np.clip(max(joints.values()), low, high)))


def finger_reach(gripper, cell_m=CELL_M, spacing_m=None):
    """The FingerReach of `gripper` (core.robots.linear_4310) from its tip meshes sampled every `spacing_m`."""
    from .yam_usd import tip_points_in_link
    spacing_m = cell_m / 2 if spacing_m is None else spacing_m
    tcp = gripper.mount_T_tcp
    approach = tcp[:3, 2]
    closing = np.array([1.0, 0.0, 0.0])  # the mount's x is the TCP's closing axis (s0.kinematics.YAM_MOUNT_ROTATION)
    across = np.cross(approach, closing)
    points = {}
    for tip in ('tip_left', 'tip_right'):
        frame = gripper.link_frames[tip]
        points[tip] = tip_points_in_link(gripper, tip, spacing_m) @ frame[:3, :3].T + frame[:3, 3] - tcp[:3, 3]
    end = max(float((tip_points @ approach).max()) for tip_points in points.values())
    lateral_min = min(float((tip_points @ across).min()) for tip_points in points.values()) - cell_m
    rows = int(np.ceil((end - min(float((p @ approach).min()) for p in points.values())) / cell_m)) + 1
    cols = int(np.ceil((max(float((p @ across).max()) for p in points.values()) - lateral_min) / cell_m)) + 2
    side, reach, swept = {}, {}, {}
    for tip, tip_points in points.items():
        outward = np.asarray(gripper.slide_axes[tip], dtype=float) / np.linalg.norm(gripper.slide_axes[tip])
        side[tip] = float(np.sign(outward @ closing))
        row = np.floor((end - tip_points @ approach) / cell_m).astype(int)
        col = np.floor((tip_points @ across - lateral_min) / cell_m).astype(int)
        table = np.full((rows, cols), -np.inf)
        np.maximum.at(table, (row, col), tip_points @ -outward)
        reach[tip] = table
        swept[tip] = np.maximum.accumulate(table, axis=0)
    return FingerReach(gripper, cell_m, lateral_min, side, reach, swept)


@lru_cache(maxsize=1)
def yam_finger_reach():
    """The FingerReach of the I2RT linear_4310 (computed once)."""
    from core.robots.linear_4310 import load_linear_4310
    return finger_reach(load_linear_4310())
