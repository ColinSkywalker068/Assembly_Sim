"""I2RT linear_4310 parallel gripper (the YAM "Linear 4310"), read from the official model.

core/assets/robots/i2rt_linear_4310 holds the unmodified upstream MJCF, meshes and licence, pinned by SOURCE.json
(i2rt-robotics/i2rt at a fixed commit); `load_linear_4310` refuses files that differ from it. Everything here is in
the gripper's mount frame, the YAM joint6 output: +z is the approach axis, +x right, +y down.

- Meshes are placed as MuJoCo places them: raw vertices at the geom pose, bodies composed from their pos/quat
  (quaternions normalized), each tip translated along its slide axis by the joint value q.
- The two fingers move together (an equality constraint makes joint8 follow joint7), each sliding 0..travel_m away
  from the centre.
- The TCP is the MJCF's grasp_site, the centre of the two fingertip end facets. It does not move with q, because
  the fingers slide transversely.
- Width is the gap between the pad faces, measured from the meshes over the last PAD_DEPTH_M before the fingertip
  ends. At q = 0 the pads overlap by 1 mm (the MJCF excludes contacts between the tips), so they meet at
  q = contact_joint_m and open to max_width_m; a smaller q reads as width 0.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from core.robots.kinematics import T

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots" / "i2rt_linear_4310"
MJCF_NAME = "linear_4310.xml"
LINKS = ("gripper", "tip_left", "tip_right")
TIPS = ("tip_left", "tip_right")
PAD_DEPTH_M = 0.03
UNSUPPORTED_ORIENTATIONS = ("euler", "axisangle", "xyaxes", "zaxis")


def read_binary_stl(path) -> np.ndarray:
    """Triangle vertices (3N, 3) of a binary STL."""
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path}: not a binary STL")
    count = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path}: binary STL size does not match its triangle count")
    record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(data, dtype=record, count=count, offset=84)
    return triangles["vertices"].reshape(-1, 3).astype(float)


def verify_assets(asset_dir) -> None:
    """Raise unless every file listed in SOURCE.json has its pinned size and sha256."""
    source = json.loads((Path(asset_dir) / "SOURCE.json").read_text(encoding="utf-8"))
    for name, record in source["files"].items():
        data = (Path(asset_dir) / name).read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"{name} differs from the pinned upstream file ({source['repository']} "
                             f"{source['commit'][:7]})")


def _floats(element, name, default):
    text = element.get(name)
    return np.array(default if text is None else [float(value) for value in text.split()], dtype=float)


def _pose(element) -> np.ndarray:
    """Local pose of a body, geom or site from its pos and quat (w x y z) attributes."""
    unsupported = [name for name in UNSUPPORTED_ORIENTATIONS if element.get(name) is not None]
    if unsupported:
        raise ValueError(f"{element.get('name')}: orientation given as {unsupported[0]}; only quat is read")
    quat = _floats(element, "quat", (1.0, 0.0, 0.0, 0.0))
    norm = float(np.linalg.norm(quat))
    if quat.shape != (4,) or not norm > 0:
        raise ValueError(f"{element.get('name')}: invalid quat")
    return T(_floats(element, "pos", (0.0, 0.0, 0.0)), quat / norm)


def _transform(pose, points):
    return points @ pose[:3, :3].T + pose[:3, 3]


def _read_only(array):
    array = np.array(array, dtype=float)
    array.flags.writeable = False
    return array


def _child(parent, tag, name):
    found = [element for element in parent.findall(tag) if element.get("name") == name]
    if len(found) != 1:
        raise ValueError(f"expected one {tag} named {name!r}")
    return found[0]


def _mesh_geom(body, meshes, mesh_dir):
    geoms = [geom for geom in body.findall("geom") if geom.get("type") == "mesh"]
    if len(geoms) != 1 or geoms[0].get("mesh") not in meshes:
        raise ValueError(f"{body.get('name')}: expected one mesh geom")
    mesh = meshes[geoms[0].get("mesh")]
    if any(mesh.get(name) is not None for name in ("scale", "refpos", "refquat")):
        raise ValueError(f"mesh {mesh.get('name')}: scale/refpos/refquat are not read")
    return _pose(geoms[0]), mesh_dir / mesh.get("file")


def _slide_joint(tip):
    joints = tip.findall("joint")
    if len(joints) != 1 or joints[0].get("type") != "slide":
        raise ValueError(f"{tip.get('name')}: expected one slide joint")
    axis = _floats(joints[0], "axis", (0.0, 0.0, 1.0))
    low, high = _floats(joints[0], "range", (0.0, 0.0))
    if not math.isclose(low, 0.0) or not high > 0:
        raise ValueError(f"{tip.get('name')}: expected a slide range 0..travel")
    return joints[0].get("name"), axis / np.linalg.norm(axis), float(high)


def _check_coupling(root, names):
    joints = root.findall("./equality/joint")
    if (len(joints) != 1 or {joints[0].get("joint1"), joints[0].get("joint2")} != set(names)
            or not np.allclose(_floats(joints[0], "polycoef", ()), (0.0, 1.0, 0.0, 0.0, 0.0))):
        raise ValueError("expected the two finger joints coupled 1:1 by one equality constraint")


@dataclass(frozen=True)
class Inertial:
    """A link's mass properties in its link frame, as the MJCF gives them."""

    mass_kg: float
    center_of_mass_m: np.ndarray
    principal_axes_wxyz: np.ndarray
    diagonal_inertia_kg_m2: np.ndarray


def _inertial(body) -> Inertial:
    element = body.find("inertial")
    quat = _floats(element, "quat", (1.0, 0.0, 0.0, 0.0))
    return Inertial(float(element.get("mass")), _read_only(_floats(element, "pos", (0.0, 0.0, 0.0))),
                    _read_only(quat / np.linalg.norm(quat)), _read_only(_floats(element, "diaginertia", ())))


@dataclass(frozen=True)
class Linear4310:
    """The gripper in its mount frame. `link_frames` and `mesh_poses` are the q = 0 link poses and each link's mesh
    pose in its link frame; `tip_points` are at q = 0."""

    link_frames: Mapping[str, np.ndarray]
    mesh_poses: Mapping[str, np.ndarray]
    mesh_files: Mapping[str, Path]
    mount_T_tcp: np.ndarray
    body_points: np.ndarray
    tip_points: Mapping[str, np.ndarray]
    slide_axes: Mapping[str, np.ndarray]
    travel_m: float
    closed_gap_m: float
    inertials: Mapping[str, Inertial]

    @property
    def masses_kg(self) -> Mapping[str, float]:
        return MappingProxyType({link: inertial.mass_kg for link, inertial in self.inertials.items()})

    @property
    def contact_joint_m(self) -> float:
        """Finger joint value at which the pad faces meet."""
        return max(0.0, -self.closed_gap_m / 2)

    @property
    def max_width_m(self) -> float:
        return 2 * self.travel_m + self.closed_gap_m

    def tips_at(self, q) -> Mapping[str, np.ndarray]:
        q = self._joint(q)
        return {tip: self.tip_points[tip] + q * self.slide_axes[tip] for tip in TIPS}

    def width(self, q) -> float:
        """Gap between the pad faces at finger joint value q (m); 0 once the pads meet."""
        return max(0.0, 2 * self._joint(q) + self.closed_gap_m)

    def width_between(self, q_left, q_right) -> float:
        """Gap between the pad faces when the two fingers are at different joint values."""
        return max(0.0, self._joint(q_left) + self._joint(q_right) + self.closed_gap_m)

    def joint_for_width(self, width) -> float:
        width = float(width)
        if not 0 <= width <= self.max_width_m:
            raise ValueError(f"width {width} m is outside [0, {self.max_width_m:.5f}]")
        return (width - self.closed_gap_m) / 2

    def _joint(self, q) -> float:
        q = float(q)
        if not 0 <= q <= self.travel_m:
            raise ValueError(f"finger joint {q} m is outside [0, {self.travel_m}]")
        return q


def load_linear_4310(asset_dir=ASSET_DIR, *, verify=True) -> Linear4310:
    asset_dir = Path(asset_dir)
    if verify:
        verify_assets(asset_dir)
    root = ElementTree.parse(asset_dir / MJCF_NAME).getroot()
    compiler = root.find("compiler")
    mesh_dir = asset_dir / ("" if compiler is None else compiler.get("meshdir", ""))
    meshes = {mesh.get("name"): mesh for mesh in root.find("asset").findall("mesh")}
    body = _child(root.find("worldbody"), "body", "gripper")
    mount_T_body = _pose(body)
    frames, mesh_poses, files, inertials, local_points = {"gripper": mount_T_body}, {}, {}, {}, {}
    mesh_poses["gripper"], files["gripper"] = _mesh_geom(body, meshes, mesh_dir)
    inertials["gripper"] = _inertial(body)
    slides, travel, joint_names = {}, set(), []
    for tip in TIPS:
        element = _child(body, "body", tip)
        frames[tip] = mount_T_body @ _pose(element)
        mesh_poses[tip], files[tip] = _mesh_geom(element, meshes, mesh_dir)
        inertials[tip] = _inertial(element)
        name, axis, high = _slide_joint(element)
        joint_names.append(name)
        slides[tip], travel = frames[tip][:3, :3] @ axis, travel | {high}
    _check_coupling(root, joint_names)
    # The CAD export rounds quaternions to 6 digits, so the axes are opposite to about 4e-6 rad.
    if len(travel) != 1 or not np.allclose(slides["tip_left"], -slides["tip_right"], rtol=0, atol=1e-4):
        raise ValueError("the fingers must slide the same distance in opposite directions")
    for link in LINKS:
        local_points[link] = _transform(frames[link] @ mesh_poses[link], read_binary_stl(files[link]))
    mount_T_tcp = mount_T_body @ _pose(_child(body, "site", "grasp_site"))
    approach = mount_T_tcp[:3, 2]
    opening = slides["tip_right"]
    pads = {}
    for tip in TIPS:
        along = local_points[tip] @ approach
        pads[tip] = local_points[tip][along >= along.max() - PAD_DEPTH_M] @ opening
    closed_gap = float(pads["tip_right"].min() - pads["tip_left"].max())

    def frozen(mapping):
        return MappingProxyType({key: _read_only(value) for key, value in mapping.items()})

    return Linear4310(frozen(frames), frozen(mesh_poses), MappingProxyType(files), _read_only(mount_T_tcp),
                      _read_only(local_points["gripper"]), frozen({tip: local_points[tip] for tip in TIPS}),
                      frozen(slides), travel.pop(), closed_gap, MappingProxyType(inertials))
