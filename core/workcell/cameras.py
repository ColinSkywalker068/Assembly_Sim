"""Shared agent and robot-mounted camera contracts and authoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.workcell.config import WorkcellConfig


@dataclass(frozen=True)
class CameraSpec:
    name: str
    prim_path: str
    parent: str
    position: tuple[float, float, float]
    look_vector: tuple[float, float, float]
    focal_length: float
    clipping_range: tuple[float, float]


@dataclass(frozen=True)
class CameraHandles:
    paths: Mapping[str, str]

    def path(self, name: str) -> str:
        return self.paths[name]

    @property
    def agent_path(self) -> str:
        return self.paths["agent"]


@dataclass(frozen=True)
class CaptureResult:
    path: Path
    shape: tuple[int, ...]
    attempts: int


def camera_specs(config: WorkcellConfig) -> Mapping[str, CameraSpec]:
    enabled = ("agent", *(f"{name}_wrist" for name in config.robot_names))
    result: dict[str, CameraSpec] = {}
    for name in enabled:
        value = config.cameras[name]
        position = tuple(float(component) for component in value["position"])
        target_key = "look_at" if name == "agent" else "look_at_local"
        target = tuple(float(component) for component in value[target_key])
        result[name] = CameraSpec(
            name=name,
            prim_path=str(value["prim_path"]),
            parent=str(value["parent"]),
            position=position,
            look_vector=tuple(end - start for start, end in zip(position, target)),
            focal_length=float(value["focal_length"]),
            clipping_range=tuple(float(component) for component in value["clipping_range"]),
        )
    return result


def validate_wrist_parents(specs: Mapping[str, CameraSpec], robots: Mapping[str, Any]) -> None:
    for robot_name in robots:
        camera_name = f"{robot_name}_wrist"
        spec = specs[camera_name]
        if spec.prim_path.rsplit("/", 1)[0] != spec.parent:
            raise ValueError(
                f"{camera_name} camera prim path {spec.prim_path} is not nested under "
                f"declared parent {spec.parent}"
            )
        if spec.parent != robots[robot_name].flange_path:
            raise ValueError(
                f"{camera_name} camera parent {spec.parent} does not match "
                f"{robot_name} robot flange {robots[robot_name].flange_path}"
            )


def _look_at_quaternion(look_vector, up=(0.0, 0.0, 1.0)):
    import numpy as np

    z_axis = -np.asarray(look_vector, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.asarray(up, dtype=float), z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.cross(np.asarray((0.0, 1.0, 0.0)), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.stack((x_axis, y_axis, z_axis), axis=1)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        )
    if index == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return (
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        )
    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return (
        (matrix[1, 0] - matrix[0, 1]) / scale,
        (matrix[0, 2] + matrix[2, 0]) / scale,
        (matrix[1, 2] + matrix[2, 1]) / scale,
        0.25 * scale,
    )


def author_cameras(stage: Any, config: WorkcellConfig, robots: Mapping[str, Any]) -> CameraHandles:
    from pxr import Gf, UsdGeom

    specs = camera_specs(config)
    validate_wrist_parents(specs, robots)
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    for spec in specs.values():
        if not stage.GetPrimAtPath(spec.parent).IsValid():
            raise RuntimeError(f"camera parent prim does not exist: {spec.parent}")
        camera = UsdGeom.Camera.Define(stage, spec.prim_path)
        camera.CreateFocalLengthAttr(spec.focal_length)
        camera.CreateClippingRangeAttr(Gf.Vec2f(*spec.clipping_range))
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*spec.position))
        quaternion = _look_at_quaternion(spec.look_vector)
        xform.AddOrientOp().Set(Gf.Quatf(quaternion[0], *quaternion[1:]))
    return CameraHandles({name: spec.prim_path for name, spec in specs.items()})


def capture_rgb(
    camera_path: str,
    output: Path,
    max_retries: int = 12,
    resolution: tuple[int, int] = (640, 480),
) -> CaptureResult:
    if max_retries < 1 or max_retries > 120:
        raise ValueError(f"max_retries must be bounded to 1..120, got {max_retries}")
    import numpy as np
    import omni.replicator.core as rep
    from PIL import Image

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    render_product = rep.create.render_product(camera_path, resolution)
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    observed = None
    try:
        for attempt in range(1, max_retries + 1):
            rep.orchestrator.step(rt_subframes=2)
            observed = np.asarray(annotator.get_data())
            if (
                observed.dtype == np.uint8
                and observed.ndim == 3
                and observed.shape[:2] == (resolution[1], resolution[0])
                and observed.shape[2] >= 3
                and int(observed[..., :3].max()) > 4
            ):
                rgb = observed[..., :3]
                Image.fromarray(rgb).save(output)
                return CaptureResult(output, tuple(rgb.shape), attempt)
    finally:
        annotator.detach(render_product)
        render_product.destroy()
    shape = None if observed is None else (str(observed.dtype), tuple(observed.shape))
    raise RuntimeError(
        f"camera {camera_path} returned no valid RGB frame after {max_retries} attempts; "
        f"last observation={shape}"
    )
