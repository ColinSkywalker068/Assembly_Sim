"""Camera authoring, bounded RGB capture, stability checks, and stage saving."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scene_config import SceneConfig


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
    agent_path: str
    left_wrist_path: str
    right_wrist_path: str


@dataclass(frozen=True)
class CaptureResult:
    path: Path
    shape: tuple[int, ...]
    attempts: int


@dataclass(frozen=True)
class StabilityReport:
    ok: bool
    frames: int
    max_linear_speed: float
    max_angular_speed: float
    failures: tuple[str, ...]


def camera_specs(config: SceneConfig) -> Mapping[str, CameraSpec]:
    result: dict[str, CameraSpec] = {}
    for name in ("agent", "left_wrist", "right_wrist"):
        value = config.data["cameras"][name]
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


def validated_scene_output(config: SceneConfig, key: str) -> Path:
    if key not in {
        "generated_usd",
        "agent_image",
        "left_wrist_image",
        "right_wrist_image",
    }:
        raise KeyError(f"unknown generated output: {key}")
    output = config.resolve_repo_path(key)
    scene_root = (config.repo_root / "06_interactive_scene").resolve()
    if not output.is_relative_to(scene_root):
        raise ValueError(f"generated output must remain under 06_interactive_scene: {output}")
    return output


def validate_wrist_parents(
    specs: Mapping[str, CameraSpec], robots: Mapping[str, Any]
) -> None:
    for robot_name in ("left", "right"):
        camera_name = f"{robot_name}_wrist"
        spec = specs[camera_name]
        prim_parent = spec.prim_path.rsplit("/", 1)[0]
        if prim_parent != spec.parent:
            raise ValueError(
                f"{camera_name} camera prim path {spec.prim_path} is not nested under "
                f"declared parent {spec.parent}"
            )
        if spec.parent != robots[robot_name].flange_path:
            raise ValueError(
                f"{camera_name} camera parent {spec.parent} does not match "
                f"{robot_name} robot flange {robots[robot_name].flange_path}"
            )


def _look_at_quaternion(eye, look_vector, up=(0.0, 0.0, 1.0)):
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
        s = math.sqrt(trace + 1.0) * 2.0
        quat = (
            0.25 * s,
            (matrix[2, 1] - matrix[1, 2]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = (
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            )
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = (
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            )
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = (
                (matrix[1, 0] - matrix[0, 1]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
            )
    return quat


def author_cameras(stage: Any, config: SceneConfig, robots: Mapping[str, Any]) -> CameraHandles:
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
        quaternion = _look_at_quaternion(spec.position, spec.look_vector)
        xform.AddOrientOp().Set(Gf.Quatf(quaternion[0], *quaternion[1:]))
    return CameraHandles(
        specs["agent"].prim_path,
        specs["left_wrist"].prim_path,
        specs["right_wrist"].prim_path,
    )


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

    output = Path(output)
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
                return CaptureResult(output.resolve(), tuple(rgb.shape), attempt)
    finally:
        annotator.detach(render_product)
        render_product.destroy()
    shape = None if observed is None else (str(observed.dtype), tuple(observed.shape))
    raise RuntimeError(
        f"camera {camera_path} returned no valid uint8 RGB frame after {max_retries} attempts; "
        f"last observation={shape}"
    )


def _rotation_angle(previous, current) -> float:
    relative = previous.GetInverse() * current
    return math.radians(float(relative.GetAngle()))


def validate_stability(handles: Any, seconds: float = 3.0) -> StabilityReport:
    import numpy as np
    from pxr import Usd, UsdGeom

    frames = max(1, int(math.ceil(seconds / handles.config.physics_dt)))
    tracked: dict[str, tuple[np.ndarray, Any]] = {}
    max_linear = 0.0
    max_angular = 0.0
    final_window = min(60, frames)
    for frame in range(frames):
        handles.world.step(render=False)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for fragment in handles.fragments:
            prim = handles.stage.GetPrimAtPath(fragment.root_path)
            matrix = cache.GetLocalToWorldTransform(prim)
            position = np.asarray(matrix.ExtractTranslation(), dtype=float)
            rotation = matrix.ExtractRotation()
            if not np.all(np.isfinite(position)):
                return StabilityReport(False, frame + 1, math.inf, math.inf, (f"non-finite pose: {fragment.name}",))
            if fragment.name in tracked and frame >= frames - final_window:
                old_position, old_rotation = tracked[fragment.name]
                max_linear = max(
                    max_linear,
                    float(np.linalg.norm(position - old_position) / handles.config.physics_dt),
                )
                max_angular = max(
                    max_angular,
                    _rotation_angle(old_rotation, rotation) / handles.config.physics_dt,
                )
            tracked[fragment.name] = (position, rotation)

    failures: list[str] = []
    environment = handles.config.data["environment"]
    table_position = environment["table_position"]
    table_size = environment["table_size"]
    table_top = float(table_position[2]) + float(table_size[2]) / 2.0
    table_half_x = float(table_size[0]) / 2.0 + 0.05
    table_half_y = float(table_size[1]) / 2.0 + 0.05
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for fragment in handles.fragments:
        prim = handles.stage.GetPrimAtPath(fragment.root_path)
        bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = bounds.GetMin()
        maximum = bounds.GetMax()
        if float(minimum[2]) < table_top - 0.012:
            failures.append(f"{fragment.name} penetrates table by {table_top - float(minimum[2]):.6f} m")
        if max(abs(float(minimum[0])), abs(float(maximum[0]))) > table_half_x:
            failures.append(f"{fragment.name} left table x bounds")
        if max(abs(float(minimum[1])), abs(float(maximum[1]))) > table_half_y:
            failures.append(f"{fragment.name} left table y bounds")
    if max_linear > 0.05:
        failures.append(f"final linear speed {max_linear:.6f} m/s exceeds 0.05")
    if max_angular > 0.35:
        failures.append(f"final angular speed {max_angular:.6f} rad/s exceeds 0.35")
    return StabilityReport(not failures, frames, max_linear, max_angular, tuple(failures))


def save_portable_stage(stage: Any, output: Path) -> None:
    """Export a USDA layer with repository-local references rewritten relative."""
    from pxr import Usd

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.usda")
    stage.GetRootLayer().Export(str(temporary), args={"format": "usda"})
    text = temporary.read_text(encoding="utf-8")
    repo_root = output.parents[2]

    def relative_asset(match: re.Match[str]) -> str:
        raw = match.group(1)
        candidate = Path(raw)
        if not candidate.is_absolute():
            return match.group(0)
        resolved = candidate.resolve()
        if not resolved.is_relative_to(repo_root):
            raise ValueError(f"stage contains non-portable external asset reference: {raw}")
        relative = os.path.relpath(resolved, output.parent).replace("\\", "/")
        return f"@{relative}@"

    text = re.sub(r"@([^@\r\n]+)@", relative_asset, text)
    if str(repo_root).lower() in text.lower() or "D:/i45" in text or "D:\\i45" in text:
        raise ValueError("portable stage still contains a machine-specific absolute path")
    text = text.rstrip() + "\n"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    reopened = Usd.Stage.Open(str(temporary))
    if reopened is None or not reopened.GetPrimAtPath("/World").IsValid():
        raise RuntimeError(f"saved stage could not be reopened: {temporary}")
    os.replace(temporary, output)
