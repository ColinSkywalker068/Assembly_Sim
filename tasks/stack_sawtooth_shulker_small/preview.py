"""Preview-only scene guides and, later, dataset preview rendering."""

from __future__ import annotations

import math
import re
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.workcell.environment import author_visual_material, bind_visual, set_transform

from .config import SCALE, PLACEMENT_SQUARE_SIZE
from .conditions import DemoCondition
from .conditions import condition_to_task_config
from .dataset import (
    DatasetManifest,
    condition_to_dict,
    validate_manifest,
    write_manifest,
)


PREVIEW_ROOT = "/World/PreviewGuides"
IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)
CONTACT_TILE_SIZE = (640, 480)
CAPTION_HEIGHT = 72


@dataclass(frozen=True)
class PreviewGuideSpec:
    path: str
    shape: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class PreviewRunResult:
    run_directory: Path
    manifest_path: Path
    summary_path: Path
    agent_contact_sheet: Path
    wrist_contact_sheet: Path
    image_paths: tuple[Path, ...]


def preview_guide_specs(
    condition: DemoCondition, table_top_z: float
) -> tuple[PreviewGuideSpec, ...]:
    """Describe nonphysical guides for one preview without authoring USD."""

    target_x, target_y = condition.base_target_xy
    if not all(math.isfinite(value) for value in (target_x, target_y)):
        raise ValueError("preview target coordinates must be finite")
    if not all(-PLACEMENT_SQUARE_SIZE / 2 <= value <= PLACEMENT_SQUARE_SIZE / 2 for value in (target_x, target_y)):
        raise ValueError("preview target lies outside the imaginary square")
    z = float(table_top_z) + 0.006 * SCALE
    outline_color = (1.0, 0.78, 0.05)
    marker_size = tuple(v * SCALE for v in (0.022, 0.022, 0.008))
    specs = [
        PreviewGuideSpec(
            f"{PREVIEW_ROOT}/PlacementSquare/Top",
            "cube",
            (0.0, PLACEMENT_SQUARE_SIZE / 2, z),
            (PLACEMENT_SQUARE_SIZE, 0.006 * SCALE, 0.004 * SCALE),
            IDENTITY_WXYZ,
            outline_color,
        ),
        PreviewGuideSpec(
            f"{PREVIEW_ROOT}/PlacementSquare/Bottom",
            "cube",
            (0.0, -PLACEMENT_SQUARE_SIZE / 2, z),
            (PLACEMENT_SQUARE_SIZE, 0.006 * SCALE, 0.004 * SCALE),
            IDENTITY_WXYZ,
            outline_color,
        ),
        PreviewGuideSpec(
            f"{PREVIEW_ROOT}/PlacementSquare/Left",
            "cube",
            (-PLACEMENT_SQUARE_SIZE / 2, 0.0, z),
            (0.006 * SCALE, PLACEMENT_SQUARE_SIZE, 0.004 * SCALE),
            IDENTITY_WXYZ,
            outline_color,
        ),
        PreviewGuideSpec(
            f"{PREVIEW_ROOT}/PlacementSquare/Right",
            "cube",
            (PLACEMENT_SQUARE_SIZE / 2, 0.0, z),
            (0.006 * SCALE, PLACEMENT_SQUARE_SIZE, 0.004 * SCALE),
            IDENTITY_WXYZ,
            outline_color,
        ),
        PreviewGuideSpec(
            f"{PREVIEW_ROOT}/TargetCenter",
            "sphere",
            (0.0, 0.0, z + 0.003 * SCALE),
            marker_size,
            IDENTITY_WXYZ,
            (1.0, 1.0, 1.0),
        ),
        PreviewGuideSpec(
            f"{PREVIEW_ROOT}/SampledTarget",
            "sphere",
            (target_x, target_y, z + 0.003 * SCALE),
            marker_size,
            IDENTITY_WXYZ,
            (0.9, 0.1, 0.85) if condition.kind != "standard" else (0.0, 0.9, 1.0),
        ),
    ]
    length = math.hypot(target_x, target_y)
    if length > 0.0:
        yaw = math.atan2(target_y, target_x)
        specs.append(
            PreviewGuideSpec(
                f"{PREVIEW_ROOT}/CenterToSample",
                "cube",
                (target_x / 2, target_y / 2, z),
                (length, 0.004 * SCALE, 0.003 * SCALE),
                (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                (0.9, 0.1, 0.85),
            )
        )
    return tuple(specs)


def author_preview_guides(stage, specs: tuple[PreviewGuideSpec, ...]) -> tuple[str, ...]:
    """Replace the preview-only USD subtree with the supplied visual guides."""

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    if stage.GetPrimAtPath(PREVIEW_ROOT).IsValid():
        stage.RemovePrim(PREVIEW_ROOT)
    UsdGeom.Xform.Define(stage, PREVIEW_ROOT)
    UsdGeom.Xform.Define(stage, f"{PREVIEW_ROOT}/PlacementSquare")
    paths = []
    for index, spec in enumerate(specs):
        if not spec.path.startswith(f"{PREVIEW_ROOT}/"):
            raise ValueError(f"preview guide path is outside preview root: {spec.path}")
        if spec.shape == "cube":
            geometry = UsdGeom.Cube.Define(stage, spec.path)
            geometry.CreateSizeAttr(1.0)
        elif spec.shape == "sphere":
            geometry = UsdGeom.Sphere.Define(stage, spec.path)
            geometry.CreateRadiusAttr(0.5)
        else:
            raise ValueError(f"unsupported preview guide shape: {spec.shape}")
        set_transform(
            geometry,
            spec.position,
            spec.orientation_wxyz,
            scale=spec.size,
        )
        material_path = f"/World/Looks/PreviewGuides/Guide_{index:02d}"
        material = author_visual_material(stage, material_path, spec.color)
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"{material_path}/Shader"))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*spec.color)
        )
        shader.CreateInput("emissiveIntensity", Sdf.ValueTypeNames.Float).Set(1.5)
        bind_visual(geometry.GetPrim(), material)
        paths.append(spec.path)
    return tuple(paths)


def create_preview_run_directory(
    output_root: Path, dataset_seed: int, timestamp: str | None = None
) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}", stamp):
        raise ValueError("preview timestamp must use YYYYMMDDTHHMMSS")
    run = root / f"dataset_scene_preview_{int(dataset_seed)}_{stamp}"
    run.mkdir()
    return run.resolve()


def caption_text(condition: DemoCondition, dataset_seed: int) -> str:
    classification = (
        "standard" if condition.kind == "standard" else f"group {condition.group}"
    )
    return "\n".join(
        (
            f"{condition.demonstration_id} | {classification} | "
            f"{condition.assembly_order[0]} -> {condition.assembly_order[1]}",
            f"Blue {condition.blue.pose_label} | Green {condition.green.pose_label} | "
            f"high {condition.high_color}",
            f"dataset seed {dataset_seed} | demo seed {condition.demo_seed}",
        )
    )


def caption_frame(path: Path, condition: DemoCondition, dataset_seed: int) -> None:
    from PIL import Image, ImageDraw

    destination = Path(path).expanduser().resolve()
    try:
        with Image.open(destination) as source:
            frame = source.convert("RGB")
    except Exception as exc:
        raise ValueError(f"cannot read preview image: {destination}") from exc
    try:
        canvas = Image.new(
            "RGB", (frame.width, frame.height + CAPTION_HEIGHT), (20, 20, 24)
        )
        canvas.paste(frame, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.multiline_text(
            (8, frame.height + 5),
            caption_text(condition, dataset_seed),
            fill=(245, 245, 245),
            spacing=3,
        )
        canvas.save(destination)
    finally:
        frame.close()


def write_contact_sheet(
    paths: tuple[Path, ...], destination: Path, columns: int = 5
) -> Path:
    from PIL import Image, ImageOps

    if not paths:
        raise ValueError("contact sheet requires at least one image")
    if columns < 1:
        raise ValueError("contact sheet columns must be positive")
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new(
        "RGB",
        (CONTACT_TILE_SIZE[0] * columns, CONTACT_TILE_SIZE[1] * rows),
        (32, 32, 36),
    )
    try:
        for index, path in enumerate(paths):
            try:
                with Image.open(path) as source:
                    tile = ImageOps.contain(source.convert("RGB"), CONTACT_TILE_SIZE)
            except Exception as exc:
                raise ValueError(f"cannot read contact-sheet image: {path}") from exc
            try:
                x = (index % columns) * CONTACT_TILE_SIZE[0]
                y = (index // columns) * CONTACT_TILE_SIZE[1]
                offset = (
                    x + (CONTACT_TILE_SIZE[0] - tile.width) // 2,
                    y + (CONTACT_TILE_SIZE[1] - tile.height) // 2,
                )
                sheet.paste(tile, offset)
            finally:
                tile.close()
        sheet.save(output)
    finally:
        sheet.close()
    return output


def _build_preview_scene(config):
    from .builder import build_stack_sawtooth_scene

    return build_stack_sawtooth_scene(config=config, headless=True)


def _apply_preview_config(handles, config) -> None:
    from .builder import apply_stack_sawtooth_config

    apply_stack_sawtooth_config(handles, config)


def _capture_preview_rgb(camera_path: str, output: Path, resolution):
    from core.workcell.cameras import capture_rgb

    return capture_rgb(camera_path, output, resolution=resolution)


def _write_json(path: Path, value: dict) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _validate_raw_capture(path: Path, resolution: tuple[int, int]) -> None:
    from PIL import Image, ImageStat

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            try:
                if rgb.size != tuple(resolution):
                    raise ValueError(
                        f"capture has size {rgb.size}, expected {tuple(resolution)}: {path}"
                    )
                extrema = ImageStat.Stat(rgb).extrema
                if max(channel[1] for channel in extrema) <= 4:
                    raise ValueError(f"capture is blank or black: {path}")
            finally:
                rgb.close()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"capture is unreadable: {path}") from exc


def validate_preview_artifacts(
    run_directory: Path, manifest: DatasetManifest
) -> None:
    from PIL import Image, ImageStat

    run = Path(run_directory).expanduser().resolve()
    expected_images = []
    observed_size = None
    for condition in manifest.conditions:
        directory = run / "demonstrations" / condition.demonstration_id
        condition_path = directory / "condition.json"
        if not condition_path.is_file():
            raise ValueError(f"missing condition artifact: {condition_path}")
        try:
            saved = json.loads(condition_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unreadable condition artifact: {condition_path}") from exc
        expected = json.loads(json.dumps(condition_to_dict(condition), allow_nan=False))
        if saved != expected:
            raise ValueError(
                f"condition artifact is duplicate, misordered, or corrupt: {condition_path}"
            )
        for camera_name in ("agent_view", "wrist_view"):
            path = directory / f"{camera_name}.png"
            expected_images.append(path)
            if not path.is_file():
                raise ValueError(f"missing preview image: {path}")
            try:
                with Image.open(path) as image:
                    image.load()
                    rgb = image.convert("RGB")
                    try:
                        if rgb.width < 1 or rgb.height <= CAPTION_HEIGHT:
                            raise ValueError(f"preview image has invalid dimensions: {path}")
                        if observed_size is None:
                            observed_size = rgb.size
                        elif rgb.size != observed_size:
                            raise ValueError(
                                f"preview image dimensions do not match: {path}"
                            )
                        extrema = ImageStat.Stat(rgb).extrema
                        if max(channel[1] for channel in extrema) <= 4:
                            raise ValueError(f"preview image is blank or black: {path}")
                    finally:
                        rgb.close()
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"preview image is unreadable: {path}") from exc
    actual_images = tuple(sorted((run / "demonstrations").glob("demo_*/*_view.png")))
    if len(actual_images) != 100 or set(actual_images) != set(expected_images):
        raise ValueError("preview image set is incomplete or contains unexpected files")
    for name in ("agent_view.png", "wrist_view.png"):
        path = run / "contact_sheets" / name
        if not path.is_file():
            raise ValueError(f"missing contact sheet: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"contact sheet is unreadable: {path}") from exc


def render_dataset_preview(
    manifest: DatasetManifest,
    output_root: Path,
    timestamp: str | None = None,
) -> PreviewRunResult:
    validate_manifest(manifest)
    run = create_preview_run_directory(output_root, manifest.dataset_seed, timestamp)
    manifest_path = write_manifest(manifest, run / "manifest.json")
    summary_path = run / "summary.json"
    demonstrations = run / "demonstrations"
    contact_sheets = run / "contact_sheets"
    demonstrations.mkdir()
    contact_sheets.mkdir()
    handles = None
    active_demo = None
    active_camera = None
    image_paths = []
    agent_paths = []
    wrist_paths = []
    resolution = None
    try:
        first_config = condition_to_task_config(manifest.conditions[0])
        handles = _build_preview_scene(first_config)
        resolution = tuple(handles.workcell.config.camera_resolution)
        for condition in manifest.conditions:
            active_demo = condition.demonstration_id
            config = condition_to_task_config(condition)
            _apply_preview_config(handles, config)
            author_preview_guides(
                handles.stage, preview_guide_specs(condition, config.table_top_z)
            )
            handles.world.step(render=True)
            handles.world.step(render=True)
            directory = demonstrations / condition.demonstration_id
            directory.mkdir()
            _write_json(directory / "condition.json", condition_to_dict(condition))
            for camera_key, filename, collection in (
                ("agent", "agent_view.png", agent_paths),
                ("right_wrist", "wrist_view.png", wrist_paths),
            ):
                active_camera = camera_key
                path = directory / filename
                _capture_preview_rgb(
                    handles.cameras.path(camera_key), path, resolution=resolution
                )
                _validate_raw_capture(path, resolution)
                caption_frame(path, condition, manifest.dataset_seed)
                collection.append(path)
                image_paths.append(path)
        active_demo = None
        active_camera = None
        agent_sheet = write_contact_sheet(
            tuple(agent_paths), contact_sheets / "agent_view.png", columns=5
        )
        wrist_sheet = write_contact_sheet(
            tuple(wrist_paths), contact_sheets / "wrist_view.png", columns=5
        )
        validate_preview_artifacts(run, manifest)
        _write_json(
            summary_path,
            {
                "status": "complete",
                "dataset_seed": manifest.dataset_seed,
                "condition_count": len(manifest.conditions),
                "image_count": len(image_paths),
                "camera_resolution": list(resolution),
                "caption_height": CAPTION_HEIGHT,
                "contact_sheet_count": 2,
            },
        )
        result = PreviewRunResult(
            run,
            manifest_path,
            summary_path.resolve(),
            agent_sheet,
            wrist_sheet,
            tuple(image_paths),
        )
    except BaseException as exc:
        _write_json(
            summary_path,
            {
                "status": "failed",
                "dataset_seed": manifest.dataset_seed,
                "image_count": len(image_paths),
                "failure": {
                    "demo_id": active_demo,
                    "camera": active_camera,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            },
        )
        raise
    finally:
        if handles is not None:
            handles.app.close()
    return result
