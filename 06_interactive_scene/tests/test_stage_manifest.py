import sys
from pathlib import Path

import pytest


SCENE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SCENE_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scene_assets import expected_stage_manifest
from scene_config import SceneConfig


def test_manifest_requires_two_robot_roots_and_three_cameras():
    config = SceneConfig.load(SCENE_ROOT / "config" / "scene.json")

    manifest = expected_stage_manifest(config)

    assert manifest.floor_path == "/World/Environment/Floor"
    assert manifest.table_path == "/World/Environment/Table"
    assert manifest.plate_path == "/World/Environment/Plate"
    assert manifest.robot_paths == (
        "/World/Robots/Left",
        "/World/Robots/Right",
    )
    assert manifest.fragment_paths == tuple(
        f"/World/Fragments/piece_{index}" for index in range(8)
    )
    assert len(set(manifest.fragment_paths)) == 8
    assert manifest.camera_paths == (
        "/World/Cameras/Agent",
        "/World/Robots/Left/flange/WristCamera",
        "/World/Robots/Right/flange/WristCamera",
    )


def test_manifest_rejects_duplicate_robot_roots():
    config = SceneConfig.load(SCENE_ROOT / "config" / "scene.json")
    config.data["robots"]["right"]["prim_path"] = config.data["robots"]["left"]["prim_path"]

    with pytest.raises(ValueError, match="robot root paths must be unique"):
        expected_stage_manifest(config)


def test_manifest_rejects_duplicate_camera_paths():
    config = SceneConfig.load(SCENE_ROOT / "config" / "scene.json")
    config.data["cameras"]["right_wrist"]["prim_path"] = config.data["cameras"]["left_wrist"]["prim_path"]

    with pytest.raises(ValueError, match="camera paths must be unique"):
        expected_stage_manifest(config)
