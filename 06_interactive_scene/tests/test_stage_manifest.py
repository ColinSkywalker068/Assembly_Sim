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


def test_generic_manifest_has_three_fragment_paths_and_no_plate():
    class GenericConfig:
        fragment_names = ("alpha", "beta", "gamma")
        robot_names = ("left", "right")
        has_support_surface = False
        data = {
            "cameras": {
                "agent": {"prim_path": "/World/Cameras/Agent"},
                "left_wrist": {"prim_path": "/World/Left/WristCamera"},
                "right_wrist": {"prim_path": "/World/Right/WristCamera"},
            }
        }

        def robot_spec(self, name):
            return {"prim_path": f"/World/Robots/{name.title()}"}

    manifest = expected_stage_manifest(GenericConfig())

    assert manifest.fragment_paths == (
        "/World/Fragments/alpha",
        "/World/Fragments/beta",
        "/World/Fragments/gamma",
    )
    assert manifest.plate_path is None


def test_legacy_manifest_keeps_plate():
    config = SceneConfig.load(SCENE_ROOT / "config" / "scene.json")

    assert expected_stage_manifest(config).plate_path == "/World/Environment/Plate"


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
