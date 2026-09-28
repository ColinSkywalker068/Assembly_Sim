import sys
from pathlib import Path


SCENE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SCENE_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scene_assets import expected_stage_manifest
from scene_config import SceneConfig


def test_expected_stage_manifest_pins_scene_prim_contract():
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
