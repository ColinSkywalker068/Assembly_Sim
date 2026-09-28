import json
import sys
from pathlib import Path

import pytest


SCENE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SCENE_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scene_config import SceneConfig


CONFIG_PATH = SCENE_ROOT / "config" / "scene.json"


def test_loads_required_scene_contract():
    config = SceneConfig.load(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.physics_dt == pytest.approx(1 / 120)
    assert config.render_dt == pytest.approx(1 / 30)
    assert config.camera_resolution == (640, 480)
    assert config.fragment_names == tuple(f"piece_{index}" for index in range(8))


def test_dual_robot_contract_is_left_then_right():
    config = SceneConfig.load(CONFIG_PATH)

    assert config.robot_names == ("left", "right")
    assert config.robot_spec("left")["prim_path"] == "/World/Robots/Left"
    assert config.robot_spec("right")["prim_path"] == "/World/Robots/Right"


def test_demo_arm_and_fragment_transforms_are_exact():
    config = SceneConfig.load(CONFIG_PATH)

    assert config.robot_spec("left")["base_position"] == [-0.78, 0.0, 0.75]
    assert config.robot_spec("left")["base_orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert config.robot_spec("right")["base_position"] == [0.78, 0.0, 0.75]
    assert config.robot_spec("right")["base_orientation_wxyz"] == [0.0, 0.0, 0.0, 1.0]
    expected_positions = {
        "piece_0": [-0.42, -0.234, 0.75],
        "piece_1": [0.42, 0.236, 0.734],
        "piece_2": [-0.42, 0.242, 0.75],
        "piece_3": [-0.42, -0.062, 0.654],
        "piece_4": [0.42, -0.188, 0.75],
        "piece_5": [-0.42, 0.086, 0.75],
        "piece_6": [0.42, 0.048, 0.622],
        "piece_7": [-0.22, 0.46, 0.75],
    }
    assert {
        name: pose["position"]
        for name, pose in config.data["fragments"]["initial_poses"].items()
    } == expected_positions
    assert all(
        pose["orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
        for pose in config.data["fragments"]["initial_poses"].values()
    )


@pytest.mark.parametrize("robot_names", [("left",), ("left", "right", "spare")])
def test_rejects_missing_or_extra_robot_names(tmp_path, robot_names):
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    shared = dict(source["robots"]["left"])
    source["robots"] = {name: dict(shared) for name in robot_names}
    config_path = tmp_path / "repository" / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (config_path.parents[2] / "02_robot_assets").mkdir()
    (config_path.parents[2] / "README.md").write_text("repository", encoding="utf-8")
    config_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly left and right"):
        SceneConfig.load(config_path)


def test_paths_resolve_after_repo_relocation(tmp_path):
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    relocated = tmp_path / "repository with spaces"
    config_path = relocated / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (relocated / "02_robot_assets").mkdir()
    (relocated / "README.md").write_text("relocated repository", encoding="utf-8")
    config_path.write_text(json.dumps(source), encoding="utf-8")

    config = SceneConfig.load(config_path)

    expected = relocated / source["paths"]["arm_usd"]
    assert config.repo_root == relocated.resolve()
    assert config.resolve_repo_path("arm_usd") == expected.resolve()


def test_missing_asset_reports_field_and_resolved_path(tmp_path):
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    relocated = tmp_path / "repo"
    config_path = relocated / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (relocated / "02_robot_assets").mkdir()
    (relocated / "README.md").write_text("repository", encoding="utf-8")
    source["paths"]["arm_usd"] = "missing/arm.usd"
    config_path.write_text(json.dumps(source), encoding="utf-8")
    config = SceneConfig.load(config_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        config.validate_inputs()

    expected = str((relocated / "missing" / "arm.usd").resolve())
    assert "arm_usd" in str(exc_info.value)
    assert expected in str(exc_info.value)


def test_rejects_non_eight_fragment_contract(tmp_path):
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source["fragments"]["names"] = source["fragments"]["names"][:-1]
    config_path = tmp_path / "repository" / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (config_path.parents[2] / "02_robot_assets").mkdir()
    (config_path.parents[2] / "README.md").write_text("repository", encoding="utf-8")
    config_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly piece_0 through piece_7"):
        SceneConfig.load(config_path)
