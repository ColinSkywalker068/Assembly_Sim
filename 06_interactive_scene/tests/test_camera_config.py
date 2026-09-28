import inspect
import sys
from pathlib import Path

import pytest


SCENE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCENE_ROOT / "scripts"))

from scene_cameras import camera_specs, capture_rgb, validate_wrist_parents, validated_scene_output
from scene_config import SceneConfig


@pytest.fixture
def config():
    return SceneConfig.load(SCENE_ROOT / "config" / "scene.json")


def test_camera_contract_has_agent_and_two_wrists(config):
    specs = camera_specs(config)
    assert tuple(specs) == ("agent", "left_wrist", "right_wrist")
    assert config.camera_resolution == (640, 480)
    assert specs["agent"].parent == "/World"
    assert specs["left_wrist"].parent == "/World/Robots/Left/flange"
    assert specs["right_wrist"].parent == "/World/Robots/Right/flange"
    assert specs["agent"].position == (1.35, -2.05, 1.72)
    assert specs["agent"].look_vector == pytest.approx((-1.35, 2.10, -0.94))
    assert specs["agent"].focal_length == 25.0


def test_camera_clipping_and_look_vectors_are_valid(config):
    for spec in camera_specs(config).values():
        assert 0.0 < spec.clipping_range[0] < spec.clipping_range[1]
        assert sum(component * component for component in spec.look_vector) > 0.0


def test_capture_retry_default_is_bounded():
    default = inspect.signature(capture_rgb).parameters["max_retries"].default
    assert default == 12


@pytest.mark.parametrize(
    "key", ["generated_usd", "agent_image", "left_wrist_image", "right_wrist_image"]
)
def test_generated_outputs_are_confined_to_scene_directory(config, key):
    output = validated_scene_output(config, key)
    assert output.is_relative_to(SCENE_ROOT.resolve())


def test_output_path_escape_is_rejected(config):
    config.data["paths"]["generated_usd"] = "../escaped.usd"
    with pytest.raises(ValueError, match="06_interactive_scene"):
        validated_scene_output(config, "generated_usd")


def test_three_camera_output_paths_are_portable(config):
    outputs = {
        key: validated_scene_output(config, key).name
        for key in ("agent_image", "left_wrist_image", "right_wrist_image")
    }
    assert outputs == {
        "agent_image": "agent_camera.png",
        "left_wrist_image": "left_wrist_camera.png",
        "right_wrist_image": "right_wrist_camera.png",
    }


def test_each_wrist_parent_matches_named_flange(config):
    specs = camera_specs(config)
    robots = {
        name: type("Robot", (), {"flange_path": f"/World/Robots/{name.title()}/flange"})()
        for name in config.robot_names
    }

    assert validate_wrist_parents(specs, robots) is None

    robots["right"].flange_path = "/World/Wrong/flange"
    with pytest.raises(ValueError, match="right_wrist"):
        validate_wrist_parents(specs, robots)
