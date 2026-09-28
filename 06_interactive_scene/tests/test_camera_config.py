import inspect
import sys
from pathlib import Path

import pytest


SCENE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCENE_ROOT / "scripts"))

from scene_cameras import camera_specs, capture_rgb, validated_scene_output
from scene_config import SceneConfig


@pytest.fixture
def config():
    return SceneConfig.load(SCENE_ROOT / "config" / "scene.json")


def test_camera_contract_has_exactly_agent_and_wrist(config):
    specs = camera_specs(config)
    assert tuple(specs) == ("agent", "wrist")
    assert config.camera_resolution == (640, 480)
    assert specs["agent"].parent == "/World"
    assert specs["wrist"].parent.startswith("/World/Robot/flange")


def test_camera_clipping_and_look_vectors_are_valid(config):
    for spec in camera_specs(config).values():
        assert 0.0 < spec.clipping_range[0] < spec.clipping_range[1]
        assert sum(component * component for component in spec.look_vector) > 0.0


def test_capture_retry_default_is_bounded():
    default = inspect.signature(capture_rgb).parameters["max_retries"].default
    assert default == 12


@pytest.mark.parametrize("key", ["generated_usd", "agent_image", "wrist_image"])
def test_generated_outputs_are_confined_to_scene_directory(config, key):
    output = validated_scene_output(config, key)
    assert output.is_relative_to(SCENE_ROOT.resolve())


def test_output_path_escape_is_rejected(config):
    config.data["paths"]["generated_usd"] = "../escaped.usd"
    with pytest.raises(ValueError, match="06_interactive_scene"):
        validated_scene_output(config, "generated_usd")
