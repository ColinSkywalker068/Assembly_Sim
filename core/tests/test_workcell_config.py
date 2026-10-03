from pathlib import Path

import pytest

from core.workcell.config import (
    available_presets,
    load_workcell_preset,
    validate_task_workcell_boundary,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_available_presets_are_the_supported_workcell_variants():
    assert available_presets() == (
        "dual_arm",
        "single_arm_left",
        "single_arm_right",
    )


def test_dual_arm_preset_preserves_canonical_robot_placements():
    config = load_workcell_preset("dual_arm")

    assert config.robot_names == ("left", "right")
    assert config.robot("left").base_position == (-0.78, 0.0, 0.75)
    assert config.robot("left").base_orientation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert config.robot("right").base_position == (0.78, 0.0, 0.75)
    assert config.robot("right").base_orientation_wxyz == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("single_name", "robot_name"),
    (("single_arm_left", "left"), ("single_arm_right", "right")),
)
def test_single_arm_presets_reuse_the_complete_dual_arm_robot(single_name, robot_name):
    dual = load_workcell_preset("dual_arm")
    single = load_workcell_preset(single_name)

    assert single.robot_names == (robot_name,)
    assert single.robot(robot_name) == dual.robot(robot_name)


def test_workcell_invariants_match_the_existing_scene():
    config = load_workcell_preset("dual_arm")

    assert config.preset_name == "dual_arm"
    assert config.physics_dt == pytest.approx(1.0 / 120.0)
    assert config.render_dt == pytest.approx(1.0 / 30.0)
    assert config.camera_resolution == (1920, 1440)
    assert config.environment["table_position"] == [0.0, 0.0, 0.725]
    assert config.environment["table_size"] == [2.1, 1.3, 0.05]
    assert config.environment["assembly_pad"] == {
        "center": [0.0, 0.1, 0.75],
        "size": [0.512, 0.416, 0.0096],
        "grid_step": 0.04,
        "lane_centers_x": [-0.42, 0.42],
    }
    assert config.physics == {
        "dt": pytest.approx(1.0 / 120.0),
        "gravity": [0.0, 0.0, -9.81],
        "fragment_density": 650.0,
        "static_friction": 0.8,
        "dynamic_friction": 0.65,
        "restitution": 0.02,
        "linear_damping": 0.15,
        "angular_damping": 0.25,
    }
    assert config.asset_path("arm_usd") == (
        REPO_ROOT / "core/assets/robots/fanuc_crx10ial/crx10ial.usd"
    )
    assert config.asset_path("gripper_usd") == (
        REPO_ROOT / "core/assets/robots/robotiq_2f85/Robotiq_2F_85_flattened.usd"
    )


def test_unknown_preset_lists_valid_choices():
    with pytest.raises(ValueError) as error:
        load_workcell_preset("mobile_arm")

    message = str(error.value)
    assert "mobile_arm" in message
    for preset in available_presets():
        assert preset in message


@pytest.mark.parametrize("forbidden", ("environment", "physics", "render", "robots", "cameras"))
def test_task_configuration_cannot_override_core_workcell(forbidden):
    with pytest.raises(ValueError, match=forbidden):
        validate_task_workcell_boundary({forbidden: {}})
