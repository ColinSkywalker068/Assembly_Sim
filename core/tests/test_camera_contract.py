import pytest

from core.workcell.cameras import camera_specs, validate_wrist_parents
from core.workcell.config import load_workcell_preset


def test_dual_arm_camera_contract_preserves_agent_and_both_wrists():
    config = load_workcell_preset("dual_arm")
    specs = camera_specs(config)

    assert tuple(specs) == ("agent", "left_wrist", "right_wrist")
    assert specs["agent"].parent == "/World"
    assert specs["agent"].position == (1.35, -2.05, 1.72)
    assert specs["agent"].look_vector == pytest.approx((-1.35, 2.10, -0.94))
    assert specs["left_wrist"].parent == "/World/Robots/Left/flange"
    assert specs["right_wrist"].parent == "/World/Robots/Right/flange"


@pytest.mark.parametrize(
    ("preset", "expected"),
    (
        ("single_arm_left", ("agent", "left_wrist")),
        ("single_arm_right", ("agent", "right_wrist")),
    ),
)
def test_single_arm_cameras_include_agent_and_selected_wrist_only(preset, expected):
    assert tuple(camera_specs(load_workcell_preset(preset))) == expected


def test_wrist_parent_validation_uses_selected_robot_flange():
    config = load_workcell_preset("single_arm_left")
    specs = camera_specs(config)
    robots = {
        "left": type("Robot", (), {"flange_path": "/World/Robots/Left/flange"})()
    }

    assert validate_wrist_parents(specs, robots) is None
    robots["left"].flange_path = "/World/Wrong/flange"
    with pytest.raises(ValueError, match="left_wrist"):
        validate_wrist_parents(specs, robots)
