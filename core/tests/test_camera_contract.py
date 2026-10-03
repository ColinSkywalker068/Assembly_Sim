import math

import pytest

from core.workcell.cameras import camera_specs, validate_wrist_parents
from core.workcell.config import load_workcell_preset


def test_dual_arm_camera_contract_preserves_agent_and_both_wrists():
    config = load_workcell_preset("dual_arm")
    specs = camera_specs(config)

    assert tuple(specs) == ("agent", "left_wrist", "right_wrist")
    assert specs["agent"].parent == "/World"
    assert specs["agent"].position == (0.0, -2.10, 3.44)
    assert specs["agent"].look_vector == pytest.approx((0.0, 2.10, -2.69))
    horizontal = math.hypot(*specs["agent"].look_vector[:2])
    depression = math.degrees(
        math.atan2(-specs["agent"].look_vector[2], horizontal)
    )
    assert 40.0 <= depression <= 65.0
    assert specs["left_wrist"].parent == "/World/Robots/Left/flange"
    assert specs["right_wrist"].parent == "/World/Robots/Right/flange"


@pytest.mark.parametrize(
    ("preset", "expected", "agent_position", "agent_look_vector"),
    (
        (
            "single_arm_left",
            ("agent", "left_wrist"),
            (1.040404, 0.0, 3.880667),
            (-1.564904, 0.0, -3.130667),
        ),
        (
            "single_arm_right",
            ("agent", "right_wrist"),
            (-1.040404, 0.0, 3.880667),
            (1.564904, 0.0, -3.130667),
        ),
    ),
)
def test_single_arm_agent_camera_faces_point_a_between_robot_and_nearest_tape(
    preset, expected, agent_position, agent_look_vector
):
    specs = camera_specs(load_workcell_preset(preset))

    assert tuple(specs) == expected
    assert specs["agent"].position == agent_position
    assert specs["agent"].look_vector == pytest.approx(agent_look_vector)
    assert math.sqrt(sum(component**2 for component in specs["agent"].look_vector)) == pytest.approx(3.5)


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
