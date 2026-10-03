import pytest

from core.scene.builder import simulation_launch_config
from core.scene.validation import expected_workcell_manifest
from core.workcell.config import load_workcell_preset


def test_dual_arm_manifest_has_shared_environment_robots_and_cameras_only():
    config = load_workcell_preset("dual_arm")
    manifest = expected_workcell_manifest(config)

    assert manifest.floor_path == "/World/Environment/Floor"
    assert manifest.table_path == "/World/Environment/Table"
    assert manifest.assembly_pad_path == "/World/Environment/AssemblyPad"
    assert manifest.robot_paths == (
        "/World/Robots/Left",
        "/World/Robots/Right",
    )
    assert manifest.camera_paths == (
        "/World/Cameras/Agent",
        "/World/Robots/Left/flange/WristCamera",
        "/World/Robots/Right/flange/WristCamera",
    )
    assert not hasattr(manifest, "fragment_paths")


@pytest.mark.parametrize(
    ("preset", "robot_path", "wrist_path"),
    (
        (
            "single_arm_left",
            "/World/Robots/Left",
            "/World/Robots/Left/flange/WristCamera",
        ),
        (
            "single_arm_right",
            "/World/Robots/Right",
            "/World/Robots/Right/flange/WristCamera",
        ),
    ),
)
def test_single_arm_manifest_has_one_robot_agent_and_matching_wrist(
    preset, robot_path, wrist_path
):
    manifest = expected_workcell_manifest(load_workcell_preset(preset))

    assert manifest.robot_paths == (robot_path,)
    assert manifest.camera_paths == ("/World/Cameras/Agent", wrist_path)


def test_streaming_launch_uses_one_gpu_and_visible_ui():
    launch = simulation_launch_config(
        load_workcell_preset("dual_arm"), headless=True, stream=True
    )

    assert launch["headless"] is True
    assert launch["hide_ui"] is False
    assert launch["multi_gpu"] is False
    assert launch["max_gpu_count"] == 1
