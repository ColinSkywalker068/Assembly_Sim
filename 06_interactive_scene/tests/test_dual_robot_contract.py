import sys
from pathlib import Path
from types import SimpleNamespace


SCENE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCENE_ROOT / "scripts"))

from scene_config import SceneConfig
from scene_controls import SceneController, robot_prim_paths, robot_reset_targets


CONFIG_PATH = SCENE_ROOT / "config" / "scene.json"


class RecordingArm:
    def __init__(self, robot, config):
        self.name = robot.name
        self.calls = []

    def jog(self, joint_index, delta_rad):
        self.calls.append(("jog", joint_index, delta_rad))
        return self.name

    def home(self):
        self.calls.append(("home",))
        return self.name


class RecordingGripper:
    def __init__(self, robot, config):
        self.name = robot.name
        self.calls = []

    def command(self, fraction):
        self.calls.append(("command", fraction))
        return self.name


def controller():
    config = SceneConfig.load(CONFIG_PATH)
    robots = {name: SimpleNamespace(name=name) for name in config.robot_names}
    handles = SimpleNamespace(config=config, robots=robots)
    return SceneController(handles, arm_factory=RecordingArm, gripper_factory=RecordingGripper)


def test_tab_toggles_left_and_right():
    subject = controller()

    assert subject.active_robot_name == "left"
    assert subject.toggle_robot() == "right"
    assert subject.toggle_robot() == "left"


def test_joint_and_gripper_actions_route_to_active_robot():
    subject = controller()

    assert subject.jog(2, 0.1) == "left"
    assert subject.command_gripper(0.0) == "left"
    subject.toggle_robot()
    assert subject.jog(4, -0.1) == "right"
    assert subject.command_gripper(1.0) == "right"

    assert subject.arms["left"].calls == [("jog", 2, 0.1)]
    assert subject.grippers["left"].calls == [("command", 0.0)]
    assert subject.arms["right"].calls == [("jog", 4, -0.1)]
    assert subject.grippers["right"].calls == [("command", 1.0)]


def test_reset_targets_both_robot_specs():
    config = SceneConfig.load(CONFIG_PATH)
    mimic = {name: 1.0 for name in config.robot_spec("left")["gripper_dof_names"]}

    targets = robot_reset_targets(config, mimic)

    assert tuple(targets) == config.robot_names
    for name in config.robot_names:
        assert targets[name].arm == tuple(config.robot_spec(name)["home_joint_positions"])
        assert len(targets[name].gripper) == 8


def test_robot_prim_namespaces_are_disjoint():
    config = SceneConfig.load(CONFIG_PATH)
    left = robot_prim_paths(config, "left")
    right = robot_prim_paths(config, "right")

    assert left.root == "/World/Robots/Left"
    assert right.root == "/World/Robots/Right"
    assert left.gripper_root == "/World/Grippers/Left/Robotiq_2F_85"
    assert right.gripper_root == "/World/Grippers/Right/Robotiq_2F_85"
    assert not set(left).intersection(right)
