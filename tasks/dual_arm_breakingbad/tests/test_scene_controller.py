from types import SimpleNamespace

from tasks.dual_arm_breakingbad.scene.controller import (
    AssemblySceneController,
    fragment_reset_poses,
)


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
    delegated = SimpleNamespace(
        active_robot_name="left",
        toggle_robot=lambda: "right",
        jog=lambda joint, delta: (joint, delta),
        home=lambda: "home",
        command_gripper=lambda fraction: fraction,
        reset=lambda: None,
    )
    workcell = SimpleNamespace(controller=delegated)
    handles = SimpleNamespace(workcell=workcell, fragments=(), layout=None)
    return AssemblySceneController(handles)


def test_tab_toggles_left_and_right():
    subject = controller()

    assert subject.active_robot_name == "left"
    assert subject.toggle_robot() == "right"


def test_joint_and_gripper_actions_route_to_active_robot():
    subject = controller()

    assert subject.jog(2, 0.1) == (2, 0.1)
    assert subject.home() == "home"
    assert subject.command_gripper(0.0) == 0.0


def test_reset_targets_every_loaded_fragment():
    poses = {
        "alpha": {"position": [0.1, 0.2, 0.75], "orientation_wxyz": [1, 0, 0, 0]},
        "beta": {"position": [-0.1, 0.3, 0.75], "orientation_wxyz": [1, 0, 0, 0]},
        "gamma": {"position": [0.4, -0.2, 0.75], "orientation_wxyz": [1, 0, 0, 0]},
    }
    layout = SimpleNamespace(initial_pose=lambda name: poses[name])

    positions, orientations = fragment_reset_poses(layout, ("alpha", "beta", "gamma"))

    assert positions == ([0.1, 0.2, 0.75], [-0.1, 0.3, 0.75], [0.4, -0.2, 0.75])
    assert orientations == ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0])


def test_reset_uses_staging_not_goal_pose():
    staging = {"position": [0.4, 0.3, 0.75], "orientation_wxyz": [1, 0, 0, 0]}
    goal = {"position": [0.0, 0.0, 0.0], "orientation_wxyz": [1, 0, 0, 0]}
    layout = SimpleNamespace(
        initial_pose=lambda name: staging,
        goal_pose=lambda name: goal,
    )

    positions, _ = fragment_reset_poses(layout, ("fragment",))

    assert positions == ([0.4, 0.3, 0.75],)
    assert positions != (goal["position"],)
