from dataclasses import dataclass
from pathlib import Path

from tasks.dual_arm_breakingbad.__main__ import build_parser
from tasks.dual_arm_breakingbad.scene.runtime import (
    ACTION_KEYS,
    KeyboardController,
    camera_action_paths,
    capture_requests,
    key_action,
    ready_message,
)


@dataclass
class FakeInput:
    name: str


@dataclass
class FakeEvent:
    type: str
    input: FakeInput


def event(key, event_type="KEY_PRESS"):
    return FakeEvent(event_type, FakeInput(key))


def test_six_joint_selection_keys():
    for index in range(6):
        action = key_action(event(str(index + 1)))
        assert action.kind == "select_joint"
        assert action.joint_index == index


def test_jog_and_operator_actions():
    assert key_action(event("LEFT_BRACKET")).direction == -1
    assert key_action(event("RIGHT_BRACKET")).direction == 1
    expected = {
        "H": "home",
        "O": "gripper_open",
        "K": "gripper_close",
        "TAB": "toggle_robot",
        "R": "reset",
        "P": "save",
        "I": "capture_all",
        "F7": "camera_agent",
        "F8": "camera_left_wrist",
        "F9": "camera_right_wrist",
    }
    for key, kind in expected.items():
        assert key_action(event(key)).kind == kind


def test_release_repeat_and_unknown_events_are_ignored():
    assert key_action(event("H", "KEY_RELEASE")) is None
    assert key_action(event("H", "KEY_REPEAT")) is None
    assert key_action(event("UNKNOWN")) is None


def test_bindings_do_not_collide_with_documented_viewport_navigation():
    viewport_navigation = {"W", "A", "S", "D", "Q", "E"}
    assert viewport_navigation.isdisjoint(ACTION_KEYS)


class FakeInputInterface:
    def __init__(self):
        self.unsubscribe_calls = []

    def subscribe_to_keyboard_events(self, keyboard, callback):
        self.callback = callback
        return 42

    def unsubscribe_to_keyboard_events(self, keyboard, subscription):
        self.unsubscribe_calls.append((keyboard, subscription))


def test_keyboard_controller_subscribes_and_closes_once():
    interface = FakeInputInterface()
    received = []
    keyboard = object()
    controller = KeyboardController(interface, keyboard, received.append)
    controller.subscribe()
    assert interface.callback(event("H")) is True
    assert received[0].kind == "home"
    controller.close()
    controller.close()
    assert interface.unsubscribe_calls == [(keyboard, 42)]


def test_f7_f8_f9_select_distinct_views():
    cameras = type(
        "Cameras",
        (),
        {
            "agent_path": "/Agent",
            "left_wrist_path": "/LeftWrist",
            "right_wrist_path": "/RightWrist",
        },
    )()

    assert camera_action_paths(cameras) == {
        "camera_agent": "/Agent",
        "camera_left_wrist": "/LeftWrist",
        "camera_right_wrist": "/RightWrist",
    }


def test_capture_mapping_has_three_unambiguous_outputs():
    cameras = type(
        "Cameras",
        (),
        {
            "agent_path": "/Agent",
            "left_wrist_path": "/LeftWrist",
            "right_wrist_path": "/RightWrist",
        },
    )()

    assert capture_requests(cameras) == (
        ("agent", "/Agent", "agent_image"),
        ("left_wrist", "/LeftWrist", "left_wrist_image"),
        ("right_wrist", "/RightWrist", "right_wrist_image"),
    )


def test_ready_message_names_initial_active_robot():
    assert "Active robot: left" in ready_message("left")


def test_run_options_accept_assembly_override():
    arguments = build_parser().parse_args(
        ["launch", "--assembly", "generated/layout.json"]
    )

    assert arguments.assembly == Path("generated/layout.json")


def test_run_options_accept_webrtc_streaming():
    arguments = build_parser().parse_args(
        ["launch", "--assembly", "generated/layout.json", "--stream"]
    )

    assert arguments.stream is True
