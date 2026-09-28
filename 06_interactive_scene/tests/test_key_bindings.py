import sys
from dataclasses import dataclass
from pathlib import Path


SCENE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCENE_ROOT / "scripts"))

from run_scene import ACTION_KEYS, KeyboardController, key_action


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
        "I": "capture_both",
        "F7": "camera_agent",
        "F8": "camera_wrist",
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
