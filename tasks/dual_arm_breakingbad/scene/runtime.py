"""Keyboard bindings and interactive runtime for the assembly scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from core.workcell.cameras import capture_rgb

from .builder import build_assembly_scene


@dataclass(frozen=True)
class ControlAction:
    kind: str
    joint_index: int | None = None
    direction: int = 0


ACTION_KEYS = {
    "1", "2", "3", "4", "5", "6", "LEFT_BRACKET", "RIGHT_BRACKET",
    "H", "O", "K", "TAB", "R", "P", "I", "F7", "F8", "F9",
}


def _camera_path(cameras: Any, name: str) -> str:
    if hasattr(cameras, "path"):
        return cameras.path(name)
    return getattr(cameras, f"{name}_path")


def camera_action_paths(cameras: Any) -> dict[str, str]:
    return {
        "camera_agent": _camera_path(cameras, "agent"),
        "camera_left_wrist": _camera_path(cameras, "left_wrist"),
        "camera_right_wrist": _camera_path(cameras, "right_wrist"),
    }


def capture_requests(cameras: Any) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (name, _camera_path(cameras, name), f"{name}_image")
        for name in ("agent", "left_wrist", "right_wrist")
    )


def ready_message(active_robot_name: str) -> str:
    return (
        f"Interactive scene ready. Active robot: {active_robot_name}. "
        "Press Tab to switch arms; see the task README for controls."
    )


def _event_key(event: Any) -> str:
    value = getattr(event, "input", event)
    return str(getattr(value, "name", value)).upper()


def key_action(event: Any) -> Optional[ControlAction]:
    if not str(getattr(event, "type", "")).upper().endswith("KEY_PRESS"):
        return None
    key = _event_key(event)
    if key in {"1", "2", "3", "4", "5", "6"}:
        return ControlAction("select_joint", int(key) - 1)
    if key == "LEFT_BRACKET":
        return ControlAction("jog", direction=-1)
    if key == "RIGHT_BRACKET":
        return ControlAction("jog", direction=1)
    kinds = {
        "H": "home", "O": "gripper_open", "K": "gripper_close",
        "TAB": "toggle_robot", "R": "reset", "P": "save",
        "I": "capture_all", "F7": "camera_agent",
        "F8": "camera_left_wrist", "F9": "camera_right_wrist",
    }
    return ControlAction(kinds[key]) if key in kinds else None


class KeyboardController:
    def __init__(self, input_interface: Any, keyboard: Any, dispatch: Callable):
        self.input_interface = input_interface
        self.keyboard = keyboard
        self.dispatch = dispatch
        self.subscription = None

    def _on_event(self, event: Any) -> bool:
        action = key_action(event)
        if action is not None:
            self.dispatch(action)
        return True

    def subscribe(self) -> None:
        if self.subscription is None:
            self.subscription = self.input_interface.subscribe_to_keyboard_events(
                self.keyboard, self._on_event
            )

    def close(self) -> None:
        if self.subscription is not None:
            self.input_interface.unsubscribe_to_keyboard_events(
                self.keyboard, self.subscription
            )
            self.subscription = None


class SceneActionDispatcher:
    def __init__(self, handles, output_directory: Path | None = None):
        self.handles = handles
        self.controller = handles.controller
        self.selected_joints = {
            name: 0 for name in handles.workcell.config.robot_names
        }
        self.output_directory = Path(output_directory).resolve() if output_directory else None

    def __call__(self, action: ControlAction) -> None:
        active = self.controller.active_robot_name
        if action.kind == "select_joint":
            self.selected_joints[active] = int(action.joint_index)
            print(f"Selected {active} arm joint {int(action.joint_index) + 1}", flush=True)
            return
        if action.kind == "toggle_robot":
            print(f"Active robot: {self.controller.toggle_robot()}", flush=True)
            return
        if action.kind == "jog":
            step = self.handles.workcell.config.robot(active).joint_jog_radians
            result = self.controller.jog(self.selected_joints[active], action.direction * step)
            print(f"{active}: {result.message}", flush=True)
            return
        if action.kind == "home":
            print(f"{active}: {self.controller.home().message}", flush=True)
        elif action.kind == "gripper_open":
            print(f"{active}: {self.controller.command_gripper(1.0).message}", flush=True)
        elif action.kind == "gripper_close":
            print(f"{active}: {self.controller.command_gripper(0.0).message}", flush=True)
        elif action.kind == "reset":
            self.controller.reset()
            print("Scene reset", flush=True)
        elif action.kind == "save":
            if self.output_directory is None:
                print("No --output-directory supplied; stage was not saved", flush=True)
            else:
                output = self.output_directory / "assembly_scene.usda"
                output.parent.mkdir(parents=True, exist_ok=True)
                self.handles.stage.GetRootLayer().Export(str(output))
                print(f"Saved {output}", flush=True)
        elif action.kind == "capture_all":
            if self.output_directory is None:
                print("No --output-directory supplied; cameras were not captured", flush=True)
            else:
                for name, camera_path, _ in capture_requests(self.handles.cameras):
                    result = capture_rgb(
                        camera_path,
                        self.output_directory / f"{name}.png",
                        resolution=self.handles.workcell.config.camera_resolution,
                    )
                    print(f"Captured {result.path}", flush=True)
        elif action.kind in camera_action_paths(self.handles.cameras):
            from omni.kit.viewport.utility import get_active_viewport

            path = camera_action_paths(self.handles.cameras)[action.kind]
            get_active_viewport().set_active_camera(path)
            print(f"Viewport camera: {path}", flush=True)


def run_interactive(
    assembly: Path,
    stream: bool = False,
    smoke_frames: int = 0,
    output_directory: Path | None = None,
) -> int:
    handles = build_assembly_scene(assembly, headless=stream, stream=stream)
    keyboard_controller = None
    try:
        import carb.input
        import omni.appwindow

        dispatcher = SceneActionDispatcher(handles, output_directory)
        app_window = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard_controller = KeyboardController(
            input_interface, app_window.get_keyboard(), dispatcher
        )
        keyboard_controller.subscribe()
        dispatcher(ControlAction("camera_agent"))
        print(ready_message(dispatcher.controller.active_robot_name), flush=True)
        frames = 0
        while handles.app.is_running():
            handles.world.step(render=True)
            frames += 1
            if smoke_frames and frames >= smoke_frames:
                break
        return 0
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
        handles.app.close()
