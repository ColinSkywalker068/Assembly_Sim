"""Launch the interactive Isaac Sim GUI scene and keyboard controls."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


@dataclass(frozen=True)
class ControlAction:
    kind: str
    joint_index: int | None = None
    direction: int = 0


ACTION_KEYS = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "LEFT_BRACKET",
    "RIGHT_BRACKET",
    "H",
    "O",
    "K",
    "R",
    "P",
    "I",
    "F7",
    "F8",
}


def _event_key(event: Any) -> str:
    value = getattr(event, "input", event)
    return str(getattr(value, "name", value)).upper()


def key_action(event: Any) -> Optional[ControlAction]:
    event_type = str(getattr(event, "type", "")).upper()
    if not event_type.endswith("KEY_PRESS"):
        return None
    key = _event_key(event)
    if key in {"1", "2", "3", "4", "5", "6"}:
        return ControlAction("select_joint", int(key) - 1)
    if key == "LEFT_BRACKET":
        return ControlAction("jog", direction=-1)
    if key == "RIGHT_BRACKET":
        return ControlAction("jog", direction=1)
    kinds = {
        "H": "home",
        "O": "gripper_open",
        "K": "gripper_close",
        "R": "reset",
        "P": "save",
        "I": "capture_both",
        "F7": "camera_agent",
        "F8": "camera_wrist",
    }
    return ControlAction(kinds[key]) if key in kinds else None


class KeyboardController:
    def __init__(self, input_interface: Any, keyboard: Any, dispatch: Callable[[ControlAction], None]):
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
    def __init__(self, handles: Any):
        from scene_controls import SceneController

        self.handles = handles
        self.controller = SceneController(handles)
        self.selected_joint = 0

    def __call__(self, action: ControlAction) -> None:
        from scene_cameras import capture_rgb, save_portable_stage, validated_scene_output

        config = self.handles.config
        if action.kind == "select_joint":
            self.selected_joint = int(action.joint_index)
            print(f"Selected arm joint {self.selected_joint + 1}", flush=True)
            return
        if action.kind == "jog":
            result = self.controller.arm.jog(
                self.selected_joint,
                action.direction * float(config.data["robot"]["joint_jog_radians"]),
            )
            print(result.message, flush=True)
            return
        if action.kind == "home":
            print(self.controller.arm.home().message, flush=True)
        elif action.kind == "gripper_open":
            print(self.controller.gripper.command(1.0).message, flush=True)
        elif action.kind == "gripper_close":
            print(self.controller.gripper.command(0.0).message, flush=True)
        elif action.kind == "reset":
            self.controller.reset()
            print("Scene reset", flush=True)
        elif action.kind == "save":
            output = validated_scene_output(config, "generated_usd")
            save_portable_stage(self.handles.stage, output)
            print(f"Saved {output}", flush=True)
        elif action.kind == "capture_both":
            for camera_path, key in (
                (self.handles.cameras.agent_path, "agent_image"),
                (self.handles.cameras.wrist_path, "wrist_image"),
            ):
                result = capture_rgb(
                    camera_path,
                    validated_scene_output(config, key),
                    resolution=config.camera_resolution,
                )
                print(f"Captured {result.path}", flush=True)
        elif action.kind in {"camera_agent", "camera_wrist"}:
            from omni.kit.viewport.utility import get_active_viewport

            path = (
                self.handles.cameras.agent_path
                if action.kind == "camera_agent"
                else self.handles.cameras.wrist_path
            )
            get_active_viewport().set_active_camera(path)
            print(f"Viewport camera: {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()

    from build_scene import build_stage
    from scene_config import SceneConfig

    config = SceneConfig.load(arguments.config)
    handles = build_stage(config, headless=False)
    keyboard_controller = None
    try:
        import carb.input
        import omni.appwindow

        dispatcher = SceneActionDispatcher(handles)
        app_window = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard_controller = KeyboardController(
            input_interface, app_window.get_keyboard(), dispatcher
        )
        keyboard_controller.subscribe()
        dispatcher(ControlAction("camera_agent"))
        print("Interactive scene ready. Press 1-6 to select a joint; see README for controls.", flush=True)
        frames = 0
        while handles.app.is_running():
            handles.world.step(render=True)
            frames += 1
            if arguments.smoke_frames and frames >= arguments.smoke_frames:
                break
        return 0
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
        handles.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
