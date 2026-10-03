"""Small runtime helpers shared by interactive and rendering tasks."""

from __future__ import annotations


def step_frames(handles, count: int, render: bool = True) -> None:
    if count < 0:
        raise ValueError("frame count must be non-negative")
    for _ in range(count):
        handles.world.step(render=render)


def reset_workcell(handles) -> None:
    if handles.controller is None:
        raise RuntimeError("workcell controller is not initialized")
    handles.controller.reset()


def close_workcell(handles) -> None:
    handles.app.close()
