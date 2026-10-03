"""View-only runtime and generated-output boundary for the task scene."""

from __future__ import annotations

from pathlib import Path

from .builder import build_stack_sawtooth_scene


REPO_ROOT = Path(__file__).resolve().parents[2]


def require_external_output(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve an output path and reject repository-local generated artifacts."""

    resolved = Path(path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError(f"generated output must be outside the repository: {resolved}")
    return resolved


def run_viewer(smoke_frames: int = 0) -> int:
    """Open the scene for observation, without installing any task controls."""

    if smoke_frames < 0:
        raise ValueError("smoke_frames must be non-negative")
    handles = build_stack_sawtooth_scene(headless=False)
    try:
        frames = 0
        while handles.app.is_running():
            handles.world.step(render=True)
            frames += 1
            if smoke_frames and frames >= smoke_frames:
                break
    finally:
        handles.app.close()
    return 0
