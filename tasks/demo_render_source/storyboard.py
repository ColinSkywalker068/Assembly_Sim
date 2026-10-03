"""Import-safe public interface for storyboard frame rendering."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.workcell.config import WorkcellConfig, load_workcell_preset
from tasks.demo_render_source.paths import REPO_ROOT, require_external_output


def storyboard_workcell_config() -> WorkcellConfig:
    return load_workcell_preset("dual_arm")


def render_storyboard(
    choreography_path: Path,
    output_dir: Path,
    *,
    quick: bool = False,
    subframes: int = 6,
    width: int = 960,
    height: int = 540,
    frames: str | None = None,
) -> Path:
    output = require_external_output(output_dir)
    command = [
        sys.executable,
        "-m",
        "tasks.demo_render_source._storyboard_legacy",
        str(Path(choreography_path).expanduser().resolve()),
        str(output),
        "--subframes",
        str(int(subframes)),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
    ]
    if quick:
        command.append("--quick")
    if frames:
        command.extend(("--frames", frames))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return output
