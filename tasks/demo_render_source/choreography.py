"""Public, import-safe interface to the legacy storyboard choreography."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.workcell.config import load_workcell_preset
from tasks.demo_render_source.paths import REPO_ROOT, require_external_output


def choreography_workcell_metadata(probe_path: Path) -> dict:
    config = load_workcell_preset("dual_arm")
    return {
        "assets": {
            "arm_usd": str(config.asset_path("arm_usd")),
            "gripper_usd": str(config.asset_path("gripper_usd")),
            "probe_json": str(Path(probe_path).expanduser().resolve()),
        },
        "arms": [
            {
                "base_pos": list(config.robot(name).base_position),
                "base_quat": list(config.robot(name).base_orientation_wxyz),
            }
            for name in config.robot_names
        ],
    }


def build_choreography(
    layout_path: Path,
    probe_path: Path,
    output_path: Path,
    fps: int = 30,
) -> Path:
    output = require_external_output(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.demo_render_source._choreography_legacy",
            str(Path(layout_path).expanduser().resolve()),
            str(Path(probe_path).expanduser().resolve()),
            str(output),
            "--fps",
            str(int(fps)),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return output
