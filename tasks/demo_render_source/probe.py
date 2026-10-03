"""Import-safe launcher for FANUC/Robotiq model probing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.workcell.config import load_workcell_preset
from tasks.demo_render_source.paths import REPO_ROOT, require_external_output


def probe_robot_models(output_dir: Path) -> Path:
    output = require_external_output(output_dir)
    config = load_workcell_preset("dual_arm")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.demo_render_source._probe_legacy",
            str(config.asset_path("arm_usd")),
            str(config.asset_path("gripper_usd")),
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return output / "fanuc_probe.json"
