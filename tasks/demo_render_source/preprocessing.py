"""Import-safe entry point for the original studded CRAG preprocessing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tasks.demo_render_source.paths import REPO_ROOT, require_external_output


def prepare_storyboard_assets(
    input_glb: Path,
    output_dir: Path,
    predicted_glb: Path | None = None,
    target_length: float = 0.40,
    pitch: float = 0.016,
) -> Path:
    output = require_external_output(output_dir)
    command = [
        sys.executable,
        "-m",
        "tasks.demo_render_source._preprocessing_legacy",
        str(Path(input_glb).expanduser().resolve()),
        str(output),
        "--length",
        str(float(target_length)),
        "--pitch",
        str(float(pitch)),
    ]
    if predicted_glb is not None:
        command.extend(("--pred", str(Path(predicted_glb).expanduser().resolve())))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return output / "layout.json"
