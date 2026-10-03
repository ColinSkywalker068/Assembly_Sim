"""Keep source, task code, and documentation separate from runtime artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ROOT_FILES = {".gitattributes", ".gitignore", "README.md", "requirements.txt"}
ALLOWED_DIRECTORIES = {"core", "tasks", "docs"}
GENERATED_SUFFIXES = {".png", ".mp4", ".tar", ".log", ".npz"}


def tracked_paths() -> tuple[Path, ...]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO_ROOT, text=True
    )
    return tuple(Path(line) for line in output.splitlines() if line)


def test_tracked_tree_has_only_architecture_roots_and_metadata():
    invalid = [
        str(path)
        for path in tracked_paths()
        if not (
            (len(path.parts) == 1 and path.name in ALLOWED_ROOT_FILES)
            or path.parts[0] in ALLOWED_DIRECTORIES
        )
    ]

    assert invalid == []


def test_tracked_tree_contains_no_generated_outputs():
    invalid = []
    for path in tracked_paths():
        robot_asset = path.parts[:3] == ("core", "assets", "robots")
        generated_name = path.suffix.lower() in GENERATED_SUFFIXES or (
            path.suffix.lower() in {".usd", ".usda", ".usdc"} and not robot_asset
        )
        if generated_name:
            invalid.append(str(path))

    assert invalid == []


def test_public_readme_documents_setup_tasks_and_further_reading():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "NYU Tandon" in readme
    assert "AI4CE" in readme
    assert "## Repository structure" in readme
    assert "## Installation" in readme
    assert "## Tasks" in readme
    assert "## Documentation" in readme
    assert "tasks.demo_render_source" in readme
    assert "tasks.dual_arm_breakingbad" in readme
    assert "/local_data/" not in readme
    assert "FANUC_CRAG" not in readme
    assert (REPO_ROOT / "requirements.txt").is_file()
