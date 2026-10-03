"""Keep source, task code, and documentation separate from runtime artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ROOT_FILES = {".gitattributes", ".gitignore", "README.md"}
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
