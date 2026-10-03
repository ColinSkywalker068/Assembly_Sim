"""Output-boundary checks shared by storyboard commands."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def require_external_output(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError(f"generated output must be outside the repository: {resolved}")
    return resolved
