"""The workspace around the code: git-ignored folders at the repository root (README.md, "Workspace layout").

  third_party/   upstream clones (DexAgent, ASPIRE) and generated checkouts (the patched ASPIRE); read-only references
  envs/          uv Python environments (asm, s1gpu, aspire, dexagent)
  runs/          every output: layouts, task records, smoke runs, trials, held-out claims

Paths in configs/jing.json are relative to the repository root and resolved with `repo_path`.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / 'third_party'
ENVS = REPO_ROOT / 'envs'
RUNS = REPO_ROOT / 'runs'


def repo_path(value):
    """The absolute path of a configuration value given relative to the repository root."""
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'{value!r} must be a path relative to the repository root, without ..')
    return REPO_ROOT / path
