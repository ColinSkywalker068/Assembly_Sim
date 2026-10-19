"""Prepare the patched ASPIRE checkout the baseline runs against, without touching the reference clone.

`prepare_checkout` clones a local ASPIRE clone into an empty destination, detaches at the pinned commit, checks and
applies patches/0001-fanuc-crx-embodiment.patch and verifies that the working tree differs from the commit by exactly
that patch. The source clone is only read (`git clone`); nothing is written inside its working tree.
`verify_checkout` repeats the verification for an existing checkout, such as the one `cap_harness` is imported from;
it also refuses untracked files and ignored ones other than the editable install's metadata, since stale bytecode
in an ignored __pycache__ could run instead of the verified sources (run Python with -B).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ...workspace import THIRD_PARTY

PINNED_COMMIT = 'fa00f0ef2bc1d91d594b9aa502c22133c9f538f7'
PATCH = Path(__file__).resolve().parent / 'patches' / '0001-fanuc-crx-embodiment.patch'
DEFAULT_DESTINATION = THIRD_PARTY / 'aspire-fa00f0e-fanuc_crx'
ALLOWED_IGNORED = ('src/cap_harness.egg-info/',)
# Fixed so that the comparison with the patch does not depend on the user's git configuration.
DIFF_OPTIONS = ('--no-color', '--no-ext-diff', '--no-renames', '--no-relative', '--full-index', '--src-prefix=a/',
                '--dst-prefix=b/', '--unified=3', '--inter-hunk-context=0', '--diff-algorithm=myers',
                '--indent-heuristic')


@dataclass(frozen=True)
class PreparedCheckout:
    path: Path
    commit: str
    patch: Path
    patch_sha256: str


def _git(*args):
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    result = subprocess.run(['git', *args], capture_output=True, text=True, check=False, env=environment)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed with exit code {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def _check_commit(commit):
    if not isinstance(commit, str) or len(commit) != 40 or set(commit) - set('0123456789abcdef'):
        raise ValueError('commit must be a full 40-character lowercase hex hash')


def patch_body(patch):
    """The patch from its first `diff --git` line on; `git apply` ignores the explanatory header before it."""
    text = Path(patch).read_text()
    start = text.find('diff --git ')
    if start < 0 or (start > 0 and text[start - 1] != '\n'):
        raise ValueError(f'{patch} contains no git diff')
    return text[start:]


def verify_checkout(path, commit=PINNED_COMMIT, patch=PATCH):
    """Raise unless `path` is a git checkout at `commit` whose working tree differs from it by exactly `patch`."""
    _check_commit(commit)
    path, patch = Path(path).resolve(), Path(patch).resolve()
    head = _git('-C', str(path), 'rev-parse', 'HEAD').strip()
    if head != commit:
        raise ValueError(f'{path} is at {head}, expected {commit}')
    if _git('-C', str(path), 'diff', *DIFF_OPTIONS, 'HEAD', '--') != patch_body(patch):
        raise ValueError(f'{path} does not differ from {commit} by exactly {patch.name}')
    untracked = _git('-C', str(path), 'ls-files', '--others', '--exclude-standard').splitlines()
    if untracked:
        raise ValueError(f'{path} has untracked files that are not part of {patch.name}: {untracked[:5]}')
    ignored = [entry for entry in _git('-C', str(path), 'ls-files', '--others', '--ignored', '--exclude-standard',
                                       '--directory').splitlines() if entry not in ALLOWED_IGNORED]
    if ignored:
        raise ValueError(f'{path} has ignored files that could shadow the verified sources: {ignored[:5]}; remove '
                         f'them and run Python with -B')
    return PreparedCheckout(path, commit, patch, hashlib.sha256(patch.read_bytes()).hexdigest())


def prepare_checkout(source_repo, commit=PINNED_COMMIT, destination=DEFAULT_DESTINATION, patch=PATCH):
    """Clone `source_repo` into the empty `destination`, detach at `commit`, apply `patch` and verify the result.

    Refuses a non-empty destination and one inside the source working tree. If a step after the clone fails, the
    destination is left in place for inspection and must be removed before retrying.
    """
    _check_commit(commit)
    source, destination, patch = Path(source_repo).resolve(), Path(destination).resolve(), Path(patch).resolve()
    patch_body(patch)
    top = Path(_git('-C', str(source), 'rev-parse', '--show-toplevel').strip()).resolve()
    if destination == top or top in destination.parents:
        raise ValueError(f'destination {destination} is inside the source working tree {top}')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f'destination {destination} exists and is not an empty directory')
    try:
        _git('-C', str(source), 'cat-file', '-e', f'{commit}^{{commit}}')
    except RuntimeError as exc:
        raise ValueError(f'{source} has no commit {commit}') from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    _git('clone', '--quiet', '--no-hardlinks', '--no-checkout', str(source), str(destination))
    _git('-C', str(destination), '-c', 'advice.detachedHead=false', 'checkout', '--quiet', '--detach', commit)
    try:
        _git('-C', str(destination), 'apply', '--check', str(patch))
    except RuntimeError as exc:
        raise ValueError(f'{patch.name} does not apply at {commit}: {exc}') from exc
    _git('-C', str(destination), 'apply', str(patch))
    return verify_checkout(destination, commit, patch)
