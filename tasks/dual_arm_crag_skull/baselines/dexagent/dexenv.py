"""The DexAgent side of the baseline: pinned clones, a scrubbed environment and subprocesses of the DexAgent venv.

DexAgent puts top-level packages named `agent` and `tool_library` on sys.path, changes directory during tool
calls, sets MUJOCO_GL and thread variables at import, ends stages with sys.exit, writes into its own tool library
(library_create, index.json), examples/asset_reuse, ~/.cache/dexagent and $TMPDIR, and runs model-written code
(run_python, library_create tests) with its whole environment. So nothing of it is imported here: every call is a
subprocess of the DexAgent venv on a fresh clone of the pinned commit, with an environment built from an allow
list (no credentials, HOME/TMPDIR/XDG_* inside the trial). A scrubbed environment does not hide this process's
own environment: a child of the same user reads /proc/<pid>/environ, so a live run first makes this process
non-dumpable (protect_process). This is not a sandbox: the model-written code still runs with this user's file
permissions and can read any other process or file of the user, so live trials belong under a separate user or
in a container on a host whose other processes and files hold no credentials.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ...workspace import ENVS, THIRD_PARTY

DEFAULT_PYTHON = str(ENVS / 'dexagent' / 'bin' / 'python')
DEFAULT_SOURCE = str(THIRD_PARTY / 'DexAgent')
WORKER = Path(__file__).with_name('worker.py')
PASS_THROUGH = ('LANG', 'LC_ALL', 'TZ')


def dexagent_python():
    return os.environ.get('DEXAGENT_PYTHON', DEFAULT_PYTHON)


def dexagent_source():
    return os.environ.get('DEXAGENT_SOURCE', DEFAULT_SOURCE)


def _git(*args, cwd=None):
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, timeout=600,
                            env={'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': os.environ.get('HOME', '/tmp'),
                                 'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_TERMINAL_PROMPT': '0'})
    if result.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)} failed: {result.stderr.strip()[-2000:]}')
    return result.stdout.strip()


def verify_clone(path, commit, ignored=False):
    """The clone's HEAD is `commit` and its work tree is clean (with `ignored`, not even ignored files such as
    __pycache__/ exist, which Python would load); returns the commit."""
    path = Path(path)
    if not (path / '.git').exists():
        raise ValueError(f'{path} is not a git clone of DexAgent')
    head = _git('rev-parse', 'HEAD', cwd=path)
    if head != commit:
        raise ValueError(f'{path} is at {head}, not the pinned DexAgent commit {commit}')
    dirty = _git('status', '--porcelain', '--untracked-files=all', *(['--ignored'] if ignored else []), cwd=path)
    if dirty:
        raise ValueError(f'{path} has local changes; trials need an unmodified clone:\n{dirty[:2000]}')
    return head


def fresh_clone(source, destination, commit):
    """A new clone of `source` checked out (detached) at `commit`; refuses an existing destination."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'{destination} exists; every trial gets a fresh clone')
    destination.parent.mkdir(parents=True, exist_ok=True)
    _git('clone', '--quiet', '--no-checkout', str(source), str(destination))
    _git('-c', 'advice.detachedHead=false', 'checkout', '--quiet', '--detach', commit, cwd=destination)
    return verify_clone(destination, commit, ignored=True)


def protect_process():
    """Make this process non-dumpable (Linux prctl PR_SET_DUMPABLE 0): processes of the same user, DexAgent's
    included, can then no longer read its /proc/<pid>/environ or memory, where the API key lives."""
    import ctypes
    if not sys.platform.startswith('linux'):
        raise RuntimeError('protect_process needs Linux prctl; run live trials on Linux')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE, PR_GET_DUMPABLE
        raise RuntimeError(f'prctl(PR_SET_DUMPABLE, 0) failed (errno {ctypes.get_errno()})')


def verify_venv(python):
    """Installed files of the DexAgent venv that no longer match the sha256 in their package RECORD."""
    import base64
    import csv
    import hashlib
    root = Path(os.path.abspath(python)).parent.parent  # the venv, not the interpreter its python links to
    changed = []
    for record in sorted(root.glob('lib/python*/site-packages/*.dist-info/RECORD')):
        site = record.parent.parent
        with record.open(newline='') as handle:
            for row in csv.reader(handle):
                if len(row) < 2 or not row[1].startswith('sha256='):
                    continue
                path = (site / row[0]).resolve()
                expected = row[1].split('=', 1)[1]
                if not path.is_file():
                    changed.append(str(path))
                    continue
                digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b'=').decode()
                if digest != expected:
                    changed.append(str(path))
    return changed


def scrubbed_env(trial_dir, python=None):
    """A minimal environment for DexAgent processes: an allow list, never the caller's credentials."""
    trial = Path(trial_dir).resolve()
    dirs = {'HOME': trial / 'home', 'TMPDIR': trial / 'tmp', 'XDG_CACHE_HOME': trial / 'home' / '.cache',
            'XDG_CONFIG_HOME': trial / 'home' / '.config', 'XDG_DATA_HOME': trial / 'home' / '.local' / 'share',
            'MPLCONFIGDIR': trial / 'home' / '.config' / 'matplotlib'}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    bin_dir = str(Path(python or dexagent_python()).parent)
    env = {key: str(value) for key, value in dirs.items()}
    env.update({key: os.environ[key] for key in PASS_THROUGH if key in os.environ})
    env.update({'PATH': f'{bin_dir}:/usr/local/bin:/usr/bin:/bin', 'MUJOCO_GL': 'osmesa', 'PYOPENGL_PLATFORM': 'osmesa',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1', 'DEXAGENT_PLAIN': '1',
                'GIT_TERMINAL_PROMPT': '0'})
    return env


def credential_keys(env):
    """Names in `env` that look like credentials (used to assert the scrub)."""
    markers = ('ANTHROPIC', 'OPENAI', 'API_KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'AWS_', 'GOOGLE_', 'HF_', 'SSH_AUTH')
    return sorted(key for key in env if any(marker in key.upper() for marker in markers))


def run_worker(python, command, spec, log_path, *, env, timeout_s=1800):
    """Run worker.py `command` with `spec` in the DexAgent venv; returns its result or raises RuntimeError."""
    log_path = Path(log_path)
    spec_path = log_path.with_suffix('.json')
    spec_path.write_text(json.dumps(spec, indent=1))
    if env is None:
        raise ValueError('run_worker needs an explicit scrubbed environment (dexenv.scrubbed_env)')
    if credential_keys(env):
        raise ValueError(f'refusing to pass credentials to DexAgent: {credential_keys(env)}')
    with log_path.open('w') as log:
        result = subprocess.run([str(python), '-I', '-B', str(WORKER), command, f'@{spec_path}'], stdout=subprocess.PIPE,
                                stderr=log, text=True, env=env, cwd=spec.get('out_dir') or spec_path.parent,
                                timeout=timeout_s)
    lines = result.stdout.strip().splitlines()
    try:
        reply = json.loads(lines[-1])
    except (IndexError, ValueError):
        reply = {'ok': False, 'error': f'worker printed no JSON result (exit {result.returncode})'}
    if not reply.get('ok'):
        tail = log_path.read_text()[-3000:] if log_path.exists() else ''
        raise RuntimeError(f'DexAgent worker {command} failed: {reply.get("error")}\n{reply.get("traceback", "")}\n'
                           f'--- {log_path} ---\n{tail}')
    return reply['result']


def available(python=None, source=None):
    """(ok, reason): whether the DexAgent venv and reference clone exist (integration tests skip otherwise)."""
    python, source = Path(python or dexagent_python()), Path(source or dexagent_source())
    if not python.exists():
        return False, f'no DexAgent python at {python} (set DEXAGENT_PYTHON)'
    if not (source / '.git').exists():
        return False, f'no DexAgent clone at {source} (set DEXAGENT_SOURCE)'
    return True, ''
