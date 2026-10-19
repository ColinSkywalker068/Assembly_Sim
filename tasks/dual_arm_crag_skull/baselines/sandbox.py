"""Bubblewrap (bwrap) sandboxes for the processes that run model-written code: every DexAgent process of a trial (its
Stage 3 runs the tool code the agent writes) and ASPIRE's programs.

bwrap starts the process in an empty root with its own namespaces, so only what is bound exists inside: the system
directories (/usr, /etc and the /bin, /sbin, /lib, /lib64 links) and the listed code and environments read-only, the
listed output directories writable, a private /tmp, its own /proc and a minimal /dev. There is no network (its own
network namespace has only its own loopback: neither the internet nor anything listening on the host is reachable),
no home directory and no other project; a venv whose base interpreter lives under the home directory gets that
interpreter's directory alone, read-only, at the path the venv names. The model is called from outside (the relays)
through files in the output directory, so the API key never enters. `probe` checks these properties from inside a
sandbox. GPU access (for ASPIRE's programs, which run with Isaac Sim) is not provided yet.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

SYSTEM_PATHS = ('/usr', '/etc', '/bin', '/sbin', '/lib', '/lib32', '/lib64')


def available():
    return shutil.which('bwrap') is not None


def python_paths(python):
    """What a venv's interpreter needs read-only: the venv and its base interpreter's directory (pyvenv.cfg `home`
    is that directory's bin/), as (source, path inside) pairs; the source is resolved, the path is the one the venv
    names."""
    venv = Path(python).absolute().parent.parent
    pairs = [(venv.resolve(), venv)]
    config = venv / 'pyvenv.cfg'
    if config.exists():
        for line in config.read_text().splitlines():
            key, _, value = line.partition('=')
            if key.strip() == 'home':
                base = Path(value.strip()).parent
                pairs.append((base.resolve(), base))
    return pairs


def bwrap_prefix(*, writable, readonly=(), cwd=None):
    """The bwrap command that runs what follows it in a sandbox (see the module docstring). `readonly` holds paths
    or (source, path inside) pairs; `writable` paths must exist."""
    if not available():
        raise RuntimeError('bwrap is not installed: model-written code is never run outside a sandbox')
    args = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--proc', '/proc', '--dev', '/dev',
            '--tmpfs', '/tmp']
    for path in SYSTEM_PATHS:
        if os.path.islink(path):
            args += ['--symlink', os.readlink(path), path]
        elif os.path.isdir(path):
            args += ['--ro-bind', path, path]
    for entry in readonly:
        source, inside = entry if isinstance(entry, tuple) else (Path(entry).resolve(), Path(entry))
        args += ['--ro-bind', str(source), str(inside)]
    for path in writable:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f'writable directory {path} does not exist')
        args += ['--bind', str(path.resolve()), str(path.absolute())]
    if cwd is not None:
        args += ['--chdir', str(Path(cwd).absolute())]
    return args + ['--']


PROBE = r'''
import json, os, socket, sys
hidden, home, writable, readonly, port = json.loads(sys.argv[1])
out = {'visible': [p for p in hidden if os.path.exists(p)],
       'home_listing': sorted(os.listdir(home)) if os.path.isdir(home) else None}
def connect(host, port_):
    try:
        socket.create_connection((host, port_), timeout=3).close()
        return 'connected'
    except OSError as exc:
        return type(exc).__name__ + ': ' + str(exc)
out['internet'] = connect('1.1.1.1', 443)
out['host_loopback'] = connect('127.0.0.1', port)
try:
    socket.getaddrinfo('api.anthropic.com', 443)
    out['dns'] = 'resolves'
except OSError as exc:
    out['dns'] = type(exc).__name__
def write(directory):
    path = os.path.join(directory, '.sandbox_probe')
    try:
        with open(path, 'w') as handle:
            handle.write('x')
        os.remove(path)
        return True
    except OSError:
        return False
out['writes'] = {d: write(d) for d in writable + readonly}
words = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'ANTHROPIC', 'OPENAI', 'AWS', 'SSH_AUTH', 'CREDENTIAL')
out['credential_env'] = sorted(k for k in os.environ if any(w in k.upper() for w in words))
out['processes'] = sorted(int(p) for p in os.listdir('/proc') if p.isdigit())
try:  # DexAgent's workers need POSIX semaphores (/dev/shm)
    import multiprocessing
    lock = multiprocessing.get_context('fork').Lock()
    lock.acquire()
    lock.release()
    out['shared_memory'] = True
except Exception:
    out['shared_memory'] = False
print(json.dumps(out))
'''


def probe(python, prefix, *, hidden, home, writable, readonly, env):
    """Run the checks inside a sandbox (`prefix` from bwrap_prefix) with the venv's interpreter: which of the `hidden`
    paths are visible, what the outside home directory shows, whether the internet, the host's loopback (a listener
    on it is opened here) and DNS answer, which directories take a write, credential-like variables, and the
    processes it can see, and whether POSIX semaphores work. Returns the findings and `ok` (every property as the
    module docstring says)."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    try:
        arguments = json.dumps([list(map(str, hidden)), str(home), list(map(str, writable)), list(map(str, readonly)),
                                listener.getsockname()[1]])
        result = subprocess.run([*prefix, str(python), '-I', '-c', PROBE, arguments], capture_output=True,
                                text=True, env=env, timeout=120)
    finally:
        listener.close()
    if result.returncode != 0:
        return {'ok': False, 'error': result.stderr[-2000:]}
    found = json.loads(result.stdout)
    writes = found['writes']
    found['ok'] = (not found['visible'] and set(found['home_listing'] or ()) <= {'.local'}
                   and found['internet'] != 'connected' and found['host_loopback'] != 'connected'
                   and found['dns'] != 'resolves' and all(writes[str(d)] for d in writable)
                   and not any(writes[str(d)] for d in readonly) and not found['credential_env']
                   and len(found['processes']) <= 3 and found['shared_memory'])
    return found

