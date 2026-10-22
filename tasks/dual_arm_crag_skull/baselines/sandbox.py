"""Bubblewrap (bwrap) sandboxes for the processes that run model-written code: every DexAgent process of a trial (its
Stage 3 runs the tool code the agent writes) and ASPIRE's programs.

bwrap starts the process in an empty root with its own namespaces, so only what is bound exists inside: the system
directories (/usr, /etc and the /bin, /sbin, /lib, /lib64 links) and the listed code and environments read-only, the
listed output directories writable, a private /tmp, its own /proc and a minimal /dev. There is no network (its own
network namespace has only its own loopback: neither the internet nor anything listening on the host is reachable),
no home directory and no other project; a venv whose base interpreter lives under the home directory gets that
interpreter's directory alone, read-only, at the path the venv names. The model is called from outside (the relays)
through files in the output directory, so the API key never enters. `probe` checks these properties from inside a
sandbox.

ASPIRE's trials run Isaac Sim, so their sandbox also gets the GPUs: `devices=nvidia_devices()` binds every NVIDIA
device node (CUDA_VISIBLE_DEVICES picks the GPU), `system_info` binds /sys read-only (CUDA reads the PCI topology
there), and nothing else of the host's /dev appears. A Unix socket bound in (`sockets`) is the one way a sandbox can
reach a service outside, such as the trial service or the model relay of an ASPIRE campaign.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import stat
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


def nvidia_devices():
    """The NVIDIA device nodes a CUDA or Vulkan process opens: every GPU's, the control, UVM and modeset nodes, and
    /dev/nvidia-caps."""
    nodes = sorted(str(path) for path in Path('/dev').glob('nvidia*') if path.is_char_device())
    return nodes + (['/dev/nvidia-caps'] if os.path.isdir('/dev/nvidia-caps') else [])


def _pair(entry):
    return entry if isinstance(entry, tuple) else (Path(entry).resolve(), Path(entry).absolute())


def bwrap_prefix(*, writable, readonly=(), cwd=None, devices=(), system_info=False, sockets=(), shadows=(),
                 hidden=()):
    """The bwrap command that runs what follows it in a sandbox (see the module docstring). `readonly` and `writable`
    hold paths or (source, path inside) pairs, bound in that order, so a writable directory may sit inside a read-only
    one; writable sources must be directories that exist. `shadows` are read-only (source, path inside) binds made
    after those, so one can cover part of a writable directory; `hidden` paths inside appear empty (a directory) or
    as /dev/null (a file). `devices` are device nodes bound as they are, `system_info` binds /sys read-only, and
    `sockets` are (source, path inside) pairs of Unix sockets."""
    if not available():
        raise RuntimeError('bwrap is not installed: model-written code is never run outside a sandbox')
    args = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--proc', '/proc', '--dev', '/dev',
            '--tmpfs', '/tmp']
    for device in devices:
        args += ['--dev-bind', str(device), str(device)]
    if system_info:
        args += ['--ro-bind', '/sys', '/sys']
    for path in SYSTEM_PATHS:
        if os.path.islink(path):
            args += ['--symlink', os.readlink(path), path]
        elif os.path.isdir(path):
            args += ['--ro-bind', path, path]
    for entry in readonly:
        source, inside = _pair(entry)
        args += ['--ro-bind', str(source), str(inside)]
    for entry in writable:
        source, inside = _pair(entry)
        if not Path(source).is_dir():
            raise FileNotFoundError(f'writable directory {source} does not exist')
        args += ['--bind', str(Path(source).resolve()), str(inside)]
    for source, inside in shadows:
        args += ['--ro-bind', str(Path(source).resolve()), str(inside)]
    for path in hidden:
        if os.path.isdir(path):
            args += ['--tmpfs', str(path)]
        elif os.path.lexists(path):
            args += ['--ro-bind', '/dev/null', str(path)]
    for source, inside in sockets:
        if not stat.S_ISSOCK(os.stat(source).st_mode):
            raise ValueError(f'{source} is not a Unix socket')
        args += ['--bind', str(source), str(inside)]
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

