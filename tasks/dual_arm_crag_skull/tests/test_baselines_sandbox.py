import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tasks.dual_arm_crag_skull.baselines import sandbox
from tasks.dual_arm_crag_skull.workspace import REPO_ROOT

needs_bwrap = pytest.mark.skipif(not sandbox.available(), reason='bwrap is not installed')


def test_only_what_is_listed_is_bound(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    out.mkdir()
    args = sandbox.bwrap_prefix(writable=[out], readonly=[REPO_ROOT / 'core', (Path('/usr/bin'), Path('/opt/x'))])
    assert args[0] == 'bwrap' and args[-1] == '--'
    for flag in ('--unshare-all', '--die-with-parent', '--new-session'):
        assert flag in args
    binds = [(args[i], args[i + 1], args[i + 2]) for i in range(len(args) - 2) if args[i].endswith('bind')]
    assert ('--bind', str(out.resolve()), str(out)) in binds
    assert ('--ro-bind', str((REPO_ROOT / 'core').resolve()), str(REPO_ROOT / 'core')) in binds
    assert ('--ro-bind', '/usr/bin', '/opt/x') in binds
    home = str(Path.home())
    assert not any(inside == home or inside.startswith(home + '/') for _, _, inside in binds)
    assert not any(inside in ('/', '/home', '/local_data', str(REPO_ROOT)) for _, _, inside in binds)
    with pytest.raises(FileNotFoundError):
        sandbox.bwrap_prefix(writable=[tmp_path / 'missing'])
    monkeypatch.setattr(sandbox.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='bwrap'):
        sandbox.bwrap_prefix(writable=[out])


def test_a_venv_brings_its_base_interpreter_and_nothing_else_of_home(tmp_path):
    venv = tmp_path / 'venv'
    (venv / 'bin').mkdir(parents=True)
    base = tmp_path / 'home' / '.local' / 'share' / 'uv' / 'python' / 'cpython-3.11'
    (base / 'bin').mkdir(parents=True)
    (venv / 'pyvenv.cfg').write_text(f'home = {base / "bin"}\nversion_info = 3.11\n')
    assert sandbox.python_paths(venv / 'bin' / 'python') == [(venv.resolve(), venv), (base.resolve(), base)]


def _probe(tmp_path, env=None):
    out = tmp_path / 'out'
    out.mkdir(exist_ok=True)
    secret = tmp_path / 'secret.env'
    secret.write_text('not a key\n')
    readonly = [*sandbox.python_paths(sys.executable), REPO_ROOT / 'core']
    prefix = sandbox.bwrap_prefix(writable=[out], readonly=readonly, cwd=out)
    hidden = [secret, REPO_ROOT / 'runs', REPO_ROOT / 'third_party', Path.home() / '.ssh']
    return sandbox.probe(sys.executable, prefix, hidden=hidden, home=Path.home(), writable=[out],
                         readonly=[REPO_ROOT / 'core'], env=env or {'PATH': '/usr/bin:/bin', 'HOME': str(out)})


@needs_bwrap
def test_inside_there_is_no_network_no_home_and_only_the_output_directory_takes_writes(tmp_path):
    found = _probe(tmp_path)
    assert found['ok'], found
    assert found['visible'] == [] and set(found['home_listing'] or ()) <= {'.local'}
    assert found['internet'] != 'connected' and found['host_loopback'] != 'connected' and found['dns'] != 'resolves'
    assert found['writes'] == {str(tmp_path / 'out'): True, str(REPO_ROOT / 'core'): False}
    assert found['credential_env'] == [] and found['shared_memory'] is True


@needs_bwrap
def test_the_probe_fails_on_a_credential_in_the_environment(tmp_path):
    found = _probe(tmp_path, env={'PATH': '/usr/bin:/bin', 'HOME': '/tmp', 'ANTHROPIC_API_KEY': 'placeholder'})
    assert found['credential_env'] == ['ANTHROPIC_API_KEY'] and not found['ok']


@needs_bwrap
def test_killing_the_sandbox_leaves_no_process_behind(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    prefix = sandbox.bwrap_prefix(writable=[out], readonly=sandbox.python_paths(sys.executable), cwd=out)
    marker = f'sandbox-leftover-{os.getpid()}-{time.time_ns()}'
    child = ("import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)', "
             f"'{marker}']); time.sleep(600)")
    process = subprocess.Popen([*prefix, sys.executable, '-c', child], start_new_session=True)
    time.sleep(2)
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)
    time.sleep(1)
    left = subprocess.run(['pgrep', '-f', marker], capture_output=True, text=True).stdout.split()
    assert left == []


@needs_bwrap
def test_the_cli_check_reports_a_sealed_sandbox(tmp_path, capsys):
    import json
    from tasks.dual_arm_crag_skull.baselines.__main__ import main
    assert main(['sandbox-check', '--python', sys.executable, '--out', str(tmp_path / 'check')]) == 0
    found = json.loads(capsys.readouterr().out)
    assert found['ok'] and str(Path.home() / '.config') in found['hidden'] and found['visible'] == []
