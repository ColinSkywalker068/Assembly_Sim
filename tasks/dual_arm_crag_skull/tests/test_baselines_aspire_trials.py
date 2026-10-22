"""The trial service of an ASPIRE campaign: which requests the protocol allows, what a sweep writes, reruns of harness
failures, the socket client, and the GPU sandbox command. A fake launcher stands in for Isaac Sim throughout."""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tasks.dual_arm_crag_skull.baselines import sandbox
from tasks.dual_arm_crag_skull.baselines.aspire import trial_client, trials
from tasks.dual_arm_crag_skull.baselines.aspire.trials import Refused, Rules, TrialOutcome, TrialService
from tasks.dual_arm_crag_skull.workspace import REPO_ROOT

RULES = Rules(dev_seeds=(101, 102, 103, 104, 105), unseen_seeds=(126, 127, 128, 129, 130), max_fix_versions=2,
              max_probes=2, max_reruns=2)


def settings(tmp_path, gpus=(0, 1)):
    return {'workspace': str(tmp_path / 'workspace'), 'debug': str(tmp_path / 'debug'),
            'dev_seeds': list(RULES.dev_seeds),
            'unseen_seeds': list(RULES.unseen_seeds), 'max_fix_versions': RULES.max_fix_versions,
            'max_probes': RULES.max_probes, 'max_reruns': RULES.max_reruns, 'gpus': list(gpus)}


def write(workspace, name, text):
    path = Path(workspace) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class FakeLauncher:
    """Writes a run directory with an outcome (task_success when the program contains 'win') unless the seed is in
    `crash` (no run directory: a harness failure) or `hang` (killed for the timeout); records GPU overlap."""

    def __init__(self, crash=(), hang=(), oom=(), delay=0.05):
        self.crash, self.hang, self.oom, self.delay = set(crash), set(hang), set(oom), delay
        self.calls, self.busy, self.overlap, self.lock = [], set(), False, threading.Lock()

    def __call__(self, plan, seed, gpu, program, output, log_path, trial_id):
        with self.lock:
            self.overlap |= gpu in self.busy
            self.busy.add(gpu)
            self.calls.append((plan.label, seed, gpu, Path(program).read_text()))
        time.sleep(self.delay)
        with self.lock:
            self.busy.discard(gpu)
        Path(log_path).write_text(f'seed {seed}\n')
        if seed in self.crash:
            return TrialOutcome(seed, gpu, 1, 0.1, 10.0, None, None, 'trial process exited with status 1')
        run_dir = Path(output) / 'bench' / 'suite' / 'task' / f'{seed:04d}' / f'run-{trial_id}'
        run_dir.mkdir(parents=True)
        if seed in self.hang:
            return TrialOutcome(seed, gpu, -9, 0.1, 10.0, str(run_dir), None, 'killed: no exit within 1 s')
        if seed in self.oom:
            return TrialOutcome(seed, gpu, -9, 0.1, 9e4, str(run_dir), None, 'killed: resident memory above 64 GB')
        win = 'win' in Path(program).read_text()
        (run_dir / 'outcome.json').write_text(json.dumps({'program_ok': True, 'task_success': win,
                                                          'termination_reason': 'program_completed'}))
        return TrialOutcome(seed, gpu, 0, 0.1, 10.0, str(run_dir), {'status': 'completed'}, None)


def service(tmp_path, launcher=None, gpus=(0, 1)):
    (tmp_path / 'workspace').mkdir(exist_ok=True)
    return TrialService(tmp_path, settings(tmp_path, gpus), launch=launcher or FakeLauncher())


def ask(svc, request):
    events = []
    final = svc.handle(request, events.append)
    return final, events


def test_the_first_sweep_is_v00_and_each_later_one_a_higher_new_version_within_the_budget(tmp_path):
    svc = service(tmp_path)
    workspace = svc.workspace
    write(workspace, 'code/v01_early.py', 'result = 1\n')
    assert ask(svc, {'kind': 'sweep', 'program': 'code/v01_early.py'})[0]['reason'].startswith('the first sweep is v00')
    write(workspace, 'code/v00_initial.py', 'result = 0\n')
    final, events = ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})
    assert final['event'] == 'done' and final['of'] == 5 and final['passed'] == 0
    assert [event['event'] for event in events].count('started') == 5
    write(workspace, 'code/v02_same.py', 'result = 0\n')
    assert 'byte for byte v00' in ask(svc, {'kind': 'sweep', 'program': 'code/v02_same.py'})[0]['reason']
    write(workspace, 'code/v02_win.py', 'result = "win"\n')
    assert ask(svc, {'kind': 'sweep', 'program': 'code/v02_win.py'})[0]['passed'] == 5
    write(workspace, 'code/v01_late.py', 'result = 2\n')
    assert 'not higher than v02' in ask(svc, {'kind': 'sweep', 'program': 'code/v01_late.py'})[0]['reason']
    write(workspace, 'code/v03_more.py', 'result = 3\n')
    assert ask(svc, {'kind': 'sweep', 'program': 'code/v03_more.py'})[0]['event'] == 'done'
    write(workspace, 'code/v04_over.py', 'result = 4\n')
    final = ask(svc, {'kind': 'sweep', 'program': 'code/v04_over.py'})[0]
    assert final['event'] == 'refused' and 'all 2 fix versions are used' in final['reason']
    status = ask(svc, {'kind': 'status'})[0]
    assert status['versions_swept'] == ['v00', 'v02', 'v03'] and status['fix_versions_left'] == 0


@pytest.mark.parametrize('program', ['code/../v00_initial.py', '../outside/v00_initial.py', 'v00_initial.py',
                                     'code/sub/v00_initial.py', 'code/V00_Initial.py', 'code/v0_initial.py',
                                     '/etc/passwd', 'code/v00_initial.txt'])
def test_a_program_outside_code_or_misnamed_is_refused(tmp_path, program):
    svc = service(tmp_path)
    write(svc.workspace, 'code/v00_initial.py', 'result = 0\n')
    write(tmp_path, 'outside/v00_initial.py', 'result = 0\n')
    final = ask(svc, {'kind': 'sweep', 'program': program})[0]
    assert final['event'] == 'refused' and svc.launch.calls == []


def test_a_symlinked_or_oversized_program_is_refused(tmp_path):
    svc = service(tmp_path)
    target = write(tmp_path, 'elsewhere.py', 'result = 0\n')
    (svc.workspace / 'code').mkdir()
    (svc.workspace / 'code' / 'v00_initial.py').symlink_to(target)
    assert 'not a regular file' in ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})[0]['reason']
    write(svc.workspace, 'code/v00_big.py', '#' * (trials.MAX_PROGRAM_BYTES + 1))
    assert 'larger than' in ask(svc, {'kind': 'sweep', 'program': 'code/v00_big.py'})[0]['reason']


@pytest.mark.parametrize('seed', [1, 100, 126, 106, '101', None])
def test_a_probe_runs_one_development_seed_and_no_other(tmp_path, seed):
    svc = service(tmp_path)
    write(svc.workspace, 'code/probe_scene.py', 'result = 0\n')
    final = ask(svc, {'kind': 'probe', 'program': 'code/probe_scene.py', 'seed': seed})[0]
    assert final['event'] == 'refused' and 'one development seed' in final['reason'] and svc.launch.calls == []


def test_probes_are_capped_and_run_once_each_in_their_own_directory(tmp_path):
    svc = service(tmp_path)
    write(svc.workspace, 'code/probe_scene.py', 'result = 0\n')
    for number in (1, 2):
        final, events = ask(svc, {'kind': 'probe', 'program': 'code/probe_scene.py', 'seed': 103})
        assert final['event'] == 'done' and final['of'] == 1 and events[0]['output'].endswith(f'probe0{number}_scene')
        assert (svc.debug / f'logs/explore/probe0{number}_scene.log').is_file()
    assert 'all 2 probes are used' in ask(svc, {'kind': 'probe', 'program': 'code/probe_scene.py',
                                                'seed': 103})[0]['reason']
    assert [call[:2] for call in svc.launch.calls] == [('probe01_scene', 103), ('probe02_scene', 103)]


def test_a_request_for_held_out_seeds_has_no_form(tmp_path):
    svc = service(tmp_path)
    write(svc.workspace, 'code/v00_initial.py', 'result = 0\n')
    final = ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py', 'seeds': [1, 2, 3]})[0]
    assert final['event'] == 'done' and sorted(call[1] for call in svc.launch.calls) == list(RULES.dev_seeds)
    for kind in ('evaluate', 'heldout', 'validation', None):
        assert ask(svc, {'kind': kind, 'seeds': [1]})[0]['event'] == 'refused'
    assert all(call[1] in RULES.dev_seeds for call in svc.launch.calls)


def test_the_unseen_check_runs_a_swept_version_once(tmp_path):
    svc = service(tmp_path)
    workspace = svc.workspace
    assert 'select a version first' in ask(svc, {'kind': 'unseen'})[0]['reason']
    write(workspace, 'code/v00_initial.py', 'result = "win"\n')
    ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})
    write(workspace, 'fix_code.py', 'result = "win"  # tidied\n')
    assert 'not byte for byte any swept version' in ask(svc, {'kind': 'unseen'})[0]['reason']
    write(workspace, 'fix_code.py', 'result = "win"\n')
    final, events = ask(svc, {'kind': 'unseen'})
    assert final['event'] == 'done' and final['passed'] == 5 and final['label'] == 'unseen_v00'
    assert events[0]['seeds'] == list(RULES.unseen_seeds) and events[0]['output'] == '/workspace/debug/holdout'
    assert 'run once' in ask(svc, {'kind': 'unseen'})[0]['reason']
    assert ask(svc, {'kind': 'status'})[0]['unseen_check_done'] is True


def test_a_sweep_writes_upstream_results_rows_and_logs_and_never_shares_a_gpu(tmp_path):
    launcher = FakeLauncher(delay=0.2)
    svc = service(tmp_path, launcher, gpus=(3, 5))
    write(svc.workspace, 'code/v00_initial.py', 'result = 0\n')
    ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})
    rows = [json.loads(line) for line in (svc.debug / 'initial/results.jsonl').read_text().splitlines()]
    assert sorted(row['seed'] for row in rows) == list(RULES.dev_seeds)
    assert set(rows[0]) >= {'seed', 'ok', 'run_dir', 'program_ok', 'task_success', 'termination_reason', 'wall_s',
                            'max_rss_mb', 'error'}
    assert all(row['ok'] and row['run_dir'].startswith('/workspace/debug/initial/') for row in rows)
    assert sorted(path.name for path in (svc.debug / 'logs/v00').iterdir()) == [
        f'seed_{seed:03d}.log' for seed in RULES.dev_seeds]
    assert not launcher.overlap and {call[2] for call in launcher.calls} == {3, 5}
    ledger = trials.ledger_rows(svc.ledger)
    assert ledger[-1]['status'] == 'ran' and len(ledger[-1]['trials']) == 5


def test_the_trials_run_the_bytes_the_service_checked(tmp_path):
    svc = service(tmp_path)
    source = write(svc.workspace, 'code/v00_initial.py', 'result = "checked"\n')
    original = svc.launch

    def edit_during_run(plan, seed, gpu, program, *rest):
        source.write_text('result = "edited"\n')
        return original(plan, seed, gpu, program, *rest)

    svc.launch = edit_during_run
    ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})
    assert {call[3] for call in original.calls} == {'result = "checked"\n'}


def test_harness_failures_are_rerun_without_budget_and_a_kill_is_the_programs(tmp_path):
    launcher = FakeLauncher(crash={102, 104}, hang={105}, oom={103})
    svc = service(tmp_path, launcher)
    write(svc.workspace, 'code/v00_initial.py', 'result = "win"\n')
    final = ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})[0]
    assert final['rerunnable_seeds'] == [102, 104] and final['passed'] == 1 and final['errors'] == 4
    launcher.crash = {104}
    final, events = ask(svc, {'kind': 'rerun', 'label': 'v00'})
    assert events[0]['seeds'] == [102, 104] and final['rerunnable_seeds'] == [104]
    results = svc.debug / 'initial/results.jsonl'
    rows = {json.loads(line)['seed']: json.loads(line) for line in results.read_text().splitlines()}
    assert len(results.read_text().splitlines()) == 5 and rows[102]['task_success'] is True
    assert rows[105]['error'].startswith('killed: no exit') and rows[104]['ok'] is False
    assert ask(svc, {'kind': 'status'})[0]['fix_versions_left'] == RULES.max_fix_versions
    launcher.crash = set()
    assert ask(svc, {'kind': 'rerun', 'label': 'v00'})[0]['rerunnable_seeds'] == []
    assert 'every seed of v00 has an outcome' in ask(svc, {'kind': 'rerun', 'label': 'v00'})[0]['reason']
    assert 'no request ran' in ask(svc, {'kind': 'rerun', 'label': 'v09'})[0]['reason']


def test_reruns_are_capped(tmp_path):
    launcher = FakeLauncher(crash={101})
    svc = service(tmp_path, launcher)
    write(svc.workspace, 'code/v00_initial.py', 'result = 0\n')
    ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})
    for _ in range(RULES.max_reruns):
        assert ask(svc, {'kind': 'rerun', 'label': 'v00'})[0]['event'] == 'done'
    assert 'rerun 2 times' in ask(svc, {'kind': 'rerun', 'label': 'v00'})[0]['reason']


def test_a_restarted_service_keeps_the_counts(tmp_path):
    svc = service(tmp_path)
    write(svc.workspace, 'code/v00_initial.py', 'result = 0\n')
    write(svc.workspace, 'code/probe_scene.py', 'result = 0\n')
    ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})
    ask(svc, {'kind': 'probe', 'program': 'code/probe_scene.py', 'seed': 101})
    again = service(tmp_path)
    status = ask(again, {'kind': 'status'})[0]
    assert status['versions_swept'] == ['v00'] and status['probes_left'] == 1
    write(again.workspace, 'code/v00_other.py', 'result = 5\n')
    assert 'not higher than v00' in ask(again, {'kind': 'sweep', 'program': 'code/v00_other.py'})[0]['reason']


def test_an_existing_output_directory_is_refused_and_recorded(tmp_path):
    svc = service(tmp_path)
    write(svc.workspace, 'code/v00_initial.py', 'result = 0\n')
    (svc.debug / 'initial').mkdir(parents=True)
    assert 'already exists' in ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})[0]['reason']
    assert trials.ledger_rows(svc.ledger)[-1]['status'] == 'refused' and svc.launch.calls == []


def test_the_client_talks_to_the_service_over_its_socket(tmp_path, monkeypatch, capsys):
    svc = service(tmp_path)
    write(svc.workspace, 'code/v00_initial.py', 'result = "win"\n')
    socket_path = tmp_path / 'trials.sock'
    ready = threading.Event()
    threading.Thread(target=trials.serve, args=(socket_path, svc), kwargs={'ready': ready}, daemon=True).start()
    assert ready.wait(5)
    monkeypatch.setattr(trial_client, 'SOCKET', str(socket_path))
    monkeypatch.setattr(trial_client, 'WORKSPACE', str(svc.workspace))
    monkeypatch.chdir(svc.workspace)
    assert trial_client.main(['sweep', 'code/v00_initial.py']) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]['event'] == 'accepted' and events[-1]['event'] == 'done' and events[-1]['passed'] == 5
    assert trial_client.main(['sweep', 'code/v00_initial.py']) == 3
    assert trial_client.main(['status']) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['running'] is None
    with pytest.raises(SystemExit, match='not inside'):
        trial_client.main(['sweep', str(tmp_path / 'elsewhere.py')])
    monkeypatch.setattr(trial_client, 'SOCKET', str(tmp_path / 'absent.sock'))
    assert trial_client.main(['status']) == 1


def test_the_client_needs_only_the_standard_library(tmp_path):
    script = Path(trial_client.__file__)
    found = subprocess.run([sys.executable, '-I', '-S', str(script), '--help'], capture_output=True, text=True)
    assert found.returncode == 0 and 'aspire-trial' in found.stdout


def test_watch_kills_a_trial_past_its_time(tmp_path):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    code, peak, reason = trials.watch(process, timeout_s=0.5, poll_s=0.1)
    assert code != 0 and reason.startswith('no exit within') and peak > 0
    quick = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    assert trials.watch(quick, timeout_s=30, poll_s=0.1)[::2] == (0, None)


def test_the_trial_command_sandboxes_isaac_with_the_gpu_and_no_home(tmp_path):
    env = tmp_path / 'env'
    kit = env / 'lib' / 'python3.11' / 'site-packages' / 'isaacsim' / 'kit'
    (kit / 'cache').mkdir(parents=True)
    (kit / 'logs').mkdir()
    (env / 'bin').mkdir()
    task = write(tmp_path, 'tasks/task.json', json.dumps({'layout': str(tmp_path / 'layout/layout.json'),
                                                         'fragments': [{'mesh': str(tmp_path / 'meshes/a.npz')}]}))
    workspace = tmp_path / 'workspace'
    output = tmp_path / 'debug' / 'attempts' / 'v01'
    paths = {name: tmp_path / name for name in ('record', 'kit_logs', 'home', 'kit_cache')}
    for path in [output, *paths.values()]:
        path.mkdir(parents=True)
    config = {'workspace': str(workspace), 'debug': str(tmp_path / 'debug'),
              'isaac_python': str(env / 'bin' / 'python'), 'task': str(task),
              'checkout': 'third_party/aspire-fa00f0e-fanuc_crx', 'max_steps': 18000,
              'kit_cache': str(paths['kit_cache'])}
    command = trials.isaac_trial_command(config, program=tmp_path / 'frozen.py', seed=101, output=output,
                                         record=paths['record'], kit_logs=paths['kit_logs'], home=paths['home'])
    binds = [(command[i], command[i + 1], command[i + 2]) for i in range(len(command) - 2)
             if command[i] in ('--bind', '--ro-bind', '--dev-bind')]
    inside = {target: flag for flag, _, target in binds}
    assert inside['/workspace/debug/attempts/v01'] == '--bind' and inside['/home/isaac'] == '--bind'
    assert inside[str(kit / 'cache')] == '--bind' and inside[str(env)] == '--ro-bind'
    assert inside[str(REPO_ROOT / 'tasks')] == '--ro-bind' and inside['/sys'] == '--ro-bind'
    assert {str(tmp_path / 'layout'), str(tmp_path / 'meshes'), str(task)} <= set(inside)
    assert all(inside[device] == '--dev-bind' for device in sandbox.nvidia_devices())
    assert str(REPO_ROOT / 'runs') not in inside and str(Path.home()) not in inside
    assert not any(target.startswith(str(Path.home()) + '/') for target in inside)
    tail = command[command.index('--') + 1:]
    assert tail[tail.index('--seed') + 1] == '101' and tail[tail.index('--mode') + 1] == 'practice'
    assert '--no-videos' in tail and tail[tail.index('--output-root') + 1] == '/workspace/debug/attempts/v01'
    hidden = {command[i + 1]: command[i] for i in range(len(command) - 1) if command[i] == '--tmpfs'}
    masked = {command[i + 2] for i in range(len(command) - 2) if command[i:i + 2] == ['--ro-bind', '/dev/null']}
    assert {str(REPO_ROOT / 'tasks/dual_arm_crag_skull/experiments'),
            str(REPO_ROOT / 'tasks/dual_arm_crag_skull/baselines/dexagent')} <= set(hidden)
    assert str(REPO_ROOT / 'tasks/dual_arm_crag_skull/sim/check_assembly.py') in masked
    frozen = trials.isaac_trial_command({**config, 'code': str(tmp_path / 'code')}, program=tmp_path / 'frozen.py',
                                        seed=101, output=output, record=paths['record'], kit_logs=paths['kit_logs'],
                                        home=paths['home'])
    sources = {frozen[i + 2]: frozen[i + 1] for i in range(len(frozen) - 2) if frozen[i] == '--ro-bind'}
    assert sources[str(REPO_ROOT / 'tasks')] == str(tmp_path / 'code' / 'tasks')  # the campaign's frozen copy
    assert sources[str(REPO_ROOT / 'core')] == str(tmp_path / 'code' / 'core')
    assert environment_has_no_hash_randomness(config)
    environment = trials.trial_environment(config, 4)
    assert environment['CUDA_VISIBLE_DEVICES'] == '4' and environment['HOME'] == '/home/isaac'
    words = ('KEY', 'TOKEN', 'SECRET', 'ANTHROPIC', 'SSH', 'PASSWORD')
    assert not any(word in name for name in environment for word in words)


def test_the_ledger_is_outside_the_workspace(tmp_path):
    svc = service(tmp_path)
    assert svc.workspace not in svc.ledger.parents and svc.workspace not in (svc.service_dir / 'programs').parents


def environment_has_no_hash_randomness(config):
    return trials.trial_environment(config, 0)['PYTHONHASHSEED'] == '0'


def test_a_linked_code_directory_or_a_fifo_is_refused_without_reading_it(tmp_path):
    import os
    svc = service(tmp_path)
    secret = write(tmp_path, 'secret/v00_initial.py', 'not for the agent\n')
    (svc.workspace / 'code').symlink_to(secret.parent)
    assert 'not a regular file' in ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})[0]['reason']
    (svc.workspace / 'code').unlink()
    (svc.workspace / 'code').mkdir()
    os.mkfifo(svc.workspace / 'code' / 'v00_initial.py')
    assert 'not a regular file' in ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})[0]['reason']
    (svc.workspace / 'fix_code.py').symlink_to(secret)
    assert 'select a version first' in ask(svc, {'kind': 'unseen'})[0]['reason']
    assert svc.launch.calls == [] and list((svc.service_dir / 'programs').iterdir()) == []


def test_the_service_writes_nothing_in_the_workspace(tmp_path):
    svc = service(tmp_path, FakeLauncher(crash={101}))
    write(svc.workspace, 'code/v00_initial.py', 'result = "win"\n')
    write(svc.workspace, 'code/probe_scene.py', 'result = 0\n')
    before = sorted(str(path.relative_to(svc.workspace)) for path in svc.workspace.rglob('*'))
    ask(svc, {'kind': 'probe', 'program': 'code/probe_scene.py', 'seed': 102})
    ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})
    ask(svc, {'kind': 'rerun', 'label': 'v00'})
    write(svc.workspace, 'fix_code.py', 'result = "win"\n')
    ask(svc, {'kind': 'unseen'})
    after = sorted(str(path.relative_to(svc.workspace)) for path in svc.workspace.rglob('*'))
    assert after == sorted([*before, 'fix_code.py'])
    assert {path.name for path in svc.debug.iterdir()} == {'initial', 'explore', 'holdout', 'logs'}


def test_read_regular_reads_through_real_directories_only(tmp_path):
    write(tmp_path, 'a/b/c.py', 'x = 1\n')
    assert trials.read_regular(tmp_path, ('a', 'b', 'c.py')) == b'x = 1\n'
    (tmp_path / 'link').symlink_to(tmp_path / 'a')
    with pytest.raises(Refused):
        trials.read_regular(tmp_path, ('link', 'b', 'c.py'))
    with pytest.raises(Refused, match='larger than'):
        trials.read_regular(tmp_path, ('a', 'b', 'c.py'), limit=3)


def test_a_restarted_service_sets_aside_what_it_never_recorded(tmp_path):
    svc = service(tmp_path)
    write(svc.workspace, 'code/v00_initial.py', 'result = 0\n')
    write(svc.workspace, 'code/v01_fix.py', 'result = "win"\n')
    ask(svc, {'kind': 'sweep', 'program': 'code/v00_initial.py'})
    (svc.debug / 'attempts' / 'v01' / 'partial').mkdir(parents=True)  # a sweep cut off before it was recorded
    (svc.debug / 'explore' / 'probe01_scene').mkdir(parents=True)
    again = service(tmp_path)
    assert sorted(path.split('_20')[0] for path in again.set_aside) == ['interrupted/attempts/v01',
                                                                       'interrupted/explore/probe01_scene']
    assert (again.debug / 'initial' / 'results.jsonl').is_file() and not (again.debug / 'attempts' / 'v01').exists()
    final = ask(again, {'kind': 'sweep', 'program': 'code/v01_fix.py'})[0]
    assert final['event'] == 'done' and final['passed'] == 5
    assert service(tmp_path).set_aside == []
