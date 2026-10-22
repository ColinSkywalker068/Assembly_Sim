"""The trial service of an ASPIRE campaign: the coding agent's only way to run the simulator.

The agent works in a sandbox with neither a GPU nor a simulator (campaign.py). To run a program it asks this service,
over a Unix socket bound into its sandbox (`trial_client.py`), and the service applies the campaign's protocol, the
scaled iterative-debugging protocol of the README, before anything runs:

- `sweep`: a numbered version `code/vNN_<slug>.py` on every development seed. `v00` first, then at most
  `max_fix_versions` more (upstream's "8 in total"), each with a higher number and bytes no earlier sweep ran.
- `probe`: an observation program `code/probe_<slug>.py` on one development seed, at most `max_probes` in all.
- `unseen`: `fix_code.py`, byte for byte one of the swept versions, on the unseen-check seeds, once.
- `rerun`: the seeds of an earlier request that left no outcome because the harness failed (no run directory or no
  `outcome.json`, other than a trial this service killed for its time or memory, which is the program's failure),
  with the same frozen program, at most `max_reruns` times per request; it uses nothing of the budget.
- `status`: what has been used and what is left, and the request running now; `wait` blocks until it has finished.

No request names a held-out seed. The service reads a program without following any symbolic link (the agent owns
the workspace and could swap a file for a link to anything this user can read) and runs a frozen copy, so the trials
run the bytes it checked. Each seed is one Isaac trial (`python -m ...baselines.aspire run`) in a GPU sandbox of its
own (`isaac_trial_command`), at most one per GPU at a time, writing ASPIRE's run directory like upstream's
`cap-harness run-batch` (`results.jsonl` in the output root, one log per seed). Everything the service writes goes into
its own `debug` directory, which the agent's sandbox shows read-only at /workspace/debug, so no path the agent made
can redirect a write. The service's ledger, which the agent cannot see, records every request, refused ones included,
and every trial; the counts are recomputed from it, so a restarted service keeps them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import signal
import socketserver
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import sandbox
from ...workspace import REPO_ROOT

INSIDE_WORKSPACE = Path('/workspace')
SWEEP_NAME = re.compile(r'v(\d{2})_[a-z0-9][a-z0-9_]{0,47}\.py')
PROBE_NAME = re.compile(r'probe_[a-z0-9][a-z0-9_]{0,47}\.py')
MAX_PROGRAM_BYTES = 256 * 1024
RSS_LIMIT_KB = 64 * 1024 * 1024
KINDS = ('sweep', 'probe', 'unseen', 'rerun', 'status', 'wait')
KILLED = 'killed: '  # the error of a trial this service killed (time or memory): the program's failure
# Our scripted solution, the other baseline and the tests, which no trial imports: hidden from the trial sandbox, so a
# program that escaped ASPIRE's validator could not read them either.
TRIAL_HIDDEN = ('tasks/dual_arm_crag_skull/sim/assemble.py', 'tasks/dual_arm_crag_skull/sim/check_assembly.py',
                'tasks/dual_arm_crag_skull/sim/check_grasp.py', 'tasks/dual_arm_crag_skull/experiments',
                'tasks/dual_arm_crag_skull/baselines/dexagent', 'tasks/dual_arm_crag_skull/tests')
SOCKET_PATH_MAX = 107


class Refused(ValueError):
    """A request the protocol does not allow; nothing ran."""


@dataclass(frozen=True)
class Rules:
    dev_seeds: tuple
    unseen_seeds: tuple
    max_fix_versions: int = 8
    max_probes: int = 12
    max_reruns: int = 2

    @classmethod
    def from_settings(cls, settings):
        return cls(tuple(settings['dev_seeds']), tuple(settings['unseen_seeds']), int(settings['max_fix_versions']),
                   int(settings['max_probes']), int(settings['max_reruns']))


@dataclass(frozen=True)
class Plan:
    """What an allowed request runs: `program` is workspace-relative, `output` and `logs` are paths under /workspace
    in debug/, the service's directory."""
    kind: str
    label: str
    program: str
    sha256: str
    seeds: tuple
    output: str
    logs: str


@dataclass(frozen=True)
class TrialOutcome:
    seed: int
    gpu: int | None
    exit_code: int | None
    wall_s: float
    max_rss_mb: float
    run_dir: str | None
    record: dict | None
    error: str | None


def ledger_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _ran(rows, kind):
    return [row for row in rows if row['kind'] == kind and row['status'] == 'ran']


def _harness_failed(trial):
    """Whether a trial (TrialOutcome as a dict) ended without an outcome for a reason that is not the program's."""
    if (trial['error'] or '').startswith(KILLED):
        return False
    return trial['run_dir'] is None or not (Path(trial['run_dir']) / 'outcome.json').is_file()


def missing_seeds(rows, label):
    """Seeds of the request `label` whose latest trial left no outcome because the harness failed."""
    latest = {}
    for row in rows:
        if row['status'] == 'ran' and row['label'] == label:
            latest.update({trial['seed']: trial for trial in row['trials']})
    return [seed for seed, trial in latest.items() if _harness_failed(trial)]


def usage(rules, rows):
    """What the requests in the ledger `rows` have used of `rules`, and what is left."""
    sweeps = _ran(rows, 'sweep')
    fixes = [row for row in sweeps if row['label'] != 'v00']
    return {'versions_swept': [row['label'] for row in sweeps],
            'fix_versions_left': rules.max_fix_versions - len(fixes),
            'probes_run': len(_ran(rows, 'probe')), 'probes_left': rules.max_probes - len(_ran(rows, 'probe')),
            'unseen_check_done': bool(_ran(rows, 'unseen')), 'dev_seeds': list(rules.dev_seeds),
            'unseen_seeds': list(rules.unseen_seeds)}


def read_regular(directory, parts, *, limit=MAX_PROGRAM_BYTES):
    """The bytes of the regular file `directory`/`parts...`, opened one component at a time without following a
    symbolic link (nor blocking on a FIFO), so a link the agent swaps in at any moment is refused, never read."""
    name = '/'.join(parts)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            inner = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = inner
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except OSError as exc:
        raise Refused(f'{name} is not a regular file in the workspace ({exc.strerror})') from None
    finally:
        os.close(fd)
    with os.fdopen(file_fd, 'rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise Refused(f'{name} is not a regular file in the workspace')
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise Refused(f'{name} is larger than {limit} bytes')
    return data


def _program(workspace, relative, pattern, kind):
    """The workspace-relative name and bytes of the program `relative`, a regular file in code/ named for `kind`."""
    parts = Path(relative).parts
    if len(parts) != 2 or parts[0] != 'code' or not pattern.fullmatch(parts[1]):
        expected = 'code/vNN_<slug>.py' if kind == 'sweep' else 'code/probe_<slug>.py'
        raise Refused(f'a {kind} runs {expected} (lowercase slug), not {relative!r}')
    return f'code/{parts[1]}', read_regular(workspace, parts)


def plan_request(request, workspace, rules, rows):
    """The Plan of `request` (a dict from the client) under `rules`, given the ledger `rows`; Refused otherwise."""
    kind = request.get('kind')
    if kind not in KINDS or kind in ('status', 'wait'):
        raise Refused(f'kind must be one of sweep, probe, unseen, rerun (or status, wait), not {kind!r}')
    if kind == 'rerun':
        return _plan_rerun(str(request.get('label', '')), rules, rows)
    swept = {row['sha256']: row['label'] for row in _ran(rows, 'sweep')}
    left = usage(rules, rows)
    if kind == 'sweep':
        name, data = _program(workspace, str(request.get('program', '')), SWEEP_NAME, kind)
        number = int(SWEEP_NAME.fullmatch(Path(name).name).group(1))
        label, digest = f'v{number:02d}', hashlib.sha256(data).hexdigest()
        numbers = [int(row['label'][1:]) for row in _ran(rows, 'sweep')]
        if not numbers and number != 0:
            raise Refused('the first sweep is v00 (the initial program)')
        if numbers and number <= max(numbers):
            raise Refused(f'{label} is not higher than v{max(numbers):02d}, the last version swept; a fix is a new '
                          'numbered file')
        if numbers and left['fix_versions_left'] <= 0:
            raise Refused(f'all {rules.max_fix_versions} fix versions are used; select among the swept versions')
        if digest in swept:
            raise Refused(f'{name} is byte for byte {swept[digest]}, which has been swept')
        output = 'debug/initial' if number == 0 else f'debug/attempts/{label}'
        return Plan('sweep', label, name, digest, rules.dev_seeds, output, f'debug/logs/{label}'), data
    if kind == 'probe':
        name, data = _program(workspace, str(request.get('program', '')), PROBE_NAME, kind)
        seed = request.get('seed')
        if seed not in rules.dev_seeds:
            raise Refused(f'a probe runs on one development seed {list(rules.dev_seeds)}, not {seed!r}')
        if left['probes_left'] <= 0:
            raise Refused(f'all {rules.max_probes} probes are used')
        label = f'probe{left["probes_run"] + 1:02d}_{Path(name).stem[len("probe_"):]}'
        return Plan('probe', label, name, hashlib.sha256(data).hexdigest(), (seed,), f'debug/explore/{label}',
                    'debug/logs/explore'), data
    if left['unseen_check_done']:
        raise Refused('the unseen-check seeds have been run; they are run once')
    try:
        data = read_regular(workspace, ('fix_code.py',))
    except Refused as exc:
        raise Refused(f'{exc}; select a version first') from None
    digest = hashlib.sha256(data).hexdigest()
    if digest not in swept:
        raise Refused('fix_code.py is not byte for byte any swept version; copy the selected version unchanged')
    return Plan('unseen', f'unseen_{swept[digest]}', 'fix_code.py', digest, rules.unseen_seeds, 'debug/holdout',
                'debug/logs/holdout'), data


def _plan_rerun(label, rules, rows):
    first = next((row for row in rows if row['status'] == 'ran' and row['label'] == label and
                  row['kind'] != 'rerun'), None)
    if first is None:
        raise Refused(f'no request ran under the label {label!r}')
    seeds = missing_seeds(rows, label)
    if not seeds:
        raise Refused(f'every seed of {label} has an outcome; only harness failures are rerun')
    if len([row for row in _ran(rows, 'rerun') if row['label'] == label]) >= rules.max_reruns:
        raise Refused(f'{label} has been rerun {rules.max_reruns} times')
    return Plan('rerun', label, first['program'], first['sha256'], tuple(seeds), first['output'], first['logs']), None


def _tree_rss_kb(pid):
    """Resident memory of `pid` and its descendants (kB), from /proc."""
    total, stack = 0, [pid]
    while stack:
        current = stack.pop()
        try:
            status = Path(f'/proc/{current}/status').read_text()
            total += int(next((line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')), 0))
            for task in Path(f'/proc/{current}/task').iterdir():
                stack.extend(int(child) for child in (task / 'children').read_text().split())
        except (OSError, ValueError):
            continue
    return total


def watch(process, *, timeout_s, rss_limit_kb=RSS_LIMIT_KB, poll_s=5.0):
    """Wait for `process` (its own session); kill its group on timeout or when its tree holds more than
    `rss_limit_kb`. Returns (exit code, peak kB, reason it was killed or None)."""
    deadline, peak, reason = time.monotonic() + timeout_s, 0, None
    while process.poll() is None:
        peak = max(peak, _tree_rss_kb(process.pid))
        if peak > rss_limit_kb:
            reason = f'resident memory above {rss_limit_kb // 1024 // 1024} GB'
        elif time.monotonic() > deadline:
            reason = f'no exit within {timeout_s:.0f} s'
        if reason is not None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            break
        try:
            process.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            pass
    return process.returncode, peak, reason


def kit_paths(isaac_python):
    """The Isaac Sim directories a trial must be able to write inside its read-only environment (Kit's cache and logs
    under the isaacsim package)."""
    environment = Path(isaac_python).absolute().parent.parent
    kits = sorted({kit.resolve() for kit in environment.glob('lib/python3*/site-packages/isaacsim/kit')})
    if len(kits) != 1:
        raise ValueError(f'{environment} does not hold exactly one isaacsim/kit directory')
    return environment, kits[0] / 'cache', kits[0] / 'logs'


def task_inputs(task_path):
    """The task record and the directories of the layout and meshes it names: what a trial reads besides code."""
    record = json.loads(Path(task_path).read_text())
    files = [record['layout'], *(fragment['mesh'] for fragment in record['fragments'])]
    return [Path(task_path).absolute(), *sorted({Path(name).parent for name in files})]


def inside_path(settings, host):
    """Where the service's `host` path (under settings['debug']) is in the sandboxes: /workspace/debug/..."""
    return INSIDE_WORKSPACE / 'debug' / Path(host).relative_to(settings['debug'])


def isaac_trial_command(settings, *, program, seed, output, record, kit_logs, home):
    """argv of one Isaac trial of `program` (a frozen copy) on `seed`, in a GPU sandbox that sees the code (the
    campaign's frozen copy, settings['code'], at the live code's paths; less TRIAL_HIDDEN), the task's inputs and the
    Isaac environment read-only, and writes only the trial's output root (`output`, at its /workspace path), its
    record directory, Kit's cache and logs and its home."""
    environment, kit_cache, kit_log_dir = kit_paths(settings['isaac_python'])
    checkout = REPO_ROOT / settings['checkout']
    inside_output = inside_path(settings, output)
    code = Path(settings['code']) if settings.get('code') else REPO_ROOT  # a campaign's frozen copy (prepare)
    readonly = [(code / 'tasks', REPO_ROOT / 'tasks'), (code / 'core', REPO_ROOT / 'core'), checkout, environment,
                *task_inputs(settings['task']), (Path(program), Path('/program/program.py'))]
    writable = [(Path(output), inside_output), (Path(record), Path('/record')),
                (Path(settings['kit_cache']), kit_cache), (Path(kit_logs), kit_log_dir),
                (Path(home), Path('/home/isaac'))]
    prefix = sandbox.bwrap_prefix(writable=writable, readonly=readonly, cwd=REPO_ROOT,
                                  devices=sandbox.nvidia_devices(), system_info=True,
                                  hidden=[REPO_ROOT / path for path in TRIAL_HIDDEN])
    return [*prefix, str(Path(settings['isaac_python']).absolute()), '-u', '-B', '-m',
            'tasks.dual_arm_crag_skull.baselines.aspire', 'run', '--program', '/program/program.py', '--task',
            str(Path(settings['task']).absolute()), '--seed', str(seed), '--mode', 'practice', '--output-root',
            str(inside_output), '--backend', 'isaac', '--no-videos', '--max-steps', str(settings['max_steps']),
            '--records', '/record/record.jsonl']


def trial_environment(settings, gpu):
    """The whole environment of a trial process: nothing inherited, so no credential can reach it."""
    return {'PATH': '/usr/bin:/bin', 'HOME': '/home/isaac', 'LANG': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': str(gpu),
            'OMNI_KIT_ACCEPT_EULA': 'YES', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0',
            'PYTHONPATH': f'.:{Path(settings["checkout"]) / "src"}'}


def run_dirs(output, seed):
    """The run directories of `seed` under `output` (ASPIRE nests them by benchmark, suite, task and seed)."""
    return set(Path(output).glob(f'*/*/*/{seed:04d}/run-*'))


class IsaacLauncher:
    """Runs one seed as an Isaac trial in its GPU sandbox (isaac_trial_command) and reads what it left."""

    def __init__(self, settings, service_dir):
        self.settings, self.service_dir = settings, Path(service_dir)

    def __call__(self, plan, seed, gpu, program, output, log_path, trial_id):
        trial = self.service_dir / 'trials' / trial_id
        record_dir, kit_logs, home = trial / 'record', trial / 'kit_logs', self.service_dir / 'isaac_home'
        for directory in (record_dir, kit_logs, home, Path(self.settings['kit_cache'])):
            directory.mkdir(parents=True, exist_ok=True)
        command = isaac_trial_command(self.settings, program=program, seed=seed, output=output, record=record_dir,
                                      kit_logs=kit_logs, home=home)
        started, before = time.monotonic(), run_dirs(output, seed)
        with open(log_path, 'ab') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       env=trial_environment(self.settings, gpu), start_new_session=True)
            code, peak, killed = watch(process, timeout_s=float(self.settings['trial_timeout_s']))
        records = ledger_rows(record_dir / 'record.jsonl')
        new = sorted(run_dirs(output, seed) - before)
        run_dir = new[-1] if len(new) == 1 else None
        error = f'killed: {killed}' if killed else (
            None if code == 0 else f'trial process exited with status {code}; see the seed log')
        return TrialOutcome(seed, gpu, code, time.monotonic() - started, peak / 1024,
                            None if run_dir is None else str(run_dir), records[-1] if records else None, error)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _outcome_json(run_dir):
    try:
        return json.loads((Path(run_dir) / 'outcome.json').read_text())
    except (OSError, ValueError, TypeError):
        return {}


class TrialService:
    """Plans, runs and records the agent's requests (see the module docstring), one request at a time."""

    def __init__(self, campaign_dir, settings, *, launch=None):
        self.dir = Path(campaign_dir)
        self.settings = settings
        self.workspace = Path(settings['workspace'])
        self.debug = Path(settings['debug'])  # /workspace/debug in the sandboxes; only this service writes in it
        self.service_dir = self.dir / 'service'
        self.ledger = self.service_dir / 'ledger.jsonl'
        self.rules = Rules.from_settings(settings)
        self.gpus = [int(gpu) for gpu in settings['gpus']]
        self.launch = launch or IsaacLauncher(settings, self.service_dir)
        self.lock = threading.Lock()
        self.running = None
        (self.service_dir / 'programs').mkdir(parents=True, exist_ok=True)
        self.debug.mkdir(parents=True, exist_ok=True)
        self.set_aside = self.set_aside_interrupted()

    def set_aside_interrupted(self):
        """Move the output directories of requests a stopped service never recorded (it records a request when its
        last seed ends) to debug/interrupted/, so the same request can run again; returns what was moved."""
        recorded = {row['output'] for row in ledger_rows(self.ledger) if row['status'] == 'ran'}
        found = [self.debug / 'initial', self.debug / 'holdout', *(self.debug / 'attempts').glob('v*'),
                 *(self.debug / 'explore').glob('probe*')]
        moved = []
        for path in found:
            if path.is_dir() and f'debug/{path.relative_to(self.debug)}' not in recorded:
                target = self.debug / 'interrupted' / f'{path.relative_to(self.debug)}_{_now().replace(":", "")}'
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
                moved.append(str(target.relative_to(self.debug)))
        return moved

    def host(self, inside):
        """The service's path of `inside`, a path under /workspace/debug (or debug/ relative to /workspace)."""
        return self.debug / Path(inside).relative_to('debug')

    def _append(self, row):
        with open(self.ledger, 'a') as handle:
            handle.write(json.dumps(row) + '\n')

    def handle(self, request, emit):
        """Answer one request, calling `emit(dict)` with progress; returns the final event. Requests that run trials
        run one at a time."""
        if request.get('kind') in ('status', 'wait'):
            if request['kind'] == 'wait':
                with self.lock:
                    pass
            final = {'event': 'status', 'running': self.running, **usage(self.rules, ledger_rows(self.ledger))}
            emit(final)
            return final
        with self.lock:
            rows = ledger_rows(self.ledger)
            request_id = f'r{len(rows) + 1:04d}'
            try:
                plan, data = plan_request(request, self.workspace, self.rules, rows)
                if plan.kind != 'rerun' and self.host(plan.output).exists():
                    raise Refused(f'{plan.output} already exists')
            except Refused as exc:
                self._append({'time': _now(), 'id': request_id, 'request': request, 'kind': request.get('kind'),
                              'status': 'refused', 'reason': str(exc)})
                final = {'event': 'refused', 'reason': str(exc)}
                emit(final)
                return final
            self.running = {'label': plan.label, 'kind': plan.kind, 'seeds': list(plan.seeds), 'since': _now()}
            try:
                return self._run(request_id, request, plan, data, emit)
            finally:
                self.running = None

    def _run(self, request_id, request, plan, data, emit):
        program = self.service_dir / 'programs' / f'{plan.sha256}.py'
        if data is not None:
            program.write_bytes(data)
        if hashlib.sha256(program.read_bytes()).hexdigest() != plan.sha256:
            raise RuntimeError(f'the frozen copy of {plan.program} does not match its sha256')
        output, logs = self.host(plan.output), self.host(plan.logs)
        output.mkdir(parents=True, exist_ok=plan.kind == 'rerun')
        logs.mkdir(parents=True, exist_ok=True)
        emit({'event': 'accepted', 'kind': plan.kind, 'label': plan.label, 'program': plan.program,
              'seeds': list(plan.seeds), 'output': str(INSIDE_WORKSPACE / plan.output),
              'logs': str(INSIDE_WORKSPACE / plan.logs)})
        free = queue.Queue()
        for gpu in self.gpus:
            free.put(gpu)
        results, lock = output / 'results.jsonl', threading.Lock()
        if plan.kind != 'rerun':
            results.write_text('')

        def one(seed):
            gpu = free.get()
            log = logs / (f'{plan.label}.log' if plan.label.startswith('probe') else f'seed_{seed:03d}.log')
            emit({'event': 'started', 'seed': seed, 'gpu': gpu})
            try:
                outcome = self.launch(plan, seed, gpu, program, output, log, f'{request_id}_{seed}')
            except Exception as exc:  # a launcher failure is this seed's error, the other seeds still run
                outcome = TrialOutcome(seed, gpu, None, 0.0, 0.0, None, None, f'{type(exc).__name__}: {exc}')
            finally:
                free.put(gpu)
            row = self._result_row(outcome)
            with lock:  # one row per seed: a rerun replaces the seed's row
                kept = [line for line in ledger_rows(results) if line['seed'] != seed]
                results.write_text(''.join(json.dumps(line, sort_keys=True) + '\n' for line in [*kept, row]))
            emit({'event': 'finished', **row})
            return outcome, row

        with ThreadPoolExecutor(max_workers=len(self.gpus)) as pool:
            finished = list(pool.map(one, plan.seeds))
        self._append({'time': _now(), 'id': request_id, 'request': request, 'kind': plan.kind, 'status': 'ran',
                      **asdict(plan), 'trials': [asdict(outcome) for outcome, _ in finished]})
        rows = ledger_rows(results)
        final = {'event': 'done', 'label': plan.label, 'passed': sum(bool(row['task_success']) for row in rows),
                 'of': len(rows), 'errors': sum(row['error'] is not None for row in rows),
                 'rerunnable_seeds': missing_seeds(ledger_rows(self.ledger), plan.label),
                 'results': str(INSIDE_WORKSPACE / plan.output / 'results.jsonl'),
                 **usage(self.rules, ledger_rows(self.ledger))}
        emit(final)
        return final

    def _result_row(self, outcome):
        """The results.jsonl row of a trial, in the fields of upstream run-batch's, with workspace paths."""
        found = _outcome_json(outcome.run_dir) if outcome.run_dir else {}
        return {'seed': outcome.seed, 'ok': outcome.error is None and bool(found),
                'run_dir': None if outcome.run_dir is None else str(inside_path(self.settings, outcome.run_dir)),
                'program_ok': found.get('program_ok'), 'task_success': found.get('task_success'),
                'termination_reason': found.get('termination_reason'), 'wall_s': round(outcome.wall_s, 1),
                'max_rss_mb': round(outcome.max_rss_mb, 1), 'error': outcome.error}


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        def emit(event):
            try:
                self.wfile.write((json.dumps(event) + '\n').encode())
                self.wfile.flush()
            except OSError:  # the client went away; the request still finishes and is recorded
                pass

        try:
            request = json.loads(self.rfile.readline(65536) or b'{}')
            if not isinstance(request, dict):
                raise ValueError('a request is a JSON object')
        except ValueError as exc:
            emit({'event': 'refused', 'reason': f'unreadable request: {exc}'})
            return
        try:
            self.server.service.handle(request, emit)
        except Exception as exc:  # a harness failure: tell the agent, keep serving
            emit({'event': 'error', 'reason': f'{type(exc).__name__}: {exc}'})
            raise


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def start(socket_path, service):
    """Serve `service` on the Unix socket `socket_path` (mode 0600) in a daemon thread; returns the server, which
    `stop` ends."""
    socket_path = Path(socket_path)
    if len(os.fsencode(socket_path.absolute())) > SOCKET_PATH_MAX:
        raise ValueError(f'{socket_path} is longer than a Unix socket path may be ({SOCKET_PATH_MAX} bytes)')
    if socket_path.exists():
        socket_path.unlink()
    server = _Server(str(socket_path), _Handler)
    server.service = service
    os.chmod(socket_path, 0o600)
    threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True).start()
    return server


def stop(server):
    """End a server from `start` (a request in progress finishes in its own thread) and remove its socket."""
    server.shutdown()
    server.server_close()
    Path(server.server_address).unlink(missing_ok=True)


def serve(socket_path, service, *, ready=None):
    """Serve `service` on the Unix socket `socket_path` until interrupted."""
    server = start(socket_path, service)
    if ready is not None:
        ready.set()
    try:
        threading.Event().wait()
    finally:
        stop(server)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign', type=Path, required=True, help='campaign directory (campaign.json in it)')
    parser.add_argument('--socket', type=Path, required=True)
    args = parser.parse_args(argv)
    settings = json.loads((args.campaign / 'campaign.json').read_text())['trials']
    service = TrialService(args.campaign, settings)
    print(json.dumps({'serving': str(args.socket), **usage(service.rules, ledger_rows(service.ledger))}), flush=True)
    serve(args.socket, service)
    return 0


if __name__ == '__main__':
    sys.exit(main())
