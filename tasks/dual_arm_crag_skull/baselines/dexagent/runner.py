"""One DexAgent trial: a fresh pinned clone, the exported inputs, Stage 3 under the external transport, the score.

    trial/                 one directory per trial, never reused
      DexAgent/            fresh clone of the pinned commit (stays unmodified; tool_library writes stay in it)
      home/ tmp/           HOME, TMPDIR and XDG_* of every DexAgent process
      inputs/              export.py output (Stage 1, Stage 2, stand-in video, export_manifest.json)
      dexagent.yaml        the run configuration (provider external, examples.reuse false, stage3 limits)
      transport/           DexAgent's external transport (requests, responses, images, operator checkpoint)
      runs/skull/stage3/   DexAgent's Stage 3 output (run.json, accepted.json, verified/<id>/)
      llm_ledger.jsonl     one line per API call (tokens, USD) and per stop; llm_transcript.jsonl
      stage3.log, result.json

run_stage3.py runs as a subprocess in its own process group with the scrubbed environment and is killed with
its group at the wall-clock deadline. run_stage3.py takes no --yes (only run_dexagent.py does) and asks nothing.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ... import workspace
from .. import sandbox as box
from ..budget import BudgetPool
from ..scoring import RunRecord, append_records, claim_heldout, refuse_repeats
from ..task import AssemblyTask, canonical_protocol, verify_task
from . import dexenv
from .export import (DEXAGENT_COMMIT, TASK_PROMPT, VARIANT, VARIANTS, ExportOptions, export_task, insertion_text,
                     load_manifest)
from .trajectory import accepted_trajectory, read_run, score_trajectory
from .transport import Responder, ledger_totals

RUN_NAME = 'skull'
BACKEND = 'dexagent_native_yam_sharpa_mujoco'
# A crx_yam trial's own record is written only when it ends without a delivery; a delivery runs on the shared hardware
# (execute_trial.py), whose record names that backend.
BACKENDS = {'native': BACKEND, 'crx_yam': 'dexagent_crx_yam_mujoco'}
SHARED_HARDWARE_NOTE = (
    'The robot is the hardware every method in this comparison uses: two FANUC CRX-10iA/L arms, each with an I2RT YAM '
    'parallel gripper, which the bench models as a hand whose two jaws are its thumb and index fingers (slide joints, '
    '0 = open; both jaws take one width). The stations never move. Every delivery is executed on that hardware '
    'through its safety layer S0, which refuses any command outside its limits and never slows one down: joint speed '
    'and acceleration, the arms apart, links and tools off the table, and near the table only inside a grasp contact '
    '(on the fragment) or a place contact (on its goal), slowly. The s0_delivery_check verifier, part of the gate '
    'suite, applies S0 to a delivery.')


def stage3_note(manifest):
    """stage3.notes (DexAgent's run-configuration hints for the agent): facts about the fabricated inputs (and, on the
    shared hardware, about the robot and S0)."""
    anchor = next(o for o in manifest['objects'] if o['object_id'] == manifest['anchor_object_id'])
    held = 'is fixed at' if anchor['free_joint'] is None else 'starts at'
    note = ('There is no human demonstration: the video is a stand-in rendering of the Stage 2 scene with the robot '
            'hidden, in which each skull fragment glides to its assembled pose during its Stage 1 time window; there '
            f'are no hands in it to track or retarget. The anchor, the {anchor["name"]}, {held} its assembled pose. '
            'Each fragment collides as the exact union of the boxes in its asset.')
    plan = manifest['task'].get('assembly_plan')
    if plan:
        names = {o['fragment_id']: o['name'] for o in manifest['objects']}
        rotation = np.asarray(manifest['rig_T_world'], dtype=float)[:3, :3]
        bench = 'crx_yam' if manifest.get('variant') == VARIANTS['crx_yam'] else 'native'
        steps = '; '.join(f'{k}. the {names[step["fragment_id"]]}, '
                          f'{insertion_text(rotation @ np.asarray(step["insertion_direction"], dtype=float), bench)}'
                          for k, step in enumerate(plan, 1))
        note += (f' The fragments go in in the order of the assembly plan, each by a last straight move: {steps}. '
                 'With the anchor and the fragments before it in place, that move does not collide.')
    return note + (' ' + SHARED_HARDWARE_NOTE if manifest.get('variant') == VARIANTS['crx_yam'] else '')


TRANSPORT_STATUS = {'budget': 'budget_exhausted', 'refusal': 'rejected'}
INPUT_GATE_FAILURES = ('Source video and Stage 1 provenance are inconsistent',
                       'Stage 2 source-reference mask provenance is unverified')


@dataclass(frozen=True)
class TrialConfig:
    """Everything one trial needs besides the task. `llm` is 'claude' (live: it spends from the shared budget pool in
    `budget_pool`, baselines/budget.py) or 'mock:<turns.json>'; `scoring_only` configures a finished trial for
    rescoring, which calls no model. DexAgent's processes run in the bwrap sandbox (baselines/sandbox.py); only a mock
    trial may turn it off."""

    trial_dir: Path
    llm: str = 'claude'
    budget_pool: Path | None = None
    scoring_only: bool = False
    sandbox: bool = True
    source: str = field(default_factory=dexenv.dexagent_source)
    python: str = field(default_factory=dexenv.dexagent_python)
    max_rounds: int = 100
    rounds_per_subgoal: int = 0
    max_rounds_cap: int = 300
    max_wall_minutes: float = 240.0
    kill_margin_minutes: float = 120.0
    methods: tuple = ()
    stage4_draws: int = 3
    strict_quality: bool = True
    workers: int = 4
    model: str = 'claude-opus-5-5'
    effort: str = 'high'
    max_tokens: int = 64000
    cache_ttl: str | None = '1h'
    notes: str | None = None
    export: ExportOptions = field(default_factory=ExportOptions)
    s0_limits: object = None  # s0.adapter.S0Limits: required on the shared-hardware bench (export.bench 'crx_yam')

    def __post_init__(self):
        object.__setattr__(self, 'trial_dir', Path(self.trial_dir).resolve())  # the stage runs with cwd = trial_dir
        if self.llm != 'claude' and not self.llm.startswith('mock:'):
            raise ValueError("llm must be 'claude' or 'mock:<turns.json>'")
        if self.llm == 'claude' and not self.scoring_only:
            if self.budget_pool is None:
                raise ValueError('a live trial spends from the shared budget pool: budget_pool is required')
            BudgetPool(self.budget_pool)  # FileNotFoundError when there is none
            if not self.sandbox:
                raise ValueError('a live trial runs DexAgent, and the code its agent writes, in the bwrap sandbox: '
                                 'sandbox cannot be off')
        if self.export.anchor_mode != 'fixed':
            raise ValueError("a trial holds the anchor by a fixture (baselines.anchor in jing.json); anchor_mode "
                             "'free' is for diagnostic exports only")
        if self.export.collider_inset_m != canonical_protocol()['collider_inset_m']:
            raise ValueError("a trial's fragment colliders are inset as the protocol says (baselines.collider_inset_m "
                             f"in jing.json), not {self.export.collider_inset_m}")
        if self.export.bench == 'crx_yam' and self.s0_limits is None:
            raise ValueError("the shared-hardware bench needs the S0 limits its deliveries will be executed under")
        if self.stage4_draws < 1 or not self.strict_quality:
            raise ValueError('stage4_draws >= 1 and strict_quality are required, or DexAgent cannot publish a result')
        for name in ('max_rounds', 'max_rounds_cap', 'workers'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be at least 1')
        if not (math.isfinite(self.max_wall_minutes) and math.isfinite(self.kill_margin_minutes)
                and self.max_wall_minutes > 0 and self.kill_margin_minutes >= 0):
            raise ValueError('max_wall_minutes must be finite and positive, kill_margin_minutes finite and non-negative')

    @property
    def transport_dir(self):
        return self.trial_dir / 'transport'

    @property
    def variant(self):
        return VARIANTS[self.export.bench]

    @property
    def backend(self):
        return BACKENDS[self.export.bench]

    @property
    def s0_spec(self):
        return self.trial_dir / 'inputs' / 's0_spec.json'

    @property
    def public(self):
        return self.trial_dir / 'runs' / RUN_NAME / 'stage3'


def trial_yaml(config, clone, inputs, notes):
    """The DexAgent run configuration (overlaid on the clone's dexagent.yaml by agent.config.load_config)."""
    return {
        'agent': {'provider': 'external', 'model': config.model, 'transport_dir': str(config.transport_dir),
                  'response_timeout_s': 7200, 'tool_discovery': 'search'},
        'run': {'name': RUN_NAME, 'output_dir': str(config.trial_dir / 'runs')},
        'examples': {'dir': str(Path(clone) / 'examples'), 'reuse': False, 'save_assets': False},
        'inputs': {'video': str(Path(inputs) / 'video.mp4'), 'task': TASK_PROMPT, 'setup': str(Path(clone) / 'setup'),
                   'library': str(Path(clone) / 'tool_library')},
        'stage3': {'max_rounds': config.max_rounds, 'rounds_per_subgoal': config.rounds_per_subgoal,
                   'max_rounds_cap': config.max_rounds_cap, 'notes': notes,
                   'max_wall_minutes': config.max_wall_minutes, 'stage4_draws': config.stage4_draws,
                   'strict_quality': config.strict_quality, 'methods': list(config.methods)},
        'compute': {'workers': config.workers},
    }


def stage3_command(config, clone, inputs):
    inputs = Path(inputs)
    command = [str(config.python), '-B', str(Path(clone) / 'stage3_trajectory_generation' / 'run_stage3.py'),
               str(config.trial_dir / 'dexagent.yaml'), '--semantic', str(inputs / 'stage1' / 'scene_semantics.json'),
               '--recon', str(inputs / 'stage2' / 'scene_recon.json'), '--video', str(inputs / 'video.mp4'),
               '--name', RUN_NAME]
    if config.llm.startswith('mock:'):
        command += ['--llm', f'mock:{Path(config.llm[5:]).resolve()}']
    return command


def sandbox_paths(config, extra_readonly=()):
    """What DexAgent's processes of a trial see in the sandbox: the trial directory writable; the DexAgent venv and its
    interpreter, this repository's code (the S0 delivery verifier imports it) and a mock trial's scripted turns (that
    file alone) read-only; `extra_readonly` for inputs kept outside the trial (tests). bwrap_prefix's keyword
    arguments, without cwd."""
    mock = [] if not config.llm.startswith('mock:') else [(Path(config.llm[5:]).resolve(),) * 2]
    return {'writable': [config.trial_dir],
            'readonly': [*box.python_paths(config.python), workspace.REPO_ROOT / 'core', workspace.REPO_ROOT / 'tasks',
                         *mock, *extra_readonly]}


def stage3_launch(config, clone, inputs, extra_readonly=()):
    """stage3_command in the sandbox (as it is without one only for a mock trial that turned it off)."""
    command = stage3_command(config, clone, inputs)
    if not config.sandbox:
        return command
    return [*box.bwrap_prefix(**sandbox_paths(config, extra_readonly), cwd=config.trial_dir), *command]


def kill_group(process, grace_s=60.0):
    """SIGTERM the process group, SIGKILL it after `grace_s`; returns the exit code."""
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 30.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            return process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    return process.wait()


def _reap_group(process):
    """Kill what is left of the process group (tool workers, run_python children) once the stage has exited."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_status(exit_code, run, transport_stop, timed_out, delivered):
    """RunRecord status: the transport's stop first, then our deadline, then DexAgent's own run record."""
    if transport_stop is not None:
        return TRANSPORT_STATUS.get(transport_stop['kind'], 'error')
    if timed_out:
        return 'timeout'
    if run is None or run.get('status') == 'running':
        return 'error'
    if run.get('finished') is True:
        return 'completed' if delivered and exit_code == 0 else 'error'
    reason, failure = run.get('stop_reason'), str(run.get('failure') or '')
    if failure in INPUT_GATE_FAILURES or failure.startswith('blocked: model backend failed'):
        return 'error'  # our inputs or our transport, not the agent's decision
    if reason == 'budget_cap' or failure.startswith('budget exhausted'):
        return 'budget_exhausted'
    if reason == 'wall_time_budget':
        return 'timeout'
    if reason in ('search_stalled', 'backend_no_tool_progress', 'cooperative_search_checkpoint', 'operator_checkpoint') \
            or failure.startswith('blocked') or run.get('upstream'):
        return 'blocked'
    return 'error'


def count_executions(public, run):
    """Simulated episodes in the Stage 3 trace: execute_subgoal_plan calls that ran (not dry runs, not suppressed
    duplicates, not failed calls) plus the variants search_subgoal_variants evaluated."""
    candidates = [Path(public) / 'trace.jsonl']
    if run and run.get('workspace'):
        candidates.append(Path(run['workspace']) / 'trace.jsonl')
    trace = next((p for p in candidates if p.is_file()), None)
    if trace is None:
        return None
    count = 0
    for line in trace.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('event') != 'tool':
            continue
        raw = event.get('result') if isinstance(event.get('result'), dict) else {}
        result = raw.get('result') if raw.get('ok') is True and isinstance(raw.get('result'), dict) else None
        if result is None:
            continue
        if event.get('name') == 'execute_subgoal_plan' and not result.get('dry_run') \
                and not (event.get('args') or {}).get('dry_run'):
            count += 1
        elif event.get('name') == 'search_subgoal_variants':
            count += int(result.get('evaluations') or 0)
    return count


def guard_seed(task, allow_heldout, records=None, variant=VARIANT):
    """A held-out seed runs only with `allow_heldout`, only into a records file, and once per variant."""
    if task.split != 'heldout':
        return
    if not allow_heldout:
        raise ValueError(f'seed {task.seed} is held out; pass allow_heldout only for the final evaluation')
    if records is None:
        raise ValueError('a held-out trial must be recorded: pass records')
    refuse_repeats(records, 'dexagent', variant, task.task_id, [task.seed])


def prepare_trial(task, config):
    """Fresh clone, scrubbed environment and exported inputs; returns (clone, env, manifest)."""
    trial = config.trial_dir
    if trial.exists() and any(trial.iterdir()):
        raise ValueError(f'{trial} is not empty; every trial gets a new directory')
    changed = dexenv.verify_venv(config.python)
    if changed:
        raise ValueError(f'the DexAgent venv differs from its package records: {changed[:5]}')
    trial.mkdir(parents=True, exist_ok=True)
    clone = trial / 'DexAgent'
    if config.export.bench == 'native':
        dexenv.fresh_clone(config.source, clone, DEXAGENT_COMMIT)
        env = dexenv.scrubbed_env(trial, config.python)
        manifest = export_task(task, trial / 'inputs', clone=clone, python=config.python, env=env, options=config.export)
        return clone, env, manifest
    from .bench.checkout import prepare_checkout
    from .s0_check import limits_record, write_spec
    from ...workspace import REPO_ROOT
    limits = limits_record(config.s0_limits)
    prepare_checkout(config.source, clone, limits)
    env = dexenv.scrubbed_env(trial, config.python)
    manifest = export_task(task, trial / 'inputs', clone=clone, python=config.python, env=env, options=config.export,
                           s0_limits=limits)
    env['DEXAGENT_S0_SPEC'] = str(write_spec(config.s0_spec, task=task, limits=config.s0_limits, manifest=manifest,
                                             repo=REPO_ROOT))
    return clone, env, manifest


def _test_only(config):
    """Plumbing: a mock model, or the shared hardware, whose S0 limits are TEST values until configs/jing.json has real
    ones (ASPIRE's records are test_only for the same reason)."""
    return config.llm.startswith('mock:') or config.export.bench == 'crx_yam'


def run_trial(task, config, *, client=None, records=None, allow_heldout=False):
    """Run one trial end to end and return its RunRecord (also written to trial/result.json and `records`).

    The task must be its layout under the canonical protocol (baselines.task.verify_task). A held-out seed needs
    `allow_heldout` and `records`, is claimed before anything is cloned, and runs once per variant; a trial that ends
    in an exception or an interrupt is recorded as status 'error' before the exception propagates."""
    verify_task(task)
    guard_seed(task, allow_heldout, records, config.variant)
    claim = None
    if task.split == 'heldout':
        claim = claim_heldout('dexagent', config.variant, task.task_id, task.seed, records_path=records)
    try:
        return _run_trial(task, config, client=client, records=records)
    except BaseException as exc:
        if records is not None:
            artifacts = {'trial': str(config.trial_dir), **({'heldout_claim': str(claim)} if claim else {})}
            append_records(records, [RunRecord('dexagent', config.variant, task.task_id, task.seed, task.split, 'error',
                                               None, notes=f'harness: {type(exc).__name__}: {str(exc)[:300]}',
                                               artifacts=artifacts, backend=config.backend,
                                               test_only=_test_only(config))])
        raise


def _run_trial(task, config, *, client, records):
    clone, env, manifest = prepare_trial(task, config)
    if not manifest['checks_ok']:
        record = RunRecord('dexagent', config.variant, task.task_id, task.seed, task.split, 'error', None,
                           notes=f'exported inputs failed DexAgent checks: {manifest["checks_failed"]}',
                           artifacts={'trial': str(config.trial_dir)}, backend=config.backend,
                           test_only=_test_only(config))
        return _finish(config, record, {'export': 'failed'}, records)
    # JSON is YAML: DexAgent's yaml.safe_load reads this file as written.
    notes = config.notes if config.notes is not None else stage3_note(manifest)
    (config.trial_dir / 'dexagent.yaml').write_text(json.dumps(
        trial_yaml(config, clone, config.trial_dir / 'inputs', notes), indent=1) + '\n')
    config.transport_dir.mkdir(parents=True, exist_ok=True)
    responder = None
    if config.llm == 'claude':
        dexenv.protect_process()  # the key in this process's environment is out of reach of DexAgent's processes
        responder = Responder(config.transport_dir, pool=BudgetPool(config.budget_pool),
                              who=f'dexagent:{config.trial_dir.name}', client=client, model=config.model,
                              effort=config.effort, max_tokens=config.max_tokens, cache_ttl=config.cache_ttl,
                              ledger=config.trial_dir / 'llm_ledger.jsonl',
                              transcript=config.trial_dir / 'llm_transcript.jsonl')
    command = stage3_launch(config, clone, config.trial_dir / 'inputs')
    deadline_s = 60.0 * (config.max_wall_minutes + config.kill_margin_minutes)
    started = time.monotonic()
    with (config.trial_dir / 'stage3.log').open('w') as log:
        process = subprocess.Popen(command, cwd=config.trial_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            if responder is not None:
                timed_out = responder.serve(process, deadline=started + deadline_s) == 'deadline'
            else:
                process.wait(timeout=deadline_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            exit_code = kill_group(process) if timed_out or process.poll() is None else process.returncode
            _reap_group(process)
    wall_s = round(time.monotonic() - started, 1)
    return score_trial(task, config, env=env, manifest=manifest, exit_code=exit_code, wall_s=wall_s,
                       timed_out=timed_out, transport=None if responder is None else responder.summary(),
                       records=records)


def verify_inputs(inputs, manifest):
    """Exported files whose sha256 no longer matches the export manifest (missing files included)."""
    from ..task import file_sha256
    inputs = Path(inputs)
    return sorted(name for name, digest in manifest['files_sha256'].items()
                  if not (inputs / name).is_file() or file_sha256(inputs / name) != digest)


def score_trial(task, config, *, env, manifest, exit_code, wall_s, timed_out, transport, records=None):
    """Status, score and cost of a finished Stage 3 run. A score needs DexAgent's acceptance, re-checked from a fresh
    clone against our unchanged exported Stage 2, with the DexAgent venv unchanged."""
    run = read_run(config.public)
    changed_inputs = verify_inputs(config.trial_dir / 'inputs', manifest)
    changed_venv = dexenv.verify_venv(config.python)
    if changed_inputs or changed_venv:
        directory, trajectory = None, (f'exported inputs changed during the trial: {changed_inputs[:5]}; '
                                       f'DexAgent venv files changed: {changed_venv[:5]}')
    else:
        verifier = config.trial_dir / f'verify_{uuid.uuid4().hex[:8]}' / 'DexAgent'
        if config.export.bench == 'native':
            dexenv.fresh_clone(config.source, verifier, DEXAGENT_COMMIT)
        else:
            from .bench.checkout import prepare_checkout
            from .s0_check import limits_record
            prepare_checkout(config.source, verifier, limits_record(config.s0_limits))
        directory, trajectory = accepted_trajectory(
            config.public, clone=verifier, python=config.python, env=env, log_path=verifier.parent / 'delivery.log',
            recon=config.trial_dir / 'inputs' / 'stage2' / 'scene_recon.json',
            sandbox=sandbox_paths(config) if config.sandbox else None)
    delivered = directory is not None
    # DexAgent accepts a delivery only after its own release_settle gate (settle 0.3 s, 4 mm / 10 deg by default).
    score = score_trajectory(task, manifest, trajectory, settled=True) if delivered else None
    shared = config.export.bench == 'crx_yam'
    stop = (transport or {}).get('stop')
    status = run_status(exit_code, run, stop, timed_out, delivered)
    if changed_inputs or changed_venv:
        status = 'error'
    cost, calls = ledger_totals(config.trial_dir / 'llm_ledger.jsonl') if config.llm == 'claude' else (None, None)
    notes = [f'DexAgent {DEXAGENT_COMMIT[:12]} Stage 3 on '
             + ('the CRX + YAM bench (port patch)' if shared else 'its native YAM + Sharpa bench')
             + f'; stand-in video, exported Stage 1/2 (anchor {config.export.anchor_mode})']
    if stop:
        notes.append(f'transport stop {stop["kind"]}: {stop["reason"][:300]}')
    if run and run.get('failure'):
        notes.append(f'DexAgent: {str(run["failure"])[:300]}')
    if not delivered:
        notes.append(f'no accepted delivery: {str(trajectory)[:300]}')
    if shared and delivered:
        # scored on the shared hardware (execute_trial.py); DexAgent's own MuJoCo result is only a note here
        notes.append(f'delivered; awaiting execution on the shared hardware (MuJoCo score: '
                     f'{"success" if score.success else "failure"})')
        score = None
    record = RunRecord('dexagent', config.variant, task.task_id, task.seed, task.split, status, score, cost_usd=cost,
                       llm_calls=calls, wall_s=wall_s, sim_trials=count_executions(config.public, run),
                       notes='; '.join(notes),
                       artifacts={'trial': str(config.trial_dir), 'export_manifest': str(config.trial_dir / 'inputs' /
                                                                                         'export_manifest.json'),
                                  'run_json': str(config.public / 'run.json'),
                                  'delivery': str(directory) if delivered else '',
                                  'ledger': (str(config.trial_dir / 'llm_ledger.jsonl') if config.llm == 'claude'
                                             else '')},
                       backend=config.backend, test_only=_test_only(config))
    details = {'exit_code': exit_code, 'timed_out': timed_out, 'transport': transport, 'run': run,
               'delivery': str(directory) if delivered else None, 'changed_inputs': changed_inputs,
               'changed_venv': changed_venv, 'awaiting_execution': bool(shared and delivered)}
    # a shared-hardware delivery is recorded once it has run there (execute_trial.py)
    return _finish(config, record, details, None if shared and delivered else records)


def _finish(config, record, details, records):
    (config.trial_dir / 'result.json').write_text(json.dumps({'record': record.to_record(), **details}, indent=1,
                                                             default=str) + '\n')
    if records is not None:
        append_records(records, [record])
    return record


def rescore(trial_dir, *, python=None, records=None):
    """Score a finished trial again from its files (no DexAgent run, no model call). The export manifest is read
    from disk here, so the input check is only as good as that file."""
    trial = Path(trial_dir)
    manifest = load_manifest(trial / 'inputs' / 'export_manifest.json')
    task = verify_task(AssemblyTask.from_record(manifest['task']))
    if records is not None and task.split == 'heldout':  # a second record of a held-out seed would count it twice
        refuse_repeats(records, 'dexagent', manifest.get('variant', VARIANT), task.task_id, [task.seed])
    previous = json.loads((trial / 'result.json').read_text()) if (trial / 'result.json').exists() else {}
    llm = 'claude' if (trial / 'llm_ledger.jsonl').exists() else 'mock:none'
    recorded = {'collider_inset_m': 0.0, **manifest['options']}  # exports from before the protocol's inset had none
    options = ExportOptions(**{k: tuple(v) if isinstance(v, list) else v for k, v in recorded.items()})
    limits = None
    if options.bench == 'crx_yam':  # the S0 limits the trial was prepared, and its bench generated, for
        from .s0_check import limits_from_record
        limits = limits_from_record(json.loads((trial / 'inputs' / 's0_spec.json').read_text())['limits'])
    config = TrialConfig(trial, llm=llm, scoring_only=True, python=python or dexenv.dexagent_python(), export=options,
                         s0_limits=limits)
    env = dexenv.scrubbed_env(trial, config.python)
    return score_trial(task, config, env=env, manifest=manifest, exit_code=previous.get('exit_code'),
                       wall_s=(previous.get('record') or {}).get('wall_s'), timed_out=bool(previous.get('timed_out')),
                       transport=previous.get('transport'), records=records)
