"""Run a frozen ASPIRE program on one seeded skull-assembly task and score it with baselines.scoring.

`run_program` does for this embodiment what cap_harness.run.run_program does for ASPIRE's own benchmarks (whose
benchmark list is hard-coded, so its CLI is not used): it builds the SkullAssemblyAdapter on an AssemblyBackend with
ASPIRE's RunRecorder as run observer, resets to the seed (the adapter's mode guards it), builds CapApi with the
"crx_dls" IK provider after the reset, executes the program through cap_harness.run.run_program_with_api and then
finalizes the run as run.py:235-443 does, so ASPIRE's own per-trial artifacts are written under `output_root`. The
trial is scored by the adapter's check_success (score_assembly on the backend's privileged poses, after the program)
and returned as a RunRecord: method 'aspire', variant = program sha256[:12], sim_trials 1, cost_usd None (a frozen
program makes no model calls; what writing it cost is not metered here).

RunRecord.status from ASPIRE's termination_reason: program_completed and task_succeeded -> 'completed', step_limit ->
'budget_exhausted', environment_truncated -> 'timeout', program_error -> 'error' (the program raised, or ASPIRE's
runtime refused it before it ran), environment_terminated -> 'error' (this adapter never terminates an episode).
Success is the score's, whatever the status. A harness failure (adapter, backend, recorder) and an interrupt are
re-raised after the run directory is closed, as in run.py: they are not method outcomes.

Held-out (evaluation-mode) trials are always appended to a records file, and a program variant is never run twice on
the same held-out seed of a task: the seed is claimed atomically (baselines.scoring.claim_heldout) before the backend
is touched, and a trial that ends in an exception or an interrupt is recorded as status 'error' before the exception
propagates. The program runs in this process without a sandbox (ASPIRE's runtime only validates its AST); run
untrusted programs on an isolated host without credentials.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cap_harness
from cap_harness.api import CapApi
from cap_harness.artifacts import RunRecorder, StepLimitReached, TracedProvider, redact
from cap_harness.contracts import MotionStrategy
from ...s2.aspire import run_direct_program as run_program_with_api

from ..backend import KinematicBackend
from ..scoring import RunRecord, append_records, claim_heldout, refuse_repeats
from ..task import verify_task
from .adapter import BENCHMARK, DEFAULT_CAMERA, SUITE, SkullAssemblyAdapter, guard_seed, load_task
from .checkout import verify_checkout
from .ik import IK_SOLVER, CrxIKProvider

STATUS_BY_TERMINATION = {'program_completed': 'completed', 'task_succeeded': 'completed',
                         'step_limit': 'budget_exhausted', 'environment_truncated': 'timeout',
                         'environment_terminated': 'error', 'program_error': 'error'}


def program_sha256(program_path):
    return hashlib.sha256(Path(program_path).read_bytes()).hexdigest()


def backend_label(backend):
    if isinstance(backend, KinematicBackend):
        return 'KinematicBackend (physics-free test double; its results are not baseline results)'
    return type(backend).__name__


def installed_harness():
    """The verified patched checkout cap_harness is imported from; raises if it is anything else."""
    return verify_checkout(Path(cap_harness.__file__).resolve().parents[2])


def _termination(recorder, execution, task_success):
    if recorder.terminal_reason == 'step_limit':
        return 'step_limit'
    if task_success:
        return 'task_succeeded'
    if recorder.terminated:
        return 'environment_terminated'
    if recorder.truncated:
        return 'environment_truncated'
    return 'program_completed' if execution.ok else 'program_error'


def _validated(program_path, task, seed, backend, *, config, mode, max_steps, records_path, expected_program_sha256):
    """Everything checked before a trial; a held-out seed is claimed last, once nothing else can refuse the trial."""
    split = guard_seed(seed, mode, config)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError('max_steps must be a positive integer')
    if not 0 < float(backend.control_hz) < float('inf'):
        raise ValueError('backend control_hz must be positive and finite')
    program_path, base = Path(program_path).resolve(), verify_task(load_task(task))
    digest = program_sha256(program_path)
    if expected_program_sha256 is not None and digest != expected_program_sha256:
        raise ValueError(f'{program_path} changed: sha256 {digest[:12]}, expected {expected_program_sha256[:12]}')
    if mode == 'evaluation' and records_path is None:
        raise ValueError('a held-out trial must be recorded: pass records_path')
    harness = installed_harness()
    claim = None
    if mode == 'evaluation':
        claim = claim_heldout('aspire', digest[:12], base.task_id, seed, records_path=records_path)
    return program_path, base, split, digest, harness, claim


def _execute(recorder, backend, *, model, limits, config, mode, base, seed, program_path):
    """Reset, run the program and close the run directory as run.py does; returns (adapter, termination_reason,
    program_ok, execution). Harness errors and interrupts are re-raised once the directory is closed."""
    adapter, execution, task_success, cleanup_errors = None, None, None, []

    def close():
        try:
            (backend if adapter is None else adapter).close()
        except Exception as exc:  # teardown errors are recorded without discarding the episode evidence
            cleanup_errors.append(f'adapter close: {type(exc).__name__}: {exc}')

    def success_or_none():
        try:
            return adapter.check_success()
        except Exception:  # no final state to score
            return None

    def finalize(termination_reason, program_ok, error):
        close()
        recorder.finalize(program_ok=program_ok, task_success=task_success, termination_reason=termination_reason,
                          program_result=execution, error=error, cleanup_errors=tuple(cleanup_errors))
        return adapter, termination_reason, program_ok, execution

    try:
        adapter = SkullAssemblyAdapter(backend, model, limits, config=config, mode=mode, run_observer=recorder)
        recorder.save_program(program_path)
        with recorder.timing.measure('reset'):
            observation = adapter.reset(base, seed)
        camera = DEFAULT_CAMERA if DEFAULT_CAMERA in observation.cameras else next(iter(observation.cameras))
        height, width = observation.cameras[camera].rgb.shape[:2]
        recorder.identity['capture'].update(camera_width=int(width), camera_height=int(height))
        api = CapApi(adapter, ik_providers={IK_SOLVER: TracedProvider(CrxIKProvider(model), recorder, IK_SOLVER)},
                     default_camera=camera, default_motion_strategy=MotionStrategy(ik_solver=IK_SOLVER),
                     recorder=recorder)
        execution = run_program_with_api(api=api, program_path=program_path, recorder=recorder)
        (recorder.root / 'logs/program.stdout').write_text(str(redact(execution.stdout)), encoding='utf-8')
        (recorder.root / 'logs/program.stderr').write_text('', encoding='utf-8')
        task_success = adapter.check_success()
        termination_reason = _termination(recorder, execution, task_success)
        return finalize(termination_reason, execution.ok and termination_reason != 'step_limit',
                        None if execution.ok else execution.error)
    except KeyboardInterrupt:
        task_success = success_or_none()
        finalize('user_interrupt', False, {'type': 'KeyboardInterrupt', 'message': 'execution interrupted'})
        raise
    except StepLimitReached as exc:
        task_success = success_or_none()
        return finalize('step_limit', False, {'type': type(exc).__name__, 'message': str(exc)})
    except Exception as exc:
        close()
        recorder.close_incomplete(exc, cleanup_errors=tuple(cleanup_errors))
        raise


def run_program(program_path, task, seed, *, config, backend, output_root, mode, max_steps, model, limits,
                records_path=None, expected_program_sha256=None, capture_videos=True, test_only=False):
    """One trial of the program at `program_path` on base task `task` (AssemblyTask or its JSON record) at `seed`.

    `model` and `limits` are the CRX model and the S0Limits every joint command is checked against; they are explicit
    because the S0 values in configs/jing.json are still null. The run owns `backend` and closes it, also when it
    refuses to run. The RunRecord is appended to `records_path` when given; evaluation mode requires it, claims the
    held-out seed first and refuses a seed this program variant already has. `expected_program_sha256` pins the
    program bytes; the copy ASPIRE saved and executed must match them too, or the trial is recorded as an error.
    `test_only` marks the record as never reportable (a run on S0 TEST values); the kinematic double always is. A
    backend that settles the fragments before the trial (`backend.start`) has that start saved in the run directory.
    """
    try:
        program_path, base, split, digest, harness, claim = _validated(
            program_path, task, seed, backend, config=config, mode=mode, max_steps=max_steps,
            records_path=records_path, expected_program_sha256=expected_program_sha256)
    except BaseException:
        backend.close()
        raise
    started, label = time.monotonic(), backend_label(backend)
    identity = dict(method='aspire', variant=digest[:12], task_id=base.task_id, seed=seed, split=split, backend=label,
                    test_only=bool(test_only) or isinstance(backend, KinematicBackend))
    artifacts = {'s2_planner': 'aspire_program', 'execution_mode': 'direct', 'program': str(program_path), 'program_sha256': digest, 'harness_checkout': str(harness.path),
                 'harness_patch_sha256': harness.patch_sha256, **({'heldout_claim': str(claim)} if claim else {})}
    executing = False
    try:
        recorder = RunRecorder(output_root=Path(output_root), benchmark=BENCHMARK, suite=SUITE, task_id=0,
                               task_name=base.task_id, seed=seed, max_steps=max_steps,
                               control_frequency_hz=float(backend.control_hz), capture_videos=capture_videos,
                               providers={'ik': (IK_SOLVER,), 'trajectory_planning': ('interpolation',),
                                          'pose_planning': ('composed',), 'segmentation': None, 'grasping': (),
                                          'backend': label, 'command_checks': 's0.adapter.check_joint_path'})
        artifacts['run_dir'] = str(recorder.root)
        executing = True
        adapter, termination_reason, program_ok, execution = _execute(
            recorder, backend, model=model, limits=limits, config=config, mode=mode, base=base, seed=seed,
            program_path=program_path)
        if program_sha256(recorder.root / 'source/program.py') != digest:
            raise ValueError(f'{program_path} changed during the trial; {recorder.root} is not a trial of '
                             f'{digest[:12]}')
        error = None if execution is None or execution.error is None else str(execution.error)[:300]
        start = getattr(backend, 'start', None)
        if start is not None:
            (recorder.root / 'settled_start.json').write_text(json.dumps(start.to_record(), indent=1) + '\n')
            artifacts['settled_start'] = str(recorder.root / 'settled_start.json')
        score = adapter.last_score
        notes = '; '.join([f'termination={termination_reason}', f'program_ok={program_ok}',
                           f'steps={recorder.step_count}/{max_steps}', f's0_rejections={adapter.rejected_commands}',
                           f'settled={None if score is None else score.settled}',
                           *([f'settle_error={adapter.settle_error}'] if adapter.settle_error else []),
                           f'harness={harness.commit[:12]}+{harness.patch.name}',
                           *([f'program_error={error}'] if error else [])])
        record = RunRecord(status=STATUS_BY_TERMINATION[termination_reason], score=score, cost_usd=None,
                           llm_calls=None, wall_s=time.monotonic() - started, sim_trials=1, notes=notes,
                           artifacts=artifacts, **identity)
    except BaseException as exc:
        if not executing:
            backend.close()  # _execute closes the backend once it has started
        if records_path is not None:
            append_records(records_path, [RunRecord(status='error', score=None, wall_s=time.monotonic() - started,
                                                    notes=f'harness: {type(exc).__name__}: {str(exc)[:300]}',
                                                    artifacts=artifacts, **identity)])
        raise
    if records_path is not None:
        append_records(records_path, [record])
    return record


def evaluate_frozen_program(program_path, seeds, *, task, config, backend_factory, output_root, records_path,
                            max_steps, model, limits, capture_videos=True, test_only=False):
    """Run the frozen program on held-out `seeds` in evaluation mode, one backend from `backend_factory` per seed (a
    factory may return the same backend every time: each trial resets it and closes it), appending each RunRecord to
    the JSONL `records_path` as it finishes.

    Every seed is guarded and checked against the existing records before the first trial, and every trial must run
    the bytes the program had when the evaluation started.
    """
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be a non-empty list of distinct integers')
    for seed in seeds:
        guard_seed(seed, 'evaluation', config)
    base, digest = load_task(task), program_sha256(program_path)
    refuse_repeats(records_path, 'aspire', digest[:12], base.task_id, seeds)
    return tuple(run_program(program_path, base, seed, config=config, backend=backend_factory(),
                             output_root=output_root, mode='evaluation', max_steps=max_steps, model=model,
                             limits=limits, records_path=records_path, expected_program_sha256=digest,
                             capture_videos=capture_videos, test_only=test_only) for seed in seeds)
