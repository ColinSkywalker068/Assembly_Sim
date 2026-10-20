"""A shared-hardware DexAgent trial after Stage 3: its delivery executed and recorded (baselines/dexagent/
execute_trial.py) on the kinematic double, its rescoring, and the CLI that prepares, runs and executes one."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG
from tasks.dual_arm_crag_skull.baselines.backend import KinematicBackend
from tasks.dual_arm_crag_skull.baselines.task import configured_initial_condition
from tasks.dual_arm_crag_skull.baselines.dexagent import __main__ as cli
from tasks.dual_arm_crag_skull.baselines.dexagent import execute_trial, runner, s0_check
from tasks.dual_arm_crag_skull.baselines.dexagent.export import EXPORT_SCHEMA, VARIANTS, ExportOptions
from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.tests.test_baselines_dexagent_executor import (HOME, LIMITS, MODEL, WIDTHS, backend,
                                                                               ramp, task)  # noqa: F401  (fixture)
from tasks.dual_arm_crag_skull.tests.test_baselines_dexagent_s0_check import manifest as objects_manifest
from tasks.dual_arm_crag_skull.tests.test_baselines_dexagent_s0_check import recording


def trial(tmp_path, task, left, labels):
    folder = tmp_path / 'trial'
    (folder / 'inputs').mkdir(parents=True)
    (folder / 'delivery').mkdir()
    manifest = {'schema': EXPORT_SCHEMA, 'variant': VARIANTS['crx_yam'], 'task': task.to_record(),
                **objects_manifest(task)}
    (folder / 'inputs' / 'export_manifest.json').write_text(json.dumps(manifest))
    s0_check.write_spec(folder / 'inputs' / 's0_spec.json', task=task, limits=LIMITS, manifest=manifest, repo=tmp_path)
    recording(folder / 'delivery' / 'trajectory.npz', left, labels, manifest['objects'])
    stage3 = {'method': 'dexagent', 'variant': VARIANTS['crx_yam'], 'task_id': task.task_id, 'seed': task.seed,
              'split': task.split, 'status': 'completed', 'score': None, 'cost_usd': 1.5, 'llm_calls': 7,
              'wall_s': 100.0, 'sim_trials': 3, 'notes': 'stage 3', 'artifacts': {'trial': str(folder)},
              'backend': 'dexagent_crx_yam_mujoco', 'test_only': True}
    (folder / 'result.json').write_text(json.dumps({'record': stage3, 'delivery': str(folder / 'delivery'),
                                                   'awaiting_execution': True}))
    return folder


def test_a_delivery_runs_on_the_backend_and_is_recorded_with_the_stage3_costs(task, tmp_path):
    left = np.vstack([ramp(HOME, HOME + 0.1, 1.0), ramp(HOME + 0.1, HOME, 1.0)[1:]])
    folder = trial(tmp_path, task, left, ['left: wait'] * len(left))
    records = tmp_path / 'runs.jsonl'
    record = execute_trial.execute_trial(folder, backend(), MODEL, test_only=True,
                                         records=records)
    assert record.status == 'completed' and record.score is not None and not record.score.success
    assert (record.test_only, record.cost_usd, record.llm_calls) == (True, 1.5, 7)
    assert record.backend == execute_trial.backend_label(backend()) and record.backend.startswith('KinematicBackend')
    assert record.variant == 'crx_yam_stage3' and f'executed on {record.backend} through S0: completed' in record.notes
    written = json.loads((folder / 'execution.json').read_text())
    assert written['steps'] == 120 and written['settled'] is True
    for key in task.fragment_ids:  # where the fragments ended, for comparing backends (here: never moved)
        np.testing.assert_allclose(written['final_poses'][key], task.fragment(key).initial_world_T_fragment)
    lines = [json.loads(line) for line in records.read_text().splitlines()]
    assert lines == [json.loads(json.dumps(record.to_record()))]


def test_a_delivery_s0_refuses_is_recorded_as_rejected(task, tmp_path):
    left = ramp(HOME, HOME + 1.5, 0.5)
    folder = trial(tmp_path, task, left, ['left: wait'] * len(left))
    record = execute_trial.execute_trial(folder, backend(), MODEL, test_only=True)
    assert record.status == 'rejected' and record.score is not None and 'S0 rejected' in record.notes


def test_a_protective_stop_is_recorded_as_one_with_its_log(task, tmp_path):
    from tasks.dual_arm_crag_skull.tests.test_baselines_dexagent_executor import Stopping
    left = np.vstack([ramp(HOME, HOME + 0.1, 1.0), ramp(HOME + 0.1, HOME, 1.0)[1:]])
    folder = trial(tmp_path, task, left, ['left: wait'] * len(left))
    records = tmp_path / 'runs.jsonl'
    record = execute_trial.execute_trial(folder, Stopping(at=30), MODEL, test_only=True, records=records)
    assert record.status == 'protective_stop' and record.score is None and not record.success
    assert 'protective stop: left joint_2 lagged its target' in record.notes
    written = json.loads((folder / 'execution.json').read_text())
    assert written['status'] == 'protective_stop' and written['steps'] == 29
    assert written['protective_stop_log'] == [{'time_s': pytest.approx(29 / 60)}]
    assert [json.loads(line)['status'] for line in records.read_text().splitlines()] == ['protective_stop']


def test_only_a_trial_awaiting_execution_runs(task, tmp_path):
    folder = trial(tmp_path, task, np.repeat(HOME[None], 3, axis=0), ['left: wait'] * 3)
    result = json.loads((folder / 'result.json').read_text())
    (folder / 'result.json').write_text(json.dumps({**result, 'awaiting_execution': False}))
    with pytest.raises(ValueError, match='awaiting execution'):
        execute_trial.execute_trial(folder, backend(), MODEL, test_only=True)


class SettlingDouble(KinematicBackend):
    """The kinematic double with a settled start, as Isaac's backend records one on reset (sim/backend.SettledStart):
    the task's initial poses, the fragment `shifted` moved by `offset_m` along x."""

    def __init__(self, shifted=None, offset_m=0.0):
        super().__init__(MODEL, home_joints={arm: HOME for arm in ARMS}, width_range_m=WIDTHS, control_hz=60)
        self.shifted, self.offset_m = shifted, offset_m

    def reset(self, task):
        snapshot = super().reset(task)
        poses = {key: task.fragment(key).initial_world_T_fragment.copy() for key in task.fragment_ids}
        if self.shifted is not None:
            poses[self.shifted][0, 3] += self.offset_m
        self.start = SimpleNamespace(poses=poses, to_record=lambda: {'settle_s': 1.25, 'at_rest': True})
        return snapshot


def test_a_backends_settled_start_is_saved_and_compared_with_the_planned_one(task, tmp_path):
    folder = trial(tmp_path, task, np.repeat(HOME[None], 3, axis=0), ['left: wait'] * 3)
    record = execute_trial.execute_trial(folder, SettlingDouble('frag_b', 0.0005), MODEL, test_only=True)
    assert json.loads((folder / 'settled_start.json').read_text()) == {'settle_s': 1.25, 'at_rest': True}
    assert record.artifacts['settled_start'] == str(folder / 'settled_start.json')  # as ASPIRE's run directories
    gap = json.loads((folder / 'execution.json').read_text())['start_gap']
    assert gap['frag_b'] == pytest.approx([0.0005, 0.0], abs=1e-9) and gap['frag_a'] == pytest.approx([0.0, 0.0])
    assert record.status == 'completed'


def test_a_start_other_than_the_planned_one_is_not_executed(task, tmp_path):
    folder = trial(tmp_path, task, np.repeat(HOME[None], 3, axis=0), ['left: wait'] * 3)
    record = execute_trial.execute_trial(folder, SettlingDouble('frag_b', 0.002), MODEL, test_only=True)
    assert record.status == 'error' and 'frag_b' in record.notes and 'planned from' in record.notes
    assert json.loads((folder / 'execution.json').read_text())['steps'] == 0


def test_a_trial_is_executed_once(task, tmp_path):
    folder = trial(tmp_path, task, np.repeat(HOME[None], 3, axis=0), ['left: wait'] * 3)
    execute_trial.execute_trial(folder, backend(), MODEL, test_only=True)
    with pytest.raises(ValueError, match='already executed'):  # a second record of the same delivery
        execute_trial.execute_trial(folder, backend(), MODEL, test_only=True)


def test_a_backend_must_match_the_rig_the_trial_was_prepared_for(task):
    execute_trial.check_rig(LIMITS, WIDTHS, 60)
    with pytest.raises(ValueError, match='gripper width range'):
        execute_trial.check_rig(LIMITS, {arm: (0.0, 0.05) for arm in ARMS}, 60)
    with pytest.raises(ValueError, match='servo rate'):
        execute_trial.check_rig(LIMITS, WIDTHS, 120)


def test_a_trial_whose_s0_limits_are_for_another_table_is_not_executed(task, tmp_path):
    folder = trial(tmp_path, task, np.repeat(HOME[None], 3, axis=0), ['left: wait'] * 3)
    path = folder / 'inputs' / 's0_spec.json'
    spec = json.loads(path.read_text())
    spec['limits']['table_top_z_m'] = 0.8
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='table'):
        execute_trial.execute_trial(folder, backend(), MODEL, test_only=True)


def test_rescore_keeps_the_s0_limits_a_shared_hardware_trial_was_prepared_for(task, tmp_path, monkeypatch):
    folder = trial(tmp_path, task, np.repeat(HOME[None], 3, axis=0), ['left: wait'] * 3)
    path = folder / 'inputs' / 'export_manifest.json'
    path.write_text(json.dumps({**json.loads(path.read_text()), 'options': ExportOptions(bench='crx_yam').to_record()}))
    seen = {}
    monkeypatch.setattr(runner, 'score_trial', lambda task, config, **_: seen.setdefault('config', config))
    runner.rescore(folder, python=sys.executable)
    config = seen['config']
    assert config.export.bench == 'crx_yam' and config.variant == VARIANTS['crx_yam']
    assert s0_check.limits_record(config.s0_limits) == s0_check.limits_record(LIMITS)
    assert runner._test_only(config)  # S0's TEST values: no shared-hardware result counts yet


def test_cli_executes_a_trial_on_the_kinematic_double(task, tmp_path, capsys):
    left = np.vstack([ramp(HOME, HOME + 0.1, 1.0), ramp(HOME + 0.1, HOME, 1.0)[1:]])
    folder = trial(tmp_path, task, left, ['left: wait'] * len(left))
    records = tmp_path / 'runs.jsonl'
    assert cli.main(['execute', '--trial', str(folder), '--backend', 'kinematic', '--records', str(records)]) == 0
    printed = json.loads(capsys.readouterr().out)
    line, = [json.loads(text) for text in records.read_text().splitlines()]
    assert printed == line and line['status'] == 'completed' and line['test_only'] is True
    assert line['backend'].startswith('KinematicBackend (physics-free test double')


def test_cli_prepares_a_shared_hardware_trial_under_the_test_s0_limits(task, tmp_path, monkeypatch):
    (tmp_path / 'task.json').write_text(json.dumps(task.to_record()))
    seen = {}

    def run_trial(task, config, **kwargs):
        seen['config'] = config
        return SimpleNamespace(to_record=dict)

    monkeypatch.setattr(cli, 'run_trial', run_trial)
    assert cli.main(['run', '--task', str(tmp_path / 'task.json'), '--trial', str(tmp_path / 'trial'),
                     '--llm', f'mock:{tmp_path / "turns.json"}', '--bench', 'crx_yam']) == 0
    config = seen['config']
    assert config.export.bench == 'crx_yam' and runner._test_only(config)
    assert s0_check.limits_record(config.s0_limits) == s0_check.limits_record(LIMITS)  # ASPIRE's TEST values


def test_cli_exports_on_the_shared_hardware_from_the_port_checkout(task, tmp_path, monkeypatch):
    (tmp_path / 'task.json').write_text(json.dumps(task.to_record()))
    calls = {}
    monkeypatch.setattr(cli, 'prepare_checkout', lambda source, clone, limits: calls.update(checkout=(clone, limits)))
    monkeypatch.setattr(cli.dexenv, 'fresh_clone', lambda *args: calls.update(fresh_clone=args))

    def export_task(task, out, *, clone, python, env, options, s0_limits=None):
        calls['export'] = (clone, options, s0_limits)
        return {'checks_ok': True, 'checks_failed': [], 'order': [], 'anchor_object_id': task.anchor_id}

    monkeypatch.setattr(cli, 'export_task', export_task)
    assert cli.main(['export', '--task', str(tmp_path / 'task.json'), '--out', str(tmp_path / 'out'),
                     '--bench', 'crx_yam']) == 0
    clone, options, limits = calls['export']
    assert 'fresh_clone' not in calls and calls['checkout'] == (clone, limits)
    assert options.bench == 'crx_yam' and limits == s0_check.limits_record(LIMITS)


def test_cli_export_resolves_a_relative_output_directory(task, tmp_path, monkeypatch):
    (tmp_path / 'task.json').write_text(json.dumps(task.to_record()))
    calls = {}
    monkeypatch.setattr(cli.dexenv, 'fresh_clone', lambda source, clone, commit: calls.update(clone=clone))

    def export_task(task, out, *, clone, python, env, options, s0_limits=None):
        calls['out'] = out
        return {'checks_ok': True, 'checks_failed': [], 'order': [], 'anchor_object_id': task.anchor_id}

    monkeypatch.setattr(cli, 'export_task', export_task)
    monkeypatch.chdir(tmp_path)
    assert cli.main(['export', '--task', 'task.json', '--out', 'out']) == 0  # DexAgent's workers run elsewhere
    assert calls['out'] == tmp_path.resolve() / 'out' / 'inputs' and calls['clone'] == tmp_path.resolve() / 'out' / 'DexAgent'


class Failing(KinematicBackend):
    """The kinematic double failing at its `fail_at`-th step, as a simulator can."""

    def __init__(self, fail_at):
        super().__init__(MODEL, home_joints={arm: HOME for arm in ARMS}, width_range_m=WIDTHS, control_hz=60)
        self.fail_at, self.count = fail_at, 0

    def step(self, joints, widths):
        self.count += 1
        if self.count == self.fail_at:
            raise RuntimeError('simulated failure')
        return super().step(joints, widths)


def test_an_execution_that_fails_midway_is_recorded_and_never_run_again(task, tmp_path):
    left = np.vstack([ramp(HOME, HOME + 0.1, 1.0), ramp(HOME + 0.1, HOME, 1.0)[1:]])
    folder = trial(tmp_path, task, left, ['left: wait'] * len(left))
    records = tmp_path / 'runs.jsonl'
    with pytest.raises(RuntimeError, match='simulated failure'):
        execute_trial.execute_trial(folder, Failing(50), MODEL, test_only=True, records=records)
    line, = [json.loads(text) for text in records.read_text().splitlines()]
    assert line['status'] == 'error' and 'RuntimeError' in line['notes'] and line['score'] is None
    with pytest.raises(ValueError, match='already'):  # the delivery has moved a robot once
        execute_trial.execute_trial(folder, backend(), MODEL, test_only=True, records=records)
    assert len(records.read_text().splitlines()) == 1


def test_a_heldout_delivery_is_executed_only_into_a_records_file(task, tmp_path):
    heldout = configured_initial_condition(task, 7, json.loads(DEFAULT_CONFIG.read_text()))
    assert heldout.split == 'heldout'
    folder = trial(tmp_path, heldout, np.repeat(HOME[None], 3, axis=0), ['left: wait'] * 3)
    with pytest.raises(ValueError, match='must be recorded'):
        execute_trial.execute_trial(folder, backend(), MODEL, test_only=True)
    assert not (folder / 'execution.json').exists() and not (folder / 'execution.started').exists()
