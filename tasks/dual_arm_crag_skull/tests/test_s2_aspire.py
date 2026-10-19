"""ASPIRE-generated programs share the runtime without escaping the chosen execution mode."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('cap_harness')

from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG
from tasks.dual_arm_crag_skull.s2.aspire import AspirePlanner, build_planner, execute_program
from tasks.dual_arm_crag_skull.s2.planner import PlanningContext
from tasks.dual_arm_crag_skull.tests.s2_fakes import (
    COMPLETIONS, TIMEOUTS, FakeCandidateSource, expected_goal, observation,
)

PICK = """
o = get_observation()
result = submit_skill_request(o['observation_id'], 'pick', ['left'], fragment_id='part_1')
"""


def context(**kwargs):
    return PlanningContext(observation=observation(), trained_skills=('pick', 'place'),
                           completion_defaults=COMPLETIONS, timeouts_s=TIMEOUTS,
                           candidate_source=FakeCandidateSource(), **kwargs)


def test_program_resolves_pick_with_observed_pose_and_trace():
    planner = AspirePlanner(PICK)
    result = planner.plan(context())
    assert result.status == 'request'
    assert result.skill_request.skill_id == 'pick'
    assert result.skill_request.fragment_pose_provenance['pose_source'] == 'icp'
    assert result.trace[0]['execution_mode'] == 'skills'
    assert result.trace[0]['program_sha256'] == planner.sha256
    assert [c['tool'] for c in result.trace[0]['calls']] == ['get_observation', 'submit_skill_request']


def test_program_uses_screened_candidate_without_writing_target_matrix():
    code = """
o = get_observation()
candidates = propose_assembly_candidates(o['observation_id'], ['part_0', 'part_1'], 'part_0', [41])
result = submit_skill_request(o['observation_id'], 'place', ['left'], fragment_id='part_1',
                              target_candidate_id=candidates['candidates'][0]['candidate_id'])
"""
    result = AspirePlanner(code).plan(context())
    assert result.status == 'request'
    np.testing.assert_allclose(result.skill_request.target_pose, expected_goal('part_1'))
    rejected = AspirePlanner(code.replace('[41]', '[42]')).plan(context())
    assert rejected.status == 'error' and rejected.skill_request is None


def test_untrained_arm_combination_is_refused():
    result = AspirePlanner(PICK.replace("['left']", "['right']")).plan(
        context(trained_combinations=frozenset({('pick', ('left',)), ('place', ('left',))})))
    assert result.status == 'error' and result.skill_request is None


@pytest.mark.parametrize('suffix', [
    "\nraise ValueError('broken')",
    "\nget_observation()",
    "\ntry:\n    get_observation()\nexcept ValueError:\n    pass",
    "\ntry:\n    submit_skill_request(o['observation_id'], 'pick', ['left'], fragment_id='part_1')\nexcept ValueError:\n    pass",
])
def test_no_partial_or_multiple_submission_survives(suffix):
    result = AspirePlanner(PICK + suffix).plan(context())
    assert result.status == 'error' and result.skill_request is None


@pytest.mark.parametrize('code', ["move_to_pose({})", "step({})", "import os", "get_observation().__class__"])
def test_skill_program_cannot_call_motion_or_private_api(code):
    result = AspirePlanner(code).plan(context())
    assert result.status == 'error' and result.skill_request is None


def test_no_request_and_repeated_observations_are_explicit():
    assert AspirePlanner('result = get_observation()').plan(context()).status == 'no_request'
    planner = AspirePlanner(PICK)
    first = planner.plan(context())
    next_context = context()
    next_context.observation = observation('next')
    second = planner.plan(next_context)
    assert first.skill_request.observation_id == 'obs-1'
    assert second.skill_request.observation_id == 'next'
    assert first.skill_request.request_id != second.skill_request.request_id


def test_source_is_frozen_and_factory_defaults_to_aspire(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    path = tmp_path / 'program.py'
    path.write_text(PICK)
    planner = build_planner(config, program_path=path)
    path.write_text('raise ValueError("changed")')
    assert planner.plan(context()).status == 'request'
    with pytest.raises(ValueError, match='requires a frozen program'):
        build_planner(config)
    config['s2']['execution_mode'] = 'direct'
    with pytest.raises(ValueError, match='trial runner'):
        build_planner(config, program_path=path)


def test_mixed_api_modes_are_rejected():
    with pytest.raises(ValueError, match='forbids'):
        execute_program('result = 1', mode='skills', api=object(), tools={})
    with pytest.raises(ValueError, match='forbids'):
        execute_program('result = 1', mode='direct', api=object(), tools={})
    with pytest.raises(ValueError, match='unknown'):
        execute_program('result = 1', mode='hybrid')


def test_aspire_program_reaches_real_s0_through_existing_orchestrator():
    from tasks.dual_arm_crag_skull.s2.orchestrator import AssemblyOrchestrator
    from tasks.dual_arm_crag_skull.tests.test_layers_integration import Executor, Observer, PassingVerifier, StandInS1
    executor = Executor(StandInS1())
    report = AssemblyOrchestrator(observer=Observer(), planner=AspirePlanner(PICK), executor=executor,
                                  verifier=PassingVerifier(), trained_skills=('pick',),
                                  completion_defaults=COMPLETIONS, timeouts_s=TIMEOUTS,
                                  max_planning_s=60).run(1)
    assert report.steps[0]['execution']['status'] == 'completed'
    chunk, status, trajectory = executor.log[0]
    assert status.status == 'accepted'
    assert trajectory.request_id == chunk.request_id == report.steps[0]['skill_request']['request_id']


def test_skills_cli_only_plans(tmp_path, capsys):
    from tasks.dual_arm_crag_skull.s2.__main__ import main
    from tasks.dual_arm_crag_skull.tests.test_s2_preview import write_observation
    program = Path(__file__).parents[1] / 's2/programs/reobserve.py'
    assert main(['program', '--program', str(program), '--observation', str(write_observation(tmp_path))]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['motion_executed'] is False
    assert output['skill_request']['skill_id'] == 'reobserve'


def test_direct_cli_dispatches_same_seeded_runner(tmp_path, monkeypatch):
    from tasks.dual_arm_crag_skull.s2.__main__ import main
    from tasks.dual_arm_crag_skull.baselines.aspire import __main__ as baseline
    calls = []
    monkeypatch.setattr(baseline, 'main', lambda args: calls.append(args) or 0)
    result = main(['program', '--execution-mode', 'direct', '--program', 'program.py', '--task', 'task.json',
                  '--seed', '101', '--backend', 'kinematic', '--output-root', str(tmp_path), '--no-videos'])
    assert result == 0 and calls[0][0] == 'run'
    assert calls[0][calls[0].index('--mode') + 1] == 'practice'
