"""Live S2 planner check against the Claude API on recorded skull data. Nothing here commands motion.

Poses are in the registered GARF part frames written by export_points; GARF calls are replayed from
recorded results (the GPU model is not run). Scenarios:
  garf_all_rejected   real GARF candidates for the 8-piece skull (cached encoder, seeds 41-44) with real
                      geometry screening; anchor at its assembled pose, other fragments staged.
  oracle_eligible     same scene, the left gripper holds a fragment that touches the anchor, and a
                      ground-truth-pose candidate (seed 0, labelled oracle, not a GARF output) is offered.
  vision_unlocalized  three keyframes from the Isaac smoke run, no fragment localized.
Fragment poses come from the simulator, so every observation is flagged `oracle_sim`.
"""
import argparse
import copy
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from ..__main__ import DEFAULT_CONFIG, validate
from ..s2.claude import ClaudePlanner
from ..s2.observation import FragmentState, Observation
from ..s2.planner import PlanningContext

PRICE_PER_MTOK = {'input': 4.0, 'output': 20.0}  # claude-opus-5-5, USD


class ReplaySource:
    """Answers propose_assembly_candidates from one recorded, screened result."""

    def __init__(self, screened, seeds, anchor_id, fragment_ids):
        self.screened, self.seeds = screened, tuple(seeds)
        self.anchor_id, self.fragment_ids = anchor_id, tuple(sorted(fragment_ids))
        self.calls = []

    def propose(self, request, *, current_observation_id):
        self.calls.append(dict(request))
        if request.get('observation_id') != current_observation_id:
            return {'status': 'rejected', 'reason': 'stale observation'}
        if tuple(sorted(request.get('fragment_ids', ()))) != self.fragment_ids or request.get('anchor_id') != self.anchor_id:
            return {'status': 'failed', 'reason': (f'replay mode: only the full set of {len(self.fragment_ids)} fragments '
                                                   f'with anchor {self.anchor_id} was recorded')}
        seeds = [seed for seed in request.get('seeds', []) if seed in self.seeds]
        if not seeds:
            return {'status': 'failed', 'reason': f'replay mode: recorded seeds are {list(self.seeds)}'}
        result = copy.deepcopy(self.screened)
        result['candidates'] = [dict(c, observation_id=current_observation_id) for c in result['candidates']
                                if c['seed'] in seeds]
        result['geometry_screening'] = [r for r in result['geometry_screening'] if r['seed'] in seeds]
        result['geometrically_eligible_seeds'] = [s for s in result['geometrically_eligible_seeds'] if s in seeds]
        result['geometry_status'] = 'needs_motion_validation' if result['geometrically_eligible_seeds'] else 'all_candidates_rejected'
        return result


def _pose(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return pose


def scene_poses(truth, anchor_id, held_id=None):
    """Assembled skull centred above the pad; anchor at its assembled pose, others staged on the table."""
    centre = np.mean([np.asarray(t)[:3, 3] for t in truth.values()], axis=0)
    world_T_assembly = _pose(np.eye(3), np.array([0.0, 0.0, 0.86]) - centre)
    poses, staged = {}, [key for key in sorted(truth) if key not in (anchor_id, held_id)]
    poses[anchor_id] = world_T_assembly @ np.asarray(truth[anchor_id])
    for index, key in enumerate(staged):
        yaw = Rotation.from_euler('z', 40 * index, degrees=True).as_matrix()
        side, column = (1 if index % 2 else -1), index // 2
        rotation = yaw @ np.asarray(truth[key])[:3, :3]
        poses[key] = _pose(rotation, [-0.25 + 0.17 * column, 0.32 * side, 0.80])
    if held_id is not None:
        poses[held_id] = _pose(np.asarray(truth[held_id])[:3, :3], [-0.30, 0.12, 1.02])
    return poses


def downscaled_png(path, longest=1024):
    from PIL import Image
    image = Image.open(path).convert('RGB')
    image.thumbnail((longest, longest))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def scenarios(geometry, screened, smoke_dir):
    truth = json.loads((geometry / 'evaluation_gt.json').read_text())
    anchor = json.loads((geometry / 'request.jsonl').read_text().splitlines()[0])['request']['anchor_id']
    oracle = next(r for r in screened['geometry_screening'] if r['seed'] == 0)
    contact = {}
    for pair in oracle['pairs']:
        if anchor in pair['pair']:
            other = pair['pair'][0] if pair['pair'][1] == anchor else pair['pair'][1]
            contact[other] = pair['contact_fraction']
    held = max(contact, key=contact.get)
    fragments = sorted(truth)

    def states(poses):
        return {key: FragmentState(key, poses.get(key), 'oracle_sim' if key in poses else None) for key in fragments}

    def replay_note(seeds):
        return {'event': 'garf_replay', 'note': 'GARF answers are replayed from recorded runs for this check',
                'recorded_fragment_set': 'all fragments', 'recorded_anchor_id': anchor, 'recorded_seeds': list(seeds)}

    real_seeds = [c['seed'] for c in screened['candidates'] if c['seed'] != 0]
    yield {'name': 'garf_all_rejected', 'acceptable_skills': ['pick', 'reobserve', 'retreat'],
           'observation': Observation('live-garf-1', 10.0, states(scene_poses(truth, anchor)),
                                      robot_summary={'left': {'holding': None}, 'right': {'holding': None}}),
           'source': ReplaySource(screened, real_seeds, anchor, fragments), 'feedback': (replay_note(real_seeds),),
           'expected': {'not_goal_directed': True}}
    yield {'name': 'oracle_eligible', 'acceptable_skills': ['place'],
           'observation': Observation('live-oracle-1', 20.0, states(scene_poses(truth, anchor, held)),
                                      robot_summary={'left': {'holding': held, 'gripper_width_m': 0.03},
                                                     'right': {'holding': None}}),
           'source': ReplaySource(screened, [0, *real_seeds], anchor, fragments),
           'feedback': (replay_note([0, *real_seeds]),
                        {'event': 'skill', 'skill_id': 'pick', 'fragment_id': held, 'execution': 'completed',
                         'verified': True, 'verification_reason': 'grasp held after lift'}),
           'expected': {'skill_id': 'place', 'fragment_id': held, 'active_arms': ['left'], 'candidate_seed': 0}}
    keyframes = {name: downscaled_png(smoke_dir / f'{name}.png') for name in ('agent', 'left_wrist', 'right_wrist')}
    yield {'name': 'vision_unlocalized', 'acceptable_skills': ['reobserve'],
           'observation': Observation('live-vision-1', 30.0, states({}), keyframes_png=keyframes,
                                      robot_summary={'left': {'holding': None}, 'right': {'holding': None}}),
           'source': None, 'feedback': (), 'expected': {'skill_id': 'reobserve'}}


def run_scenario(scenario, planner, planner_config):
    context = PlanningContext(observation=scenario['observation'], trained_skills=('pick', 'place'),
                              candidate_source=scenario['source'], completion_defaults=planner_config['completion_defaults'],
                              timeouts_s=planner_config['timeouts_s'], feedback=scenario['feedback'])
    start = time.monotonic()
    outcome = planner.plan(context)
    elapsed = time.monotonic() - start
    request = None if outcome.skill_request is None else outcome.skill_request.to_record()
    acceptable = outcome.status == 'request' and request['skill_id'] in scenario['acceptable_skills']
    expected = scenario['expected']
    if acceptable and 'candidate_seed' in expected:
        acceptable = (request['fragment_id'] == expected['fragment_id'] and request['active_arms'] == expected['active_arms']
                      and request['target_provenance']['seed'] == expected['candidate_seed'])
    usage = [t['usage'] for t in outcome.trace if t.get('usage')]
    tokens = {key: sum((u.get(key) or 0) for u in usage) for key in ('input_tokens', 'output_tokens', 'cache_read_input_tokens')}
    return {'scenario': scenario['name'], 'acceptable_skills': scenario['acceptable_skills'], 'expected': expected,
            'outcome_status': outcome.status, 'outcome_reason': outcome.reason, 'retryable': outcome.retryable,
            'acceptable': acceptable, 'skill_request': request, 'trace': list(outcome.trace),
            'replayed_garf_calls': [] if scenario['source'] is None else scenario['source'].calls,
            'wall_s': elapsed, 'tokens': tokens,
            'cost_usd': (tokens['input_tokens'] * PRICE_PER_MTOK['input'] + tokens['output_tokens'] * PRICE_PER_MTOK['output']) / 1e6}


class DryRunClient:
    """Offline stand-in: submits reobserve immediately, to exercise the harness without the API."""

    def __init__(self):
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        observation_id = json.loads(kwargs['messages'][0]['content'][-1]['text'].split('\n', 1)[1])['observation']['observation_id']
        block = SimpleNamespace(type='tool_use', id='dry-1', name='submit_skill_request', input={
            'observation_id': observation_id, 'skill_id': 'reobserve', 'active_arms': [], 'fragment_id': None,
            'support_fragment_id': None, 'target_candidate_id': None, 'rationale': 'dry run'})
        return SimpleNamespace(content=[block], stop_reason='tool_use', model='dry-run', _request_id=None,
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True, help='export_points output directory')
    parser.add_argument('--screened', type=Path, required=True, help='screened candidates JSON incl. oracle seed 0')
    parser.add_argument('--smoke', type=Path, required=True, help='smoke run directory with the three PNG keyframes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='no API call; exercise the harness only')
    args = parser.parse_args(argv)
    config = json.loads(DEFAULT_CONFIG.read_text())
    validate(config)
    planner_config = config['s2']
    planner = ClaudePlanner(DryRunClient() if args.dry_run else None, model=planner_config['api_model_id'],
                            effort=planner_config['effort'], max_turns=planner_config['max_turns'],
                            server_fallbacks=planner_config['server_fallbacks'] == 'default')
    screened = json.loads(args.screened.read_text())
    results = [run_scenario(s, planner, planner_config) for s in scenarios(args.geometry, screened, args.smoke)]
    report = {'date': time.strftime('%Y-%m-%d'), 'dry_run': args.dry_run, 'model': planner_config['api_model_id'],
              'effort': planner_config['effort'], 'server_fallbacks': planner_config['server_fallbacks'],
              'screening': {'eligible_seeds': screened['geometrically_eligible_seeds'],
                            'per_seed': [{k: r[k] for k in ('seed', 'rejected', 'max_penetration_fraction', 'contact_components')}
                                         for r in screened['geometry_screening']]},
              'scenarios': results, 'all_acceptable': all(r['acceptable'] for r in results),
              'total_cost_usd': sum(r['cost_usd'] for r in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=str) + '\n')
    for result in results:
        request = result['skill_request'] or {}
        print(result['scenario'], result['outcome_status'], request.get('skill_id'), request.get('fragment_id'),
              'acceptable' if result['acceptable'] else 'NOT acceptable', f"${result['cost_usd']:.3f}", flush=True)
    return 0 if report['all_acceptable'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
