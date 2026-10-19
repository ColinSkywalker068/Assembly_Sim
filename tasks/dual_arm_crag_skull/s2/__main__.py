"""Shared ASPIRE S2: `program` selects skills planning (default) or direct seeded trials.

Skills mode prints a plan without executing motion; direct uses the ASPIRE baseline runner.
`preview` prints the legacy Claude tool client's first request and sends nothing.

Observation JSON: {"observation_id": str, "time": float, "fragments": {id: {"world_T_part": 4x4 | null,
"pose_source": str | null, "inlier_rmse_m": float | null}}, "verified_clusters": [[id, ...]],
"keyframes_png": {camera: path}, "robot_summary": {...}}
"""
import argparse
import json
from pathlib import Path

from ..__main__ import DEFAULT_CONFIG, validate
from .claude import ClaudePlanner, initial_message
from .observation import FragmentState, Observation
from .planner import PlanningContext


def load_observation(path):
    data = json.loads(Path(path).read_text())
    fragments = {key: FragmentState(key, value.get('world_T_part'), value.get('pose_source'),
                                    value.get('inlier_rmse_m'))
                 for key, value in data['fragments'].items()}
    keyframes = {camera: Path(png).read_bytes() for camera, png in data.get('keyframes_png', {}).items()}
    return Observation(data['observation_id'], float(data['time']), fragments,
                       verified_clusters=tuple(frozenset(c) for c in data.get('verified_clusters', [])),
                       keyframes_png=keyframes, robot_summary=data.get('robot_summary', {}))


class _Unconnected:
    def propose(self, request, *, current_observation_id):
        return {'status': 'failed', 'reason': 'preview only: GARF runner not connected'}


def _elide_images(node):
    if isinstance(node, dict):
        if node.get('type') == 'base64':
            return {**node, 'data': f'<{len(node["data"])} base64 chars>'}
        return {key: _elide_images(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_elide_images(value) for value in node]
    return node


def preview(observation, config, trained_skills, with_garf):
    planner_config = config['s2']
    context = PlanningContext(observation=observation, trained_skills=tuple(trained_skills),
                              candidate_source=_Unconnected() if with_garf else None,
                              completion_defaults=planner_config['completion_defaults'],
                              timeouts_s=planner_config['timeouts_s'])
    planner = ClaudePlanner(client=object(), model=planner_config['api_model_id'], effort=planner_config['effort'],
                            max_turns=planner_config['max_turns'],
                            server_fallbacks=planner_config['server_fallbacks'] == 'default')
    return _elide_images(planner.request_kwargs(context, [initial_message(context)]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preview', 'program'))
    parser.add_argument('--observation', type=Path)
    parser.add_argument('--program', type=Path)
    parser.add_argument('--execution-mode', choices=('skills', 'direct'))
    parser.add_argument('--task', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--trial-mode', choices=('practice', 'evaluation'), default='practice')
    parser.add_argument('--backend', choices=('kinematic',))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--trained-skills', default='', help='comma-separated skills with a trained S1 policy')
    parser.add_argument('--with-garf', action='store_true', help='include the GARF candidate tool definition')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate(config)
    if args.command == 'program':
        return run_program_command(args, config, parser)
    if args.observation is None:
        parser.error('preview requires --observation (legacy Claude tool request; sends nothing)')
    skills = [skill for skill in args.trained_skills.split(',') if skill]
    print(json.dumps(preview(load_observation(args.observation), config, skills, args.with_garf), indent=1))


def run_program_command(args, config, parser):
    """Skills prints one plan without motion; direct runs the existing seeded ASPIRE trial."""
    mode = args.execution_mode or config['s2']['execution_mode']
    program = args.program or config['s2']['program_path']
    if program is None:
        parser.error('program requires --program or s2.program_path')
    if mode == 'direct':
        if any(value is None for value in (args.task, args.seed, args.backend, args.output_root)):
            parser.error('direct mode requires --task --seed --backend --output-root')
        from ..baselines.aspire.__main__ import main as baseline_main
        command = ['run', '--program', str(program), '--task', str(args.task), '--seed', str(args.seed),
                   '--mode', args.trial_mode, '--backend', args.backend, '--output-root', str(args.output_root),
                   '--config', str(args.config), '--max-steps', str(args.max_steps)]
        if args.records is not None:
            command += ['--records', str(args.records)]
        if args.no_videos:
            command += ['--no-videos']
        return baseline_main(command)
    if args.observation is None:
        parser.error('skills mode requires --observation; this command plans only, without executing S1/S0')
    from .aspire import AspirePlanner
    context = PlanningContext(observation=load_observation(args.observation),
                              trained_skills=tuple(s for s in args.trained_skills.split(',') if s),
                              candidate_source=_Unconnected() if args.with_garf else None,
                              completion_defaults=config['s2']['completion_defaults'],
                              timeouts_s=config['s2']['timeouts_s'])
    outcome = AspirePlanner.from_file(program).plan(context)
    print(json.dumps({'planner': 'aspire_program', 'execution_mode': 'skills', 'motion_executed': False,
                      'status': outcome.status, 'reason': outcome.reason, 'trace': list(outcome.trace),
                      'skill_request': None if outcome.skill_request is None else outcome.skill_request.to_record()},
                     indent=2))
    return 0 if outcome.status == 'request' else 1


if __name__ == '__main__':
    raise SystemExit(main())
