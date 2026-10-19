"""Shared ASPIRE program runtime for skill planning and direct motion.

skills: one program invocation per fresh PlanningContext, returning one validated
SkillRequest to AssemblyOrchestrator. No motion API is registered.
direct: CapApi's public tools, as used by the ASPIRE baseline, with its S0 adapter.

Programs are authored/refined by an external coding agent, then frozen. This
module does not implement that agent's learning loop or an Isaac executor.
ASPIRE's AST checks are not a process sandbox or a wall-clock watchdog.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .planner import PlanOutcome

MODES = ('skills', 'direct')
SKILL_TOOLS = ('get_observation', 'get_task_context', 'propose_assembly_candidates', 'submit_skill_request')


def program_digest(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def execute_program(source, *, mode, api=None, tools=None, recorder=None):
    """Both modes use exactly ASPIRE's ToolRegistry and ProgramExecutor.

    Direct callers own their checked CapApi and its environment. Skill callers
    expose only the four planning tools. Refuse ambiguous/mixed registrations.
    """
    from cap_harness.registry import FANUC_CRX_PUBLIC_TOOL_NAMES, ToolRegistry
    from cap_harness.runtime import ProgramExecutor

    if mode == 'direct':
        if api is None or tools is not None:
            raise ValueError('direct mode requires api and forbids skill tools')
        # The CRX adapter's contact declarations (fanuc_crx.declare_contact / clear_contact) are its only extensions.
        registry = api.register_tools(ToolRegistry(public_extension_allowlist=FANUC_CRX_PUBLIC_TOOL_NAMES))
    elif mode == 'skills':
        if api is not None or tools is None or set(tools) != set(SKILL_TOOLS):
            raise ValueError('skills mode requires exactly the planning tools and forbids a motion api')
        registry = ToolRegistry(shared_public_names=SKILL_TOOLS, public_extension_allowlist=())
        for name in SKILL_TOOLS:
            registry.register(tools[name], name=name, layer='s2', capability='skill_planning', public=True)
    else:
        raise ValueError(f'unknown execution mode {mode!r}; choose from {MODES}')
    return ProgramExecutor(registry, recorder=recorder).execute_program(source)


def run_direct_program(*, api, program_path, recorder=None):
    """Shared entry used by the ASPIRE baseline and S2 direct mode."""
    path = Path(program_path).expanduser().resolve()
    source = path.read_text(encoding='utf-8')
    if not source.strip():
        raise ValueError('program file must contain non-empty Python source')
    if recorder is not None:
        recorder.save_program(path)
    return execute_program(source, mode='direct', api=api, recorder=recorder)


class AspirePlanner:
    """Frozen ASPIRE Python program implementing the existing S2 planner contract.

    The source is captured once. Each planning event gets fresh tools/state;
    execution never retains a Python global from a preceding observation.
    Any execution error discards the pending request, even if already submitted.
    """

    def __init__(self, source):
        if not isinstance(source, str) or not source.strip():
            raise ValueError('ASPIRE planner needs non-empty Python source')
        self.source = source
        self.sha256 = program_digest(source)

    @classmethod
    def from_file(cls, path):
        return cls(Path(path).read_text(encoding='utf-8'))

    def plan(self, context):
        from ..baselines.aspire.runner import installed_harness
        installed_harness()
        pending, violations = [], []

        def before_call():
            if pending:
                violations.append('no tool calls are allowed after submit_skill_request')
                raise ValueError(violations[-1])

        def get_observation():
            before_call()
            # Fresh plain values; no simulator handles or privileged poses.
            return {**context.observation.summary(),
                    'keyframes_png': dict(context.observation.keyframes_png)}

        def get_task_context():
            before_call()
            return {'execution_mode': 'skills', 'available_skills': list(context.allowed_skills),
                    'trained_arm_sets': (None if context.trained_combinations is None else
                                         {skill: [list(arms) for name, arms in sorted(context.trained_combinations)
                                                  if name == skill] for skill in context.trained_skills}),
                    'completion_defaults': json.loads(json.dumps(context.completion_defaults)),
                    'timeouts_s': dict(context.timeouts_s),
                    'feedback': json.loads(json.dumps(context.feedback))}

        def propose_assembly_candidates(observation_id, fragment_ids, anchor_id, seeds):
            before_call()
            result, failed = context.run_candidates(dict(observation_id=observation_id, fragment_ids=fragment_ids,
                                                         anchor_id=anchor_id, seeds=seeds))
            if failed:
                raise ValueError(result)
            return json.loads(result)

        def submit_skill_request(observation_id, skill_id, active_arms, fragment_id=None,
                                 support_fragment_id=None, target_candidate_id=None, rationale=''):
            before_call()
            request = context.resolve(dict(observation_id=observation_id, skill_id=skill_id,
                                           active_arms=active_arms, fragment_id=fragment_id,
                                           support_fragment_id=support_fragment_id,
                                           target_candidate_id=target_candidate_id, rationale=rationale))
            pending.append(request)
            return {'accepted': True, 'request_id': request.request_id}

        execution = execute_program(self.source, mode='skills', tools={
            'get_observation': get_observation, 'get_task_context': get_task_context,
            'propose_assembly_candidates': propose_assembly_candidates, 'submit_skill_request': submit_skill_request})
        trace = ({'planner': 'aspire_program', 'execution_mode': 'skills', 'program_sha256': self.sha256,
                  'program_ok': execution.ok, 'stdout': execution.stdout,
                  'calls': [{'tool': c.tool, 'ok': c.ok, 'duration_s': c.duration_s,
                             'error': c.error_message} for c in execution.calls]},)
        if not execution.ok or violations:
            return PlanOutcome('error', reason='; '.join(violations) if violations else str(execution.error), trace=trace)
        if not pending:
            return PlanOutcome('no_request', reason='program ended without a skill submission', trace=trace)
        return PlanOutcome('request', skill_request=pending[0], trace=trace)


def build_planner(config, *, program_path=None, client=None):
    """Configured skills planner; direct mode belongs to the seeded trial runner."""
    settings = config['s2']
    if settings['execution_mode'] != 'skills':
        raise ValueError('direct mode uses the ASPIRE trial runner, not AssemblyOrchestrator')
    if settings['planner'] == 'aspire_program':
        path = program_path or settings.get('program_path')
        if path is None:
            raise ValueError('ASPIRE S2 requires a frozen program: pass --program or set s2.program_path')
        return AspirePlanner.from_file(path)
    if settings['planner'] == 'claude_tools':
        from .claude import ClaudePlanner
        return ClaudePlanner(client=client, model=settings['api_model_id'], effort=settings['effort'],
                             max_turns=settings['max_turns'], server_fallbacks=settings['server_fallbacks'] == 'default')
    raise ValueError(f"unknown S2 planner: {settings['planner']!r}")
