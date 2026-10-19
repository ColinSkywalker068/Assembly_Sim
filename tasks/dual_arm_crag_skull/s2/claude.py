"""Claude S2 planner: a bounded manual tool-use loop that ends in one validated SkillRequest.

A manual loop (not the beta tool runner) because the loop must stop deterministically on the
terminal submit tool, refuse to run tools from refused or truncated turns, and resolve goals
from local state. Forced tool_choice is rejected by Claude Opus 5.5, so the loop uses `auto`
with one tool call per turn, strict schemas and one reminder when a turn ends without a
submission. With server-side fallbacks, output before the last `fallback` block belongs to a hop
that declined: its tool calls are never run and its thinking/tool_use blocks are not echoed back.
"""
from __future__ import annotations

import base64
import json

from ..contracts.skills import SkillRejected
from .planner import PlanOutcome

DEFAULT_MODEL = 'claude-opus-5-5'
FALLBACK_BETA = 'server-side-fallback-2026-07-01'
SYSTEM_PROMPT = """\
You are S2, the task planner of a two-arm robot cell that reassembles a fractured skull from
printed fragments. Each call you receive one observation and you submit exactly one skill
request with submit_skill_request. You never command motion: S1 (a learned visuomotor policy)
executes the skill and S0 enforces joint, speed and collision limits.

Ground every choice in the observation. Fragment poses come from perception, verified clusters
are fragments already confirmed rigidly assembled, and feedback lists recent skill outcomes.
For place, align and insert, call propose_assembly_candidates first and cite a candidate whose
geometry_status is needs_motion_validation; candidates are unranked hypotheses and rejected ones
are not safe targets. Work one arm at a time unless a two-arm skill is clearly required. When
trained_arm_sets is given, a moving skill may only use one of the arm sets listed for it.
If perception is missing for a needed fragment, every candidate is rejected, or the last attempt
failed for an unexplained reason, submit reobserve and say why in the rationale.
"""


def _image_block(png):
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                        'data': base64.standard_b64encode(png).decode('ascii')}}


def initial_message(context):
    content = []
    for camera, png in sorted(context.observation.keyframes_png.items()):
        content.append({'type': 'text', 'text': f'Keyframe: {camera}'})
        content.append(_image_block(png))
    facts = {'observation': context.observation.summary(), 'trained_skills': list(context.trained_skills),
             'available_skills': list(context.allowed_skills), 'feedback': list(context.feedback)}
    if context.trained_combinations is not None:
        facts['trained_arm_sets'] = {skill: [list(arms) for name, arms in sorted(context.trained_combinations)
                                             if name == skill] for skill in context.trained_skills}
    content.append({'type': 'text', 'text': 'Current state (JSON):\n' + json.dumps(facts, indent=1)})
    return {'role': 'user', 'content': content}


def _after_last_fallback(content):
    """(echo, live): history to send back, and the blocks produced by the model that served the turn."""
    content = list(content)
    boundary = max((i for i, block in enumerate(content) if getattr(block, 'type', None) == 'fallback'), default=-1)
    declined = [block for block in content[:boundary + 1]
                if getattr(block, 'type', None) in ('text', 'fallback')]
    return declined + content[boundary + 1:], content[boundary + 1:]


def _served_by_fallback(response):
    iterations = getattr(getattr(response, 'usage', None), 'iterations', None) or []
    return any(getattr(entry, 'type', None) == 'fallback_message' for entry in iterations)


def _tool_result(block_id, text, is_error):
    result = {'type': 'tool_result', 'tool_use_id': block_id, 'content': text}
    if is_error:
        result['is_error'] = True
    return result


class ClaudePlanner:
    def __init__(self, client=None, *, model=DEFAULT_MODEL, effort='high', max_tokens=16000, max_turns=6,
                 server_fallbacks=True):
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or not 1 <= max_turns <= 20:
            raise ValueError('max_turns must be an integer in 1..20')
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # credentials from the environment or `ant auth login`
        self.client, self.model, self.effort = client, model, effort
        self.max_tokens, self.max_turns, self.server_fallbacks = max_tokens, max_turns, server_fallbacks

    def request_kwargs(self, context, messages):
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, output_config={'effort': self.effort},
                      system=[{'type': 'text', 'text': SYSTEM_PROMPT, 'cache_control': {'type': 'ephemeral'}}],
                      tools=context.tools(), tool_choice={'type': 'auto', 'disable_parallel_tool_use': True},
                      messages=messages)
        if self.server_fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks='default')
        return kwargs

    def _create(self, context, messages):
        return self.client.beta.messages.create(**self.request_kwargs(context, messages))

    def plan(self, context):
        try:
            return self._loop(context)
        except Exception as exc:  # noqa: BLE001 - mapped to typed outcomes below, never swallowed silently
            return _api_failure(exc)

    def _loop(self, context):
        messages = [initial_message(context)]
        trace, reminded = [], False
        for turn in range(self.max_turns):
            response = self._create(context, messages)
            trace.append({'turn': turn, 'model': getattr(response, 'model', None),
                          'request_id': getattr(response, '_request_id', None), 'stop_reason': response.stop_reason,
                          'fallback_block_present': any(getattr(b, 'type', None) == 'fallback' for b in response.content),
                          'served_by_fallback': _served_by_fallback(response),
                          'usage': _usage(response)})
            if response.stop_reason == 'refusal':
                details = getattr(response, 'stop_details', None)
                return PlanOutcome('refused', reason=f'model declined: {getattr(details, "category", None)}',
                                   trace=tuple(trace))
            if response.stop_reason == 'max_tokens':
                return PlanOutcome('error', reason='hit max_tokens; tool calls from a truncated turn are not run',
                                   retryable=True, trace=tuple(trace))
            if response.stop_reason not in ('tool_use', 'end_turn'):
                return PlanOutcome('error', reason=f'unexpected stop_reason {response.stop_reason!r}',
                                   trace=tuple(trace))
            echo, live = _after_last_fallback(response.content)
            messages.append({'role': 'assistant', 'content': echo})
            calls = [block for block in live if getattr(block, 'type', None) == 'tool_use']
            if not calls:
                if reminded:
                    return PlanOutcome('no_request', reason='turn ended without submit_skill_request',
                                       trace=tuple(trace))
                reminded = True
                messages.append({'role': 'user', 'content': 'Finish by calling submit_skill_request exactly once.'})
                continue
            results, request = self._run_calls(context, calls, trace)
            if request is not None:
                return PlanOutcome('request', skill_request=request, trace=tuple(trace))
            messages.append({'role': 'user', 'content': results})
        return PlanOutcome('no_request', reason=f'no valid submission within {self.max_turns} turns', trace=tuple(trace))

    @staticmethod
    def _run_calls(context, calls, trace):
        """One call per turn is requested; several are all refused so no decision rests on unseen results."""
        if len(calls) > 1:
            text = json.dumps({'error': 'make exactly one tool call per turn; none of these calls were run'})
            trace.append({'tool': [block.name for block in calls], 'is_error': True, 'result': text})
            return [_tool_result(block.id, text, True) for block in calls], None
        results, request = [], None
        for block in calls:
            if block.name == 'propose_assembly_candidates':
                text, is_error = context.run_candidates(block.input)
            elif block.name == 'submit_skill_request' and request is None:
                try:
                    request = context.resolve(dict(block.input))
                    text, is_error = json.dumps({'accepted': request.request_id}), False
                except (SkillRejected, KeyError, TypeError) as exc:
                    text, is_error = json.dumps({'rejected': str(exc)}), True
            else:
                text, is_error = json.dumps({'error': f'unexpected tool call {block.name!r}'}), True
            trace.append({'tool': block.name, 'is_error': is_error, 'result': text[:500]})
            results.append(_tool_result(block.id, text, is_error))
        return results, request


def _usage(response):
    usage = getattr(response, 'usage', None)
    if usage is None:
        return None
    return {key: getattr(usage, key, None) for key in ('input_tokens', 'output_tokens', 'cache_read_input_tokens')}


def _api_failure(exc):
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the default client needs the SDK anyway
        anthropic = None
    if anthropic is not None:
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return PlanOutcome('error', reason=f'credentials rejected: {type(exc).__name__}')
        if isinstance(exc, (anthropic.BadRequestError, anthropic.NotFoundError)):
            return PlanOutcome('error', reason=f'request rejected by the API: {getattr(exc, "message", exc)}')
        if isinstance(exc, anthropic.RateLimitError):
            return PlanOutcome('error', reason='rate limited after SDK retries', retryable=True)
        if isinstance(exc, anthropic.APIStatusError):
            return PlanOutcome('error', reason=f'API status {exc.status_code}', retryable=exc.status_code >= 500)
        if isinstance(exc, anthropic.APIConnectionError):
            return PlanOutcome('error', reason=f'connection failed: {type(exc).__name__}', retryable=True)
    raise exc
