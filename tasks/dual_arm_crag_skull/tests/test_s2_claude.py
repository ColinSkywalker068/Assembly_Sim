import json
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.dual_arm_crag_skull.s2.claude import DEFAULT_MODEL, FALLBACK_BETA, ClaudePlanner, initial_message
from tasks.dual_arm_crag_skull.s2.planner import PlanningContext
from tasks.dual_arm_crag_skull.tests.s2_fakes import (
    COMPLETIONS, TIMEOUTS, FakeCandidateSource, expected_goal, observation)


def tool_use(name, payload, block_id):
    return SimpleNamespace(type='tool_use', id=block_id, name=name, input=payload)


def text(value):
    return SimpleNamespace(type='text', text=value)


def response(*blocks, stop_reason='tool_use', iterations=None, **extra):
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, iterations=iterations)
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, model=DEFAULT_MODEL, usage=usage,
                           _request_id='req_test', **extra)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # The loop appends to one messages list, as the SDK allows; snapshot what was sent.
        self.calls.append({**kwargs, 'messages': list(kwargs['messages'])})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


PROPOSE = {'observation_id': 'obs-1', 'fragment_ids': ['part_0', 'part_1'], 'anchor_id': 'part_0', 'seeds': [41, 42]}
SUBMIT = {'observation_id': 'obs-1', 'skill_id': 'place', 'active_arms': ['left'], 'fragment_id': 'part_1',
          'support_fragment_id': None, 'target_candidate_id': 'obs-1/call-0/seed-41', 'rationale': 'screened goal'}
REOBSERVE = dict(SUBMIT, skill_id='reobserve', active_arms=[], fragment_id=None, target_candidate_id=None)


def context(source=None):
    return PlanningContext(observation=observation(), trained_skills=('pick', 'place'),
                           candidate_source=source or FakeCandidateSource(), completion_defaults=COMPLETIONS,
                           timeouts_s=TIMEOUTS, new_request_id=lambda: 'req-9')


def test_tool_loop_ends_in_validated_request_and_sends_expected_parameters():
    client = FakeClient([response(tool_use('propose_assembly_candidates', PROPOSE, 't1')),
                         response(tool_use('submit_skill_request', SUBMIT, 't2'))])
    outcome = ClaudePlanner(client).plan(context())
    assert outcome.status == 'request'
    np.testing.assert_allclose(outcome.skill_request.target_pose, expected_goal('part_1'), atol=1e-12)
    first = client.calls[0]
    assert first['model'] == 'claude-opus-5-5'
    assert first['tool_choice'] == {'type': 'auto', 'disable_parallel_tool_use': True}
    assert first['betas'] == [FALLBACK_BETA] and first['fallbacks'] == 'default'
    assert first['output_config'] == {'effort': 'high'} and 'thinking' not in first
    image = first['messages'][0]['content'][1]
    assert image['type'] == 'image' and image['source']['media_type'] == 'image/png'
    second = client.calls[1]['messages']
    assert second[-1]['content'][0]['tool_use_id'] == 't1' and 'is_error' not in second[-1]['content'][0]
    assert second[-2]['role'] == 'assistant'


def test_rejected_submission_is_returned_as_error_and_model_can_correct():
    bad = dict(SUBMIT, target_candidate_id='obs-1/call-0/seed-42')
    client = FakeClient([response(tool_use('propose_assembly_candidates', PROPOSE, 't1')),
                         response(tool_use('submit_skill_request', bad, 't2')),
                         response(tool_use('submit_skill_request', REOBSERVE, 't3'))])
    outcome = ClaudePlanner(client).plan(context())
    assert outcome.status == 'request' and outcome.skill_request.skill_id == 'reobserve'
    error_result = client.calls[2]['messages'][-1]['content'][0]
    assert error_result['is_error'] is True and 'geometry screen' in error_result['content']


def test_several_calls_in_one_turn_are_all_refused_unrun():
    source = FakeCandidateSource()
    client = FakeClient([response(tool_use('propose_assembly_candidates', PROPOSE, 't1'),
                                  tool_use('submit_skill_request', SUBMIT, 't2')),
                         response(tool_use('submit_skill_request', REOBSERVE, 't3'))])
    outcome = ClaudePlanner(client).plan(context(source))
    assert outcome.status == 'request' and outcome.skill_request.skill_id == 'reobserve'
    assert source.calls == []
    results = client.calls[1]['messages'][-1]['content']
    assert [r['tool_use_id'] for r in results] == ['t1', 't2'] and all(r['is_error'] for r in results)


def test_refusal_and_truncation_never_run_tools():
    source = FakeCandidateSource()
    refused = response(tool_use('propose_assembly_candidates', PROPOSE, 't0'), stop_reason='refusal',
                       stop_details=SimpleNamespace(category='bio'))
    assert ClaudePlanner(FakeClient([refused])).plan(context(source)).status == 'refused'
    truncated = response(tool_use('propose_assembly_candidates', PROPOSE, 't1'), stop_reason='max_tokens')
    outcome = ClaudePlanner(FakeClient([truncated])).plan(context(source))
    assert outcome.status == 'error' and outcome.retryable and source.calls == []


def test_calls_before_the_last_fallback_block_are_not_run_or_echoed():
    source = FakeCandidateSource()
    fallback = SimpleNamespace(type='fallback')
    declined = [SimpleNamespace(type='thinking', thinking=''), text('partial'),
                tool_use('submit_skill_request', SUBMIT, 'declined-1')]
    served = response(*declined, fallback, tool_use('propose_assembly_candidates', PROPOSE, 't1'),
                      iterations=[SimpleNamespace(type='message'), SimpleNamespace(type='fallback_message')])
    client = FakeClient([served, response(tool_use('submit_skill_request', REOBSERVE, 't2'))])
    outcome = ClaudePlanner(client).plan(context(source))
    assert outcome.status == 'request' and outcome.skill_request.skill_id == 'reobserve'
    assert len(source.calls) == 1
    echoed = client.calls[1]['messages'][1]['content']
    assert [getattr(b, 'type') for b in echoed] == ['text', 'fallback', 'tool_use'] and echoed[2].id == 't1'
    assert [r['tool_use_id'] for r in client.calls[1]['messages'][2]['content']] == ['t1']
    assert outcome.trace[0]['served_by_fallback'] is True and outcome.trace[0]['fallback_block_present'] is True


def test_end_turn_without_submission_gets_one_reminder_then_no_request():
    client = FakeClient([response(text('I think we should wait.'), stop_reason='end_turn'),
                         response(text('Still waiting.'), stop_reason='end_turn')])
    outcome = ClaudePlanner(client).plan(context())
    assert outcome.status == 'no_request'
    assert client.calls[1]['messages'][-1] == {'role': 'user',
                                               'content': 'Finish by calling submit_skill_request exactly once.'}


def test_turn_budget_is_enforced_and_must_be_an_integer():
    loop = [response(tool_use('propose_assembly_candidates', PROPOSE, f't{i}')) for i in range(3)]
    outcome = ClaudePlanner(FakeClient(loop), max_turns=3).plan(context())
    assert outcome.status == 'no_request' and 'within 3 turns' in outcome.reason
    with pytest.raises(ValueError):
        ClaudePlanner(FakeClient([]), max_turns=2.5)


def test_unknown_tool_is_an_error_result():
    client = FakeClient([response(tool_use('move_arm', {'q': [0] * 6}, 't1')),
                         response(tool_use('submit_skill_request', REOBSERVE, 't2'))])
    outcome = ClaudePlanner(client).plan(context())
    assert outcome.status == 'request'
    assert client.calls[1]['messages'][-1]['content'][0]['is_error'] is True


def test_sdk_errors_map_to_typed_outcomes():
    anthropic = pytest.importorskip('anthropic')
    import httpx2
    request = httpx2.Request('POST', 'https://api.anthropic.com/v1/messages')
    outcome = ClaudePlanner(FakeClient([anthropic.APIConnectionError(request=request)])).plan(context())
    assert outcome.status == 'error' and outcome.retryable
    status = httpx2.Response(429, request=request)
    limited = anthropic.RateLimitError('rate limited', response=status, body=None)
    outcome = ClaudePlanner(FakeClient([limited])).plan(context())
    assert outcome.status == 'error' and outcome.retryable and 'rate limited' in outcome.reason
    with pytest.raises(ZeroDivisionError):
        ClaudePlanner(FakeClient([ZeroDivisionError()])).plan(context())


def test_trained_arm_sets_are_stated_to_the_model():
    def facts(ctx):
        return json.loads(initial_message(ctx)['content'][-1]['text'].split('\n', 1)[1])
    assert 'trained_arm_sets' not in facts(context())
    trained = PlanningContext(observation=observation(), trained_skills=('pick', 'place'), candidate_source=None,
                              completion_defaults=COMPLETIONS, timeouts_s=TIMEOUTS,
                              trained_combinations={('pick', ('left',)), ('pick', ('right',)), ('place', ('left',))})
    assert facts(trained)['trained_arm_sets'] == {'pick': [['left'], ['right']], 'place': [['left']]}
