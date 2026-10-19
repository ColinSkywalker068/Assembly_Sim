import json
import os
import sys
import time

import pytest

from tasks.dual_arm_crag_skull.s2.garf_client import (
    CachedCandidateSource, GarfProcessClient, answer_mismatch, registry_fingerprint)

RUNNER = r'''
import json, os, sys, time
pid_file = os.environ.get("RUNNER_PID_FILE")
if pid_file:
    open(pid_file, "w").write(str(os.getpid()))
for line in sys.stdin:
    envelope = json.loads(line)
    request = envelope["request"]
    if request.get("sleep"):
        time.sleep(request["sleep"])
    if request.get("die"):
        sys.exit(3)
    if request.get("stray"):
        print("CUDA: warning written straight to fd 1", flush=True)
    seeds = request.get("seeds", [])
    if request.get("wrong_seed"):
        seeds = [s + 1 for s in seeds]
    candidates = [{"status": "candidate", "seed": s, "observation_id": envelope["current_observation_id"],
                   "assembly_T_fragment": {k: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                                           for k in request.get("fragment_ids", [])}} for s in seeds]
    print(json.dumps({"status": "candidates", "candidates": candidates, "pid": os.getpid(),
                      "saw_api_key": "ANTHROPIC_API_KEY" in os.environ}), flush=True)
'''


@pytest.fixture
def script(tmp_path):
    path = tmp_path / 'runner.py'
    path.write_text(RUNNER)
    return path


@pytest.fixture
def runner(script):
    client = GarfProcessClient([sys.executable, str(script)], call_timeout_s=1.0, startup_timeout_s=5.0)
    yield client
    client.close()


def envelope(**request):
    request.setdefault('fragment_ids', ['a', 'b'])
    request.setdefault('seeds', [1])
    return {'tool_name': 'propose_assembly_candidates', 'current_observation_id': 'obs-1', 'request': request}


def test_round_trip_keeps_one_process(runner):
    first = runner.call(envelope(seeds=[1]))
    second = runner.call(envelope(seeds=[2]))
    assert first['candidates'][0]['seed'] == 1 and first['pid'] == second['pid']


def test_timeout_kills_and_next_call_restarts(runner):
    pid = runner.call(envelope())['pid']
    result = runner.call(envelope(sleep=3))
    assert result['status'] == 'failed' and 'timeout' in result['reason']
    again = runner.call(envelope())
    assert again['status'] == 'candidates' and again['pid'] != pid


def test_stray_line_before_the_answer_is_skipped(runner):
    result = runner.call(envelope(stray=True, seeds=[4]))
    assert result['candidates'][0]['seed'] == 4 and runner.stray_lines == 1
    assert runner.call(envelope(seeds=[5]))['candidates'][0]['seed'] == 5


def test_mismatched_answer_is_desync_and_restarts(runner):
    pid = runner.call(envelope())['pid']
    result = runner.call(envelope(wrong_seed=True, seeds=[7]))
    assert result['status'] == 'failed' and 'desynchronized' in result['reason']
    fresh = runner.call(envelope(seeds=[8]))
    assert fresh['candidates'][0]['seed'] == 8 and fresh['pid'] != pid


def test_crash_is_a_failure_not_an_exception(runner):
    assert runner.call(envelope(die=True))['reason'] == 'runner exited without answering'
    assert runner.call(envelope(seeds=[5]))['status'] == 'candidates'
    assert runner.restarts == 1


def test_timeout_kills_wrapped_runner_and_returns_on_time(script, tmp_path):
    pid_file = tmp_path / 'inner.pid'
    env = {**os.environ, 'RUNNER_PID_FILE': str(pid_file)}
    client = GarfProcessClient(['sh', '-c', f'"{sys.executable}" "{script}"'], env=env, call_timeout_s=1.0,
                               startup_timeout_s=5.0)
    try:
        client.call(envelope())
        start = time.monotonic()
        result = client.call(envelope(sleep=30))
        elapsed = time.monotonic() - start
        assert result['status'] == 'failed' and elapsed < 5.0
        inner = int(pid_file.read_text())
        time.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.kill(inner, 0)
    finally:
        client.close()


def test_runner_environment_has_no_planner_credentials(script, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test-not-a-key')
    client = GarfProcessClient([sys.executable, str(script)], call_timeout_s=1.0, startup_timeout_s=5.0)
    try:
        assert client.call(envelope())['saw_api_key'] is False
    finally:
        client.close()


def test_answer_mismatch_checks_seed_observation_and_fragments():
    sent = envelope(seeds=[1, 2])
    good = {'status': 'candidates', 'candidates': [
        {'status': 'candidate', 'seed': s, 'observation_id': 'obs-1', 'assembly_T_fragment': {'a': 0, 'b': 0}} for s in (1, 2)]}
    assert answer_mismatch(sent, good) is None
    for mutate in (lambda r: r['candidates'][1].update(seed=3), lambda r: r['candidates'][0].update(observation_id='obs-0'),
                   lambda r: r['candidates'][0]['assembly_T_fragment'].pop('b'), lambda r: r['candidates'].pop()):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        assert answer_mismatch(sent, bad)
    assert answer_mismatch(sent, {'status': 'rejected', 'reason': 'stale observation'}) is None


class CountingClient:
    def __init__(self, status='candidates'):
        self.calls, self.status = [], status

    def call(self, envelope):
        self.calls.append(envelope)
        obs = envelope['current_observation_id']
        return {'status': self.status, 'candidates': [{'status': 'candidate', 'seed': 41, 'observation_id': obs}]}


def test_cache_reuses_geometry_results_and_restamps_observation():
    client = CountingClient()
    source = CachedCandidateSource(client, 'geometry-v1')
    request = {'observation_id': 'obs-1', 'fragment_ids': ['a', 'b'], 'anchor_id': 'a', 'seeds': [41]}
    first = source.propose(request, current_observation_id='obs-1')
    second = source.propose({**request, 'observation_id': 'obs-2'}, current_observation_id='obs-2')
    assert len(client.calls) == 1 and source.hits == 1
    assert first['cache_hit'] is False and second['cache_hit'] is True
    assert second['candidates'][0]['observation_id'] == 'obs-2'
    stale = source.propose({**request, 'observation_id': 'obs-1'}, current_observation_id='obs-3')
    assert stale['cache_hit'] is False and len(client.calls) == 2


def test_incomplete_results_are_not_cached():
    client = CountingClient(status='incomplete')
    source = CachedCandidateSource(client, 'geometry-v1')
    request = {'observation_id': 'obs-1', 'fragment_ids': ['a', 'b'], 'anchor_id': 'a', 'seeds': [41]}
    source.propose(request, current_observation_id='obs-1')
    source.propose(request, current_observation_id='obs-1')
    assert len(client.calls) == 2


def test_registry_fingerprint_tracks_file_contents(tmp_path):
    part = tmp_path / 'a.npz'
    part.write_bytes(b'one')
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'a': str(part)}))
    before = registry_fingerprint(registry)
    part.write_bytes(b'two')
    assert registry_fingerprint(registry) != before
