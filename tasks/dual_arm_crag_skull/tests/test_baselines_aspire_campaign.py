"""An ASPIRE campaign's layout, prompt and agent sandbox, and a whole campaign run by a scripted agent through the
trial service (a fake launcher stands in for Isaac Sim; no model is called)."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG
from tasks.dual_arm_crag_skull.baselines import sandbox
from tasks.dual_arm_crag_skull.baselines.aspire import campaign as camp
from tasks.dual_arm_crag_skull.tests.test_baselines_aspire_trials import FakeLauncher
from tasks.dual_arm_crag_skull.workspace import REPO_ROOT

CONFIG = json.loads(DEFAULT_CONFIG.read_text())
CHECKOUT = REPO_ROOT / CONFIG['baselines']['aspire']['checkout_path']
TASK = REPO_ROOT / 'runs' / 'baseline_tasks' / 'skull_base.json'
needs_inputs = pytest.mark.skipif(not (CHECKOUT / camp.UPSTREAM_PROMPT).is_file() or not TASK.is_file(),
                                  reason='the patched ASPIRE checkout or the base task is absent')
needs_bwrap = pytest.mark.skipif(not sandbox.available(), reason='bwrap is not installed')


def fake_installs(tmp_path):
    env = tmp_path / 'isaac_env'
    kit = env / 'lib' / 'python3.11' / 'site-packages' / 'isaacsim' / 'kit'
    (kit / 'cache' / 'Kit').mkdir(parents=True)
    (kit / 'cache' / 'Kit' / 'material_cache.json').write_text('{}')
    (kit / 'logs').mkdir()
    (env / 'bin').mkdir()
    package = tmp_path / 'claude-code'
    (package / 'bin').mkdir(parents=True)
    claude = package / 'bin' / 'claude.exe'
    claude.write_text('#!/bin/sh\necho "9.9.9 (Claude Code)"\n')
    claude.chmod(0o755)
    home = tmp_path / 'home'
    (home / '.cache' / 'ov').mkdir(parents=True)
    (home / '.cache' / 'ov' / 'shader.bin').write_text('x')
    return env / 'bin' / 'python', claude, home


def prepared(tmp_path, **options):
    python, claude, home = fake_installs(tmp_path)
    root = tmp_path / 'campaign'
    options.setdefault('require_clean', False)  # the working tree may carry the change under test
    camp.prepare(root, task=TASK, gpus=['3', '5'], isaac_python=python, claude=claude, config=CONFIG, home=home,
                 **options)
    return root


@needs_inputs
def test_prepare_lays_out_the_campaign_once(tmp_path):
    root = prepared(tmp_path)
    found = json.loads((root / 'campaign.json').read_text())
    assert found['trials']['dev_seeds'] == [101, 102, 103, 104, 105]
    assert found['trials']['unseen_seeds'] == [126, 127, 128, 129, 130] and found['trials']['gpus'] == [3, 5]
    assert found['prompt']['upstream_sha256'] == camp.UPSTREAM_PROMPT_SHA256
    assert found['agent']['claude_version'] == '9.9.9 (Claude Code)' and found['agent']['model'] == 'claude-opus-5-5'
    assert (root / 'service/kit_cache/Kit/material_cache.json').is_file()
    assert (root / 'service/isaac_home/.cache/ov/shader.bin').is_file()
    assert os.access(root / 'bin' / 'aspire-trial', os.X_OK)
    assert sorted(path.name for path in (root / 'reference/embodiment').iterdir()) == ['adapter.py', 'embodiment.md',
                                                                                      'ik.py']
    with pytest.raises(FileExistsError):
        prepared(tmp_path)


@needs_inputs
def test_prepare_freezes_the_code_the_trials_run(tmp_path):
    root = prepared(tmp_path)
    found = json.loads((root / 'campaign.json').read_text())
    code = Path(found['trials']['code'])
    head = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()
    assert code == root / 'service' / 'code' and found['code']['paths'] == ['tasks', 'core']
    assert found['code']['commit'] == head and len(found['code']['sha256']) == 64
    for name in ('tasks/dual_arm_crag_skull/baselines/aspire/adapter.py', 'tasks/dual_arm_crag_skull/configs/jing.json',
                 'core/assets/robots/fanuc_crx10ial/crx10ial.usd'):
        assert (code / name).read_bytes() == (REPO_ROOT / name).read_bytes()
    assert not list(code.rglob('__pycache__')) and not (code / 'runs').exists()


@needs_inputs
def test_a_live_campaign_runs_one_commit(tmp_path, monkeypatch):
    changed = ['tasks/dual_arm_crag_skull/s0/adapter.py']
    monkeypatch.setattr(camp, 'code_state', lambda *args, **kwargs: ('0' * 40, changed))
    with pytest.raises(ValueError, match='commit'):
        prepared(tmp_path / 'live', require_clean=True)
    assert not (tmp_path / 'live' / 'campaign').exists()
    assert json.loads((prepared(tmp_path / 'test') / 'campaign.json').read_text())['code']['modified'] == changed


@needs_inputs
def test_the_prompt_is_filled_and_keeps_upstreams_protocol(tmp_path):
    prompt = (prepared(tmp_path) / 'prompt.md').read_text()
    assert '<<' not in prompt and 'DEV_SEEDS: 101 102 103 104 105' in prompt
    assert 'UNSEEN_SEEDS: 126 127 128 129 130' in prompt and 'held-out seeds 1–100' in prompt
    for upstream_rule in ('Budget: 3 versions per mode, 8 in total.', 'Debug by **failure mode, not by seed**',
                          'Run the selected program **once**', 'This is a provenance rule, not only an API-name ban',
                          'Never run `rm`'):
        assert upstream_rule in prompt
    with pytest.raises(ValueError, match='placeholder'):
        camp.fill_prompt('<<NEW>>', {'max_steps': 1, 'dev_seeds': [], 'unseen_seeds': [], 'max_probes': 1})


def test_the_embodiment_notes_are_the_readme_sections_verbatim():
    notes = camp.embodiment_notes(camp.HERE / 'README.md')
    readme = (camp.HERE / 'README.md').read_text()
    assert '## Conventions' in notes and "## Every command goes through S0's checks" in notes
    body = notes.split('\n', 2)[2]
    assert all(part.strip() in readme for part in body.split('\n## ')[1:])
    assert '## Setup' not in notes and '## Information parity' not in notes


def test_the_tool_frame_the_agent_reads_is_the_models():
    """The README's tool frame (Conventions, in embodiment.md) is that of yam_model, the trials' model."""
    import numpy as np
    from tasks.dual_arm_crag_skull.s0.kinematics import yam_tool_geometry
    flange_T_tcp, _, box = yam_tool_geometry()
    notes = ' '.join(camp.embodiment_notes(camp.HERE / 'README.md').split())
    tcp = flange_T_tcp[:3, 3]
    assert np.allclose(flange_T_tcp[:3, :3], np.eye(3)) and np.allclose(tcp[1:], 0.0, atol=2e-4)
    assert f"{tcp[0] * 1000:.1f} mm out along the flange's +x" in notes
    fingers, body = box['parts']['fingers'], box['parts']['body']
    assert abs(fingers['max'][0] - tcp[0]) < 2e-4  # the TCP is on the fingertips' end faces
    assert f"front face is {round((tcp[0] - body['max'][0]) * 1000)} mm behind the TCP" in notes
    tips = fingers['stroke']['tips'].values()
    assert all(np.allclose(np.abs(tip['slide_axis']), [0.0, 1.0, 0.0], atol=1e-5) for tip in tips)  # close along y
    size = lambda slab: [round((slab['max'][k] - slab['min'][k]) * 1000) for k in (2, 1)]  # noqa: E731  (z, y) mm
    for tip in tips:
        root = [slab for slab in tip['slabs'] if abs(tcp[0] - slab['min'][0] - 0.070) < 5e-4]
        assert len(root) == 1 and size(root[0]) == [68, 21] and size(tip['slabs'][-1]) == [18, 4]
    assert "fingers' last 70 mm taper from 68 mm (tool z) by 21 mm (tool y) to 18 mm by 4 mm at the tips" in notes


@needs_inputs
def test_the_agents_sandbox_has_no_gpu_no_home_and_only_its_sockets(tmp_path):
    root = prepared(tmp_path)
    campaign = json.loads((root / 'campaign.json').read_text())
    (root / 'sockets' / 'trials.sock').touch()
    import socket
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(root / 'sockets' / 'trials.sock.real'))
    os.replace(root / 'sockets' / 'trials.sock.real', root / 'sockets' / 'trials.sock')
    command = camp.agent_command(root, campaign)
    binds = {command[i + 2]: (command[i], command[i + 1]) for i in range(len(command) - 2)
             if command[i] in ('--bind', '--ro-bind', '--dev-bind')}
    assert binds['/workspace'][0] == '--bind' and binds['/home/agent'][0] == '--bind'
    assert binds['/aspire'] == ('--ro-bind', str(CHECKOUT)) and binds['/run/aspire/trials.sock'][0] == '--bind'
    assert binds['/workspace/debug'] == ('--ro-bind', str(root / 'debug'))
    assert command.index('/workspace/debug') > command.index('/workspace')  # over the writable workspace
    assert '/run/aspire/relay.sock' not in binds and '/sys' not in binds
    assert not any(flag == '--dev-bind' for flag, _ in binds.values())
    assert not any(str(source).startswith(str(root / 'service')) for _, source in binds.values())
    assert str(Path.home()) not in binds and str(REPO_ROOT / 'runs') not in binds
    tail = command[command.index('--') + 1:]
    assert tail[:2] == ['/bin/bash', '-c'] and '--bare' not in tail and tail[tail.index('--model') + 1] == camp.MODEL
    assert tail[tail.index('--tools') + 1] == 'Bash,Read,Write,Edit,Glob,Grep'
    assert tail[tail.index('--permission-mode') + 1] == 'dontAsk' and 'Bash(rm:*)' in tail
    environment = camp.agent_environment()
    assert environment['ANTHROPIC_API_KEY'] == camp.PLACEHOLDER_KEY and not environment['ANTHROPIC_API_KEY'].startswith(
        'sk-ant-')
    assert {environment[name] for name in camp.MODEL_VARIABLES} == {camp.MODEL}
    assert environment['ANTHROPIC_BASE_URL'] == f'http://127.0.0.1:{camp.FORWARD_PORT}'
    listener.close()


SCRIPTED_AGENT = r'''
set -u
cd /workspace
mkdir -p code
cp /aspire/.claude/skills/iterative-debugging/scripts/scene_snapshot.py code/probe_scene.py
aspire-trial probe code/probe_scene.py --seed 101
echo 'result = "win"' > code/v00_initial.py
aspire-trial sweep code/v00_initial.py
echo 'result = 1' > code/v01_other.py
aspire-trial sweep code/v01_other.py
aspire-trial probe code/probe_scene.py --seed 7; echo "held-out probe exit $?"
cp code/v00_initial.py fix_code.py
aspire-trial unseen
aspire-trial unseen; echo "second unseen exit $?"
head -c 9 /workspace/debug/initial/results.jsonl; echo
touch /workspace/debug/planted 2>&1 | tail -1
ln -s /etc /workspace/debug/logs/v99 2>&1 | tail -1
ls /home/jz6676/.config 2>&1 | head -1
ls /local_data/jz6676/RoboAssembly/runs 2>&1 | head -1
cat /run/aspire/../../campaign/service/ledger.jsonl 2>&1 | head -1
python3 -c "import urllib.request; urllib.request.urlopen('https://api.anthropic.com', timeout=3)" 2>&1 | tail -1
echo "outer variables: $(env | grep -c ASPIRE_TEST_OUTER_MARKER)"
echo "prompt bytes $(wc -c)"
'''


@needs_inputs
@needs_bwrap
def test_a_scripted_agent_runs_a_whole_campaign_through_the_service_alone(tmp_path, monkeypatch):
    monkeypatch.setenv('ASPIRE_TEST_OUTER_MARKER', 'this process only')
    root = prepared(tmp_path)
    (root / 'workspace' / 'agent.sh').write_text(SCRIPTED_AGENT)
    launcher = FakeLauncher()
    summary = camp.run(root, live=False, agent=['/bin/bash', '/workspace/agent.sh'], launch=launcher)
    # the script lives in the workspace only for this test; a real agent writes its own files there
    transcript = (root / 'transcript.jsonl').read_text()
    assert summary['versions'] == ['v00', 'v01'] and summary['probes'] == 1
    assert summary['passed'] == {'probe01_scene': [0, 1], 'v00': [5, 5], 'v01': [0, 5], 'unseen_v00': [5, 5]}
    assert summary['sim_trials'] == 1 + 5 + 5 + 5 and summary['selected'] == 'v00'
    assert summary['refused_requests'] == 2 and 'held-out probe exit 3' in transcript
    assert 'second unseen exit 3' in transcript
    assert sorted({call[1] for call in launcher.calls}) == [101, 102, 103, 104, 105, 126, 127, 128, 129, 130]
    assert 'No such file or directory' in transcript and 'permission' not in transcript.lower()
    assert '"event": "done"' in transcript and summary['agent']['exit_code'] == 0
    assert 'prompt bytes' in transcript and 'prompt bytes 0' not in transcript
    assert 'outer variables: 0' in transcript
    assert transcript.count('Read-only file system') == 2 and '{"error":' in transcript
    assert not (root / 'debug' / 'planted').exists() and not (root / 'debug' / 'logs' / 'v99').exists()
    assert sorted(path.name for path in (root / 'workspace').iterdir()) == ['agent.sh', 'code', 'debug', 'fix_code.py']
    assert (root / 'frozen' / 'fix_code.py').read_text() == 'result = "win"\n'
    with pytest.raises(FileExistsError):
        camp.run(root, live=False, agent=['/bin/true'], launch=launcher)


@needs_inputs
@needs_bwrap
def test_the_agents_sandbox_hides_the_host(tmp_path):
    root = prepared(tmp_path)
    campaign = json.loads((root / 'campaign.json').read_text())
    import socket
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(root / 'sockets' / 'trials.sock'))
    prefix = camp.agent_command(root, campaign, agent=['true'])
    prefix = prefix[:prefix.index('--') + 1]
    hidden = [Path.home() / '.config', Path.home() / '.ssh', REPO_ROOT / 'runs', root / 'service', root / 'sockets',
              root / 'campaign.json', REPO_ROOT / 'tasks']
    found = sandbox.probe('/usr/bin/python3', prefix, hidden=hidden, home=Path.home(),
                          writable=[Path('/workspace'), Path('/home/agent')], readonly=[Path('/aspire')],
                          env={'PATH': '/usr/bin:/bin', 'HOME': '/home/agent'})
    listener.close()
    assert found['visible'] == [] and found['internet'] != 'connected' and found['dns'] != 'resolves'
    assert found['host_loopback'] != 'connected' and found['credential_env'] == []
    assert found['writes'] == {'/workspace': True, '/home/agent': True, '/aspire': False}


RELAY_AGENT = r"""
import json, os, socket, urllib.request, urllib.error
def post(model):
    body = json.dumps({'model': model, 'max_tokens': 64, 'messages': [{'role': 'user', 'content': 'hi'}]}).encode()
    request = urllib.request.Request(os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages', data=body, method='POST',
                                     headers={'content-type': 'application/json',
                                              'x-api-key': os.environ['ANTHROPIC_API_KEY'],
                                              'anthropic-version': '2023-06-01'})
    try:
        with urllib.request.urlopen(request, timeout=20) as answer:
            return answer.status, json.loads(answer.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
print(json.dumps({'served': post('claude-opus-5-5'), 'other': post('claude-sonnet-5-5')[0],
                  'key': os.environ['ANTHROPIC_API_KEY'], 'base': os.environ['ANTHROPIC_BASE_URL']}))
try:
    socket.create_connection(('127.0.0.1', int(os.environ['UPSTREAM_PORT'])), timeout=3).close()
    print('UPSTREAM REACHABLE')
except OSError:
    print('upstream unreachable')
"""


@needs_inputs
@needs_bwrap
def test_a_live_campaign_reaches_the_api_only_through_the_relay(tmp_path, monkeypatch, protected):
    from tasks.dual_arm_crag_skull.baselines.budget import BudgetPool
    from tasks.dual_arm_crag_skull.tests.test_baselines_relay import REAL_KEY, FakeUpstream, json_reply
    root = prepared(tmp_path)
    (root / 'workspace' / 'agent.py').write_text(RELAY_AGENT)
    BudgetPool.create(tmp_path / 'pool', 5.0, note='test')
    upstream = FakeUpstream()
    upstream.reply = json_reply(200, {'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': camp.MODEL,
                                      'content': [{'type': 'text', 'text': 'ok'}], 'stop_reason': 'end_turn',
                                      'usage': {'input_tokens': 1000, 'output_tokens': 100}})
    monkeypatch.setenv('ANTHROPIC_API_KEY', REAL_KEY)
    monkeypatch.setattr(camp, 'agent_environment', lambda: {
        **AGENT_ENVIRONMENT(), 'UPSTREAM_PORT': upstream.url.rsplit(':', 1)[1]})
    try:
        summary = camp.run(root, live=True, agent=['/usr/bin/python3', '-I', '/workspace/agent.py'],
                           launch=FakeLauncher(), pool_dir=tmp_path / 'pool', upstream=upstream.url)
    finally:
        upstream.close()
    lines = (root / 'transcript.jsonl').read_text().splitlines()
    found = json.loads(lines[0])
    assert found['served'][0] == 200 and found['served'][1]['content'][0]['text'] == 'ok'
    assert found['other'] == 400 and found['key'] == camp.PLACEHOLDER_KEY and lines[1] == 'upstream unreachable'
    assert 'ANTHROPIC_API_KEY' not in os.environ and protected == [True]
    headers = dict(upstream.seen[0]['headers'])
    assert headers['x-api-key'] == REAL_KEY and len(upstream.seen) == 1
    assert summary['llm_calls'] == 1 and summary['cost_usd'] == pytest.approx((1000 * 4 + 100 * 20) / 1e6)
    assert BudgetPool(tmp_path / 'pool').status()['spent_usd'] == pytest.approx(summary['cost_usd'])
    assert REAL_KEY not in (root / 'relay' / 'ledger.jsonl').read_text()
    assert not (root / 'sockets' / 'relay.sock').exists()


AGENT_ENVIRONMENT = camp.agent_environment


@pytest.fixture
def protected(monkeypatch):
    from tasks.dual_arm_crag_skull.baselines.dexagent import dexenv
    calls = []
    monkeypatch.setattr(dexenv, 'protect_process', lambda: calls.append(True))
    return calls


def test_the_campaign_protocol_comes_from_the_config_and_stays_out_of_held_out_seeds():
    import copy
    dev, unseen, budget = camp.campaign_protocol(CONFIG)
    assert dev == [101, 102, 103, 104, 105] and unseen == [126, 127, 128, 129, 130]
    assert budget == {'max_fix_versions': 8, 'max_probes': 12, 'max_reruns': 2}
    for change, words in ((('dev_seeds', [99, 103]), 'practice seeds'), (('dev_seeds', [1, 5]), 'practice seeds'),
                          (('unseen_split', 'heldout'), 'selection seeds'), (('unseen_split', 'practice'), 'selection'),
                          (('max_probes', -1), 'whole numbers'), (('max_fix_versions', True), 'whole numbers')):
        config = copy.deepcopy(CONFIG)
        config['baselines']['aspire']['campaign'][change[0]] = change[1]
        with pytest.raises(ValueError, match=words):
            camp.campaign_protocol(config)


@needs_inputs
@needs_bwrap
def test_a_stopped_campaign_resumes_with_its_ledger_and_session(tmp_path):
    root = prepared(tmp_path)
    workspace = root / 'workspace'
    (workspace / 'first.sh').write_text('mkdir -p code && echo \'result = "win"\' > code/v00_initial.py && '
                                        'aspire-trial sweep code/v00_initial.py > /dev/null && echo first done\n')
    (workspace / 'second.sh').write_text('cat; aspire-trial status | tail -1\n')
    camp.run(root, live=False, agent=['/bin/bash', '/workspace/first.sh'], launch=FakeLauncher())
    (root / 'debug' / 'attempts' / 'v01').mkdir(parents=True)  # what a sweep cut off by a stop leaves
    summary = camp.run(root, live=False, agent=['/bin/bash', '/workspace/second.sh'], launch=FakeLauncher(),
                       resume=True)
    lines = (root / 'transcript.jsonl').read_text().splitlines()
    marker = json.loads(next(line for line in lines if '"resumed"' in line))
    assert lines[0] == 'first done' and marker['set_aside'][0].startswith('interrupted/attempts/v01')
    assert camp.RESUME_PROMPT in (root / 'transcript.jsonl').read_text()  # the second agent read the resume note
    assert json.loads(lines[-1])['versions_swept'] == ['v00'] and summary['versions'] == ['v00']
    assert len(trials_rows(root)) == 1 and (root / 'resumes.jsonl').read_text().count('previous_exit') == 1
    with pytest.raises(FileExistsError):
        camp.run(root, live=False, agent=['/bin/true'], launch=FakeLauncher())


def trials_rows(root):
    return [json.loads(line) for line in (root / 'service' / 'ledger.jsonl').read_text().splitlines()]


@needs_inputs
def test_resuming_claude_code_continues_its_session_and_refuses_a_running_campaign(tmp_path):
    import socket
    root = prepared(tmp_path)
    campaign = json.loads((root / 'campaign.json').read_text())
    assert camp.claude_arguments(campaign, 'abc')[-2:] == ['--resume', 'abc']
    assert '--resume' not in camp.claude_arguments(campaign)
    (root / 'transcript.jsonl').write_text(json.dumps({'type': 'system', 'subtype': 'init', 'session_id': 's-1'})
                                           + '\n')
    assert camp.session_id(root) == 's-1'
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(root / 'sockets' / 'trials.sock'))
    listener.listen(1)
    try:
        with pytest.raises(RuntimeError, match='still running'):
            camp.run(root, live=False, agent=['/bin/true'], launch=FakeLauncher(), resume=True)
    finally:
        listener.close()
    command = camp.agent_command(root, campaign, resume_session='s-1')
    script = command[command.index('-c') + 1]
    assert script.endswith('< /home/agent/resume_prompt.txt') and command[-2:] == ['--resume', 's-1']
