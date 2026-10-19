import json

from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG
from tasks.dual_arm_crag_skull.s2.__main__ import load_observation, main, preview
from tasks.dual_arm_crag_skull.tests.s2_fakes import ANCHOR_WORLD


def write_observation(tmp_path):
    png = tmp_path / 'agent.png'
    png.write_bytes(b'\x89PNG\r\n\x1a\n' + bytes(64))
    data = {'observation_id': 'obs-5', 'time': 2.5,
            'fragments': {'part_0': {'world_T_part': ANCHOR_WORLD.tolist(), 'pose_source': 'icp', 'inlier_rmse_m': 4e-4},
                          'part_1': {'world_T_part': None, 'pose_source': None}},
            'verified_clusters': [], 'keyframes_png': {'agent': str(png)}}
    path = tmp_path / 'observation.json'
    path.write_text(json.dumps(data))
    return path


def test_preview_builds_request_without_client_and_elides_images(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    kwargs = preview(load_observation(write_observation(tmp_path)), config, ['pick'], with_garf=True)
    assert kwargs['model'] == 'claude-opus-5-5' and kwargs['fallbacks'] == 'default'
    assert [tool['name'] for tool in kwargs['tools']] == ['propose_assembly_candidates', 'submit_skill_request']
    assert kwargs['tools'][1]['input_schema']['properties']['skill_id']['enum'] == ['pick', 'reobserve']
    image = kwargs['messages'][0]['content'][1]
    assert image['source']['data'].startswith('<') and image['source']['data'].endswith('base64 chars>')
    facts = json.loads(kwargs['messages'][0]['content'][-1]['text'].split('\n', 1)[1])
    assert facts['observation']['fragments'][1] == {'fragment_id': 'part_1', 'localized': False, 'pose_source': None}


def test_cli_prints_json(tmp_path, capsys):
    main(['preview', '--observation', str(write_observation(tmp_path)), '--trained-skills', 'pick,place'])
    printed = json.loads(capsys.readouterr().out)
    assert printed['tool_choice'] == {'type': 'auto', 'disable_parallel_tool_use': True}
