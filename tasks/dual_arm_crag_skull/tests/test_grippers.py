"""Gripper selection must never silently load the Robotiq scene for YAM."""
import json

import pytest

from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG, main
from tasks.dual_arm_crag_skull.grippers import selected_gripper, require_scene_gripper
from tasks.dual_arm_crag_skull.smoke import run


def test_selection_and_legacy_alias():
    config = json.loads(DEFAULT_CONFIG.read_text())
    assert selected_gripper(config) == 'yam'
    assert selected_gripper(config, 'robotiq') == 'robotiq'
    assert selected_gripper(config, allow_robotiq_proxy=True) == 'robotiq'
    with pytest.raises(ValueError, match='conflicts'):
        selected_gripper(config, 'yam', True)
    config['workcell']['gripper_family'] = 'Robotiq'
    assert selected_gripper(config) == 'robotiq'
    require_scene_gripper('robotiq')


def test_default_launch_refuses_missing_yam_before_scene_import(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['launch', '--assembly', 'unused.json'])
    assert exc.value.code == 2
    assert 'YAM is the default' in capsys.readouterr().err


def test_default_smoke_refuses_before_creating_output(tmp_path):
    output = tmp_path / 'smoke'
    with pytest.raises(ValueError, match='YAM is the default'):
        run('unused.json', output)
    assert not output.exists()


def test_inspect_reports_selected_gripper(capsys):
    main(['inspect', '--gripper', 'robotiq'])
    assert json.loads(capsys.readouterr().out)['selected_gripper'] == 'robotiq'
