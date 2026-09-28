import json
import sys
from pathlib import Path

import pytest


SCENE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SCENE_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scene_config import SceneConfig


CONFIG_PATH = SCENE_ROOT / "config" / "scene.json"


def test_loads_required_scene_contract():
    config = SceneConfig.load(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.physics_dt == pytest.approx(1 / 120)
    assert config.render_dt == pytest.approx(1 / 30)
    assert config.camera_resolution == (640, 480)
    assert config.fragment_names == tuple(f"piece_{index}" for index in range(8))


def test_paths_resolve_after_repo_relocation(tmp_path):
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    relocated = tmp_path / "repository with spaces"
    config_path = relocated / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (relocated / "02_robot_assets").mkdir()
    (relocated / "README.md").write_text("relocated repository", encoding="utf-8")
    config_path.write_text(json.dumps(source), encoding="utf-8")

    config = SceneConfig.load(config_path)

    expected = relocated / source["paths"]["arm_usd"]
    assert config.repo_root == relocated.resolve()
    assert config.resolve_repo_path("arm_usd") == expected.resolve()


def test_missing_asset_reports_field_and_resolved_path(tmp_path):
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    relocated = tmp_path / "repo"
    config_path = relocated / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (relocated / "02_robot_assets").mkdir()
    (relocated / "README.md").write_text("repository", encoding="utf-8")
    source["paths"]["arm_usd"] = "missing/arm.usd"
    config_path.write_text(json.dumps(source), encoding="utf-8")
    config = SceneConfig.load(config_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        config.validate_inputs()

    expected = str((relocated / "missing" / "arm.usd").resolve())
    assert "arm_usd" in str(exc_info.value)
    assert expected in str(exc_info.value)


def test_rejects_non_eight_fragment_contract(tmp_path):
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source["fragments"]["names"] = source["fragments"]["names"][:-1]
    config_path = tmp_path / "repository" / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (config_path.parents[2] / "02_robot_assets").mkdir()
    (config_path.parents[2] / "README.md").write_text("repository", encoding="utf-8")
    config_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly piece_0 through piece_7"):
        SceneConfig.load(config_path)
