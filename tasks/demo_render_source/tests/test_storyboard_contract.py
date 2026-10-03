from pathlib import Path

from tasks.demo_render_source.storyboard import storyboard_workcell_config


def test_storyboard_selects_the_shared_dual_arm_workcell():
    config = storyboard_workcell_config()

    assert config.preset_name == "dual_arm"
    assert config.robot_names == ("left", "right")


def test_storyboard_worker_builds_core_before_task_owned_assets():
    source = (
        Path(__file__).resolve().parents[1] / "_storyboard_legacy.py"
    ).read_text(encoding="utf-8")

    assert "build_workcell(" in source
    assert "author_storyboard_assets(" in source
    assert source.index("build_workcell(") < source.index("author_storyboard_assets(")
    assert 'VisualCuboid("/World/table"' not in source
    assert 'VisualCuboid("/World/floor"' not in source
