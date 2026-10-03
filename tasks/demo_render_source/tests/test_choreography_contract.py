from pathlib import Path

from core.workcell.config import load_workcell_preset
from tasks.demo_render_source.choreography import choreography_workcell_metadata


def test_choreography_uses_core_dual_arm_assets_and_placements(tmp_path):
    probe = tmp_path / "fanuc_probe.json"
    metadata = choreography_workcell_metadata(probe)
    config = load_workcell_preset("dual_arm")

    assert metadata["assets"] == {
        "arm_usd": str(config.asset_path("arm_usd")),
        "gripper_usd": str(config.asset_path("gripper_usd")),
        "probe_json": str(probe.resolve()),
    }
    assert metadata["arms"] == [
        {
            "base_pos": list(config.robot("left").base_position),
            "base_quat": list(config.robot("left").base_orientation_wxyz),
        },
        {
            "base_pos": list(config.robot("right").base_position),
            "base_quat": list(config.robot("right").base_orientation_wxyz),
        },
    ]


def test_choreography_source_has_no_hard_coded_scratch_asset_root():
    task_root = Path(__file__).resolve().parents[1]
    source = (task_root / "_choreography_legacy.py").read_text(encoding="utf-8")

    assert "SCRATCH" not in source
    assert "/tmp/claude" not in source
    assert "choreography_workcell_metadata" in source
