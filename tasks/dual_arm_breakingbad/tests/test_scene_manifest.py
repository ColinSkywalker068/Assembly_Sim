from types import SimpleNamespace

from core.scene.validation import expected_workcell_manifest
from core.workcell.config import load_workcell_preset
from tasks.dual_arm_breakingbad.scene.assets import expected_assembly_manifest


def test_core_manifest_contains_only_invariant_workcell_paths():
    manifest = expected_workcell_manifest(load_workcell_preset("dual_arm"))

    assert manifest.floor_path == "/World/Environment/Floor"
    assert manifest.table_path == "/World/Environment/Table"
    assert manifest.assembly_pad_path == "/World/Environment/AssemblyPad"
    assert manifest.robot_paths == ("/World/Robots/Left", "/World/Robots/Right")
    assert manifest.camera_paths == (
        "/World/Cameras/Agent",
        "/World/Robots/Left/flange/WristCamera",
        "/World/Robots/Right/flange/WristCamera",
    )


def test_task_manifest_adds_arbitrary_generated_fragment_paths():
    workcell = load_workcell_preset("dual_arm")
    layout = SimpleNamespace(fragment_names=("alpha", "beta", "gamma"))

    manifest = expected_assembly_manifest(workcell, layout)

    assert manifest.workcell == expected_workcell_manifest(workcell)
    assert manifest.fragment_paths == (
        "/World/Fragments/alpha",
        "/World/Fragments/beta",
        "/World/Fragments/gamma",
    )
