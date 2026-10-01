import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCENE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SCENE_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scene_config import SceneConfig


CONFIG_PATH = SCENE_ROOT / "config" / "scene.json"


def write_v2_layout(root, names=("alpha", "beta", "gamma"), mesh_names=None):
    root.mkdir(parents=True, exist_ok=True)
    mesh_names = mesh_names or tuple(f"asset_{index}.npz" for index in range(len(names)))
    pieces = []
    for index, (name, mesh_name) in enumerate(zip(names, mesh_names)):
        np.savez(
            root / mesh_name,
            v=np.asarray([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0.1]], dtype=np.float32),
            f=np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        pieces.append(
            {
                "name": name,
                "source_name": f"piece_{index}.obj",
                "mesh": mesh_name,
                "color": [0.1, 0.2, 0.3],
                "cells": [[index, 0, 0]],
                "voxel_count": 1,
                "local_pivot_cells": [index + 0.5, 0.5, 0.0],
                "local_bounds": [[-0.05, -0.05, 0.0], [0.05, 0.05, 0.1]],
                "goal_pose": {"position": [index * 0.1, 0.0, 0.0], "orientation_wxyz": [1, 0, 0, 0]},
                "staging_pose": {"position": [index * 0.2, 0.2, 0.75], "orientation_wxyz": [1, 0, 0, 0]},
                "grasp": None,
            }
        )
    data = {
        "schema_version": 2,
        "object_id": "fixture",
        "source": {"loader": "test", "path": "/raw/no-longer-needed"},
        "processing": {"pitch": 0.1},
        "normalization": {
            "source_to_canonical": np.eye(4).tolist(),
            "grid_origin": [0, 0, 0],
            "grid_shape": [len(names), 1, 1],
            "assembly_bounds": [[0, 0, 0], [len(names) * 0.1, 0.1, 0.1]],
        },
        "fragment_count": len(names),
        "pieces": pieces,
        "staging": {"algorithm": "test"},
    }
    path = root / "layout.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_required_scene_contract():
    config = SceneConfig.load(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.physics_dt == pytest.approx(1 / 120)
    assert config.render_dt == pytest.approx(1 / 30)
    assert config.camera_resolution == (640, 480)
    assert config.fragment_names == tuple(f"piece_{index}" for index in range(8))


def test_dual_robot_contract_is_left_then_right():
    config = SceneConfig.load(CONFIG_PATH)

    assert config.robot_names == ("left", "right")
    assert config.robot_spec("left")["prim_path"] == "/World/Robots/Left"
    assert config.robot_spec("right")["prim_path"] == "/World/Robots/Right"


def test_demo_arm_and_fragment_transforms_are_exact():
    config = SceneConfig.load(CONFIG_PATH)

    assert config.robot_spec("left")["base_position"] == [-0.78, 0.0, 0.75]
    assert config.robot_spec("left")["base_orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert config.robot_spec("right")["base_position"] == [0.78, 0.0, 0.75]
    assert config.robot_spec("right")["base_orientation_wxyz"] == [0.0, 0.0, 0.0, 1.0]
    expected_positions = {
        "piece_0": [-0.42, -0.234, 0.75],
        "piece_1": [0.42, 0.236, 0.734],
        "piece_2": [-0.42, 0.242, 0.75],
        "piece_3": [-0.42, -0.062, 0.654],
        "piece_4": [0.42, -0.188, 0.75],
        "piece_5": [-0.42, 0.086, 0.75],
        "piece_6": [0.42, 0.048, 0.622],
        "piece_7": [-0.22, 0.46, 0.75],
    }
    assert {
        name: pose["position"]
        for name, pose in config.data["fragments"]["initial_poses"].items()
    } == expected_positions
    assert all(
        pose["orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
        for pose in config.data["fragments"]["initial_poses"].values()
    )


def test_plate_origin_matches_demo_assembly_frame():
    config = SceneConfig.load(CONFIG_PATH)
    layout = json.loads(config.resolve_repo_path("layout_json").read_text(encoding="utf-8"))
    extent_x, extent_y, _ = layout["assembly_extent"]
    table_position = config.data["environment"]["table_position"]
    table_size = config.data["environment"]["table_size"]
    table_top = table_position[2] + table_size[2] / 2

    assert config.data["environment"]["plate_position"] == pytest.approx(
        [-extent_x / 2, 0.10 - extent_y / 2, table_top]
    )


@pytest.mark.parametrize("robot_names", [("left",), ("left", "right", "spare")])
def test_rejects_missing_or_extra_robot_names(tmp_path, robot_names):
    source = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    shared = dict(source["robots"]["left"])
    source["robots"] = {name: dict(shared) for name in robot_names}
    config_path = tmp_path / "repository" / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (config_path.parents[2] / "02_robot_assets").mkdir()
    (config_path.parents[2] / "README.md").write_text("repository", encoding="utf-8")
    config_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly left and right"):
        SceneConfig.load(config_path)


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


def test_schema_v2_override_accepts_three_arbitrary_fragments(tmp_path):
    assembly_path = write_v2_layout(tmp_path / "assembly")

    config = SceneConfig.load(CONFIG_PATH, assembly_path=assembly_path)

    assert config.fragment_names == ("alpha", "beta", "gamma")
    assert config.has_support_surface is False


def test_override_layout_is_authoritative_over_legacy_fragment_config(tmp_path):
    assembly_path = write_v2_layout(tmp_path / "assembly", names=("only_one",))

    config = SceneConfig.load(CONFIG_PATH, assembly_path=assembly_path)

    assert config.fragment_names == ("only_one",)
    assert config.fragment_initial_pose("only_one")["position"] == [0.0, 0.2, 0.75]


def test_malformed_override_never_falls_back_to_skull(tmp_path):
    assembly_path = write_v2_layout(tmp_path / "assembly")
    data = json.loads(assembly_path.read_text(encoding="utf-8"))
    del data["pieces"][0]["staging_pose"]
    assembly_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="staging_pose"):
        SceneConfig.load(CONFIG_PATH, assembly_path=assembly_path)


def test_fragment_mesh_uses_explicit_layout_relative_path(tmp_path):
    assembly_path = write_v2_layout(
        tmp_path / "assembly", names=("unusual",), mesh_names=("not-the-fragment-name.npz",)
    )

    config = SceneConfig.load(CONFIG_PATH, assembly_path=assembly_path)

    assert config.fragment_mesh_path("unusual") == assembly_path.parent / "not-the-fragment-name.npz"


def test_legacy_skull_config_still_loads_eight_fragments():
    config = SceneConfig.load(CONFIG_PATH)

    assert config.fragment_names == tuple(f"piece_{index}" for index in range(8))
    assert config.has_support_surface is True
