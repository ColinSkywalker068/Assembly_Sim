import json
import shutil
import sys
from pathlib import Path

import pytest


SCENE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SCENE_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plan_layout import (
    AABB2D,
    check_reachability,
    compute_staging_poses,
    freeze_layout,
    load_piece_extents,
    scene_exclusions,
    table_bounds,
)
from scene_config import SceneConfig


CONFIG_PATH = SCENE_ROOT / "config" / "scene.json"


def test_staging_poses_are_stable_separated_and_inside_table():
    config = SceneConfig.load(CONFIG_PATH)
    extents = load_piece_extents(config)
    bounds = table_bounds(config)
    exclusions = scene_exclusions(config)
    gap = float(config.data["environment"]["staging_gap"])

    poses = compute_staging_poses(extents, bounds, exclusions, gap)

    assert set(poses) == set(config.fragment_names)
    aabbs = {}
    for name, pose in poses.items():
        extent = extents[name]
        assert pose.position[2] + extent.local_min[2] >= bounds.table_top_z - 1e-9
        aabb = extent.world_aabb(pose)
        aabbs[name] = aabb
        assert bounds.contains(aabb)
        assert all(not aabb.expanded(gap).overlaps(exclusion) for exclusion in exclusions)
    names = sorted(poses)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            assert not aabbs[first].expanded(gap / 2).overlaps(aabbs[second].expanded(gap / 2))


def test_reachability_names_nine_targets_with_joints_inside_limits():
    config = SceneConfig.load(CONFIG_PATH)
    extents = load_piece_extents(config)
    poses = compute_staging_poses(
        extents,
        table_bounds(config),
        scene_exclusions(config),
        float(config.data["environment"]["staging_gap"]),
    )

    report = check_reachability(config, poses)

    assert report.all_reachable
    assert set(report.targets) == {*config.fragment_names, "plate_center"}
    for target in report.targets.values():
        assert target.reachable
        assert target.position_error < 0.002
        assert target.rotation_error < 0.02
        assert all(low <= joint <= high for joint, low, high in zip(target.joints, report.lower_limits, report.upper_limits))


def test_reachability_works_from_repository_path_with_spaces(tmp_path):
    original = SceneConfig.load(CONFIG_PATH)
    relocated = tmp_path / "repository with spaces"
    config_path = relocated / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (relocated / "02_robot_assets").mkdir()
    (relocated / "README.md").write_text("repository", encoding="utf-8")
    for key in ("probe_json", "layout_json"):
        destination = relocated / original.data["paths"][key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original.resolve_repo_path(key), destination)
    config_path.write_text(json.dumps(original.data), encoding="utf-8")
    config = SceneConfig.load(config_path)
    extents = load_piece_extents(config)
    poses = compute_staging_poses(
        extents,
        table_bounds(config),
        scene_exclusions(config),
        float(config.data["environment"]["staging_gap"]),
    )

    report = check_reachability(config, poses)

    assert report.all_reachable
    assert Path(report.probe_path).is_relative_to(relocated)


def test_freeze_layout_writes_only_reachable_poses(tmp_path):
    original = SceneConfig.load(CONFIG_PATH)
    relocated = tmp_path / "repository"
    config_path = relocated / "06_interactive_scene" / "config" / "scene.json"
    config_path.parent.mkdir(parents=True)
    (relocated / "02_robot_assets").mkdir()
    (relocated / "README.md").write_text("repository", encoding="utf-8")
    for key in ("probe_json", "layout_json"):
        destination = relocated / original.data["paths"][key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original.resolve_repo_path(key), destination)
    config_path.write_text(json.dumps(original.data), encoding="utf-8")

    report = freeze_layout(config_path)
    frozen = json.loads(config_path.read_text(encoding="utf-8"))

    assert report.all_reachable
    assert set(frozen["fragments"]["initial_poses"]) == set(original.fragment_names)
    assert frozen["placement"]["reachability"]["all_reachable"] is True
    assert set(frozen["placement"]["reachability"]["targets"]) == {
        *original.fragment_names,
        "plate_center",
    }
