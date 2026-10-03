from dataclasses import replace

import pytest

from tasks.stack_sawtooth_shulker.config import (
    Pose,
    load_task_config,
    validate_task_config,
)
from tasks.stack_sawtooth_shulker.geometry import fragment_bounds, fragment_mass


IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _replace_fragment(config, name, **changes):
    fragments = tuple(
        replace(fragment, **changes) if fragment.name == name else fragment
        for fragment in config.fragments
    )
    return replace(config, fragments=fragments)


def test_default_configuration_matches_the_single_right_workcell_and_task_scale():
    config = load_task_config()

    assert config.workcell.preset_name == "single_arm_right"
    assert config.workcell.robot_names == ("right",)
    assert config.unit_size == pytest.approx(0.08)
    assert config.target_center == pytest.approx((0.0, 0.0, 0.75))
    assert config.target_size == pytest.approx((0.55, 0.55))
    assert config.table_top_z == pytest.approx(0.75)
    assert config.collider_clearance == pytest.approx(0.002)
    assert config.density == pytest.approx(20.0)
    assert config.robot_base_near_x == pytest.approx(0.685)


def test_default_fragments_have_distinct_colors_and_exact_poses():
    config = load_task_config()
    a, b = config.fragments

    assert (a.name, b.name) == ("FragmentA", "FragmentB")
    assert a.color == pytest.approx((0.08, 0.28, 0.85))
    assert b.color == pytest.approx((0.12, 0.65, 0.25))
    assert a.initial_pose == Pose((0.48, -0.185, 0.75), IDENTITY)
    assert b.initial_pose == Pose((0.48, 0.185, 0.75), IDENTITY)
    assert a.goal_pose == Pose((0.0, 0.0, 0.75), IDENTITY)
    assert b.goal_pose == Pose((0.0, 0.0, 0.83), IDENTITY)
    assert a.canonical_z_offset == 0
    assert b.canonical_z_offset == 1


def test_default_bounds_masses_and_static_clearances_are_exact():
    config = load_task_config()
    a, b = config.fragments

    a_min, a_max = fragment_bounds(a, config.unit_size)
    b_min, b_max = fragment_bounds(b, config.unit_size)
    assert a_min == pytest.approx((-0.12, -0.12, 0.0))
    assert a_max == pytest.approx((0.12, 0.12, 0.16))
    assert b_min == pytest.approx((-0.12, -0.12, 0.0))
    assert b_max == pytest.approx((0.12, 0.12, 0.16))
    assert fragment_mass(a, config.unit_size, config.density) == pytest.approx(
        0.14336
    )
    assert fragment_mass(b, config.unit_size, config.density) == pytest.approx(
        0.13312
    )

    fragment_min_x = a.initial_pose.position[0] - 0.12
    fragment_max_x = a.initial_pose.position[0] + 0.12
    a_max_y = a.initial_pose.position[1] + 0.12
    b_min_y = b.initial_pose.position[1] - 0.12
    assert fragment_min_x - 0.275 == pytest.approx(0.085)
    assert config.robot_base_near_x - fragment_max_x == pytest.approx(0.085)
    assert b_min_y - a_max_y == pytest.approx(0.13)


def test_repeated_default_loads_are_deterministic():
    assert load_task_config() == load_task_config()


@pytest.mark.parametrize(
    "target_size",
    ((0.55, 0.50), (0.23, 0.23)),
)
def test_validation_requires_a_square_target_that_contains_the_assembled_cube(target_size):
    config = replace(load_task_config(), target_size=target_size)

    with pytest.raises(ValueError, match="target"):
        validate_task_config(config)


@pytest.mark.parametrize("clearance", (-0.001, float("nan"), float("inf"), 0.08))
def test_validation_rejects_invalid_collider_clearance(clearance):
    config = replace(load_task_config(), collider_clearance=clearance)

    with pytest.raises(ValueError, match="clearance"):
        validate_task_config(config)


def test_zero_collider_clearance_is_valid():
    config = replace(load_task_config(), collider_clearance=0.0)

    assert validate_task_config(config) is None


@pytest.mark.parametrize("density", (0.0, -1.0, float("nan"), float("inf")))
def test_validation_rejects_nonpositive_or_nonfinite_density(density):
    config = replace(load_task_config(), density=density)

    with pytest.raises(ValueError, match="density"):
        validate_task_config(config)


@pytest.mark.parametrize(
    "cells",
    (
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 0),),
        ((3, 0, 0),),
    ),
)
def test_validation_rejects_duplicate_incomplete_or_out_of_range_cells(cells):
    config = _replace_fragment(load_task_config(), "FragmentA", canonical_cells=cells)

    with pytest.raises(ValueError, match="cell|partition|mapping"):
        validate_task_config(config)


def test_validation_rejects_cells_owned_by_both_fragments():
    config = load_task_config()
    b = config.fragments[1]
    overlapping = ((0, 0, 0), *b.canonical_cells[1:])
    config = _replace_fragment(config, "FragmentB", canonical_cells=overlapping)

    with pytest.raises(ValueError, match="overlap|partition|mapping"):
        validate_task_config(config)


@pytest.mark.parametrize(
    ("name", "position", "message"),
    (
        ("FragmentB", (0.48, -0.185, 0.75), "overlap"),
        ("FragmentA", (0.40, -0.185, 0.75), "target"),
        ("FragmentA", (0.48, -0.60, 0.75), "table"),
        ("FragmentA", (0.55, -0.185, 0.75), "robot"),
    ),
)
def test_validation_rejects_invalid_initial_placement(name, position, message):
    config = load_task_config()
    fragment = next(fragment for fragment in config.fragments if fragment.name == name)
    config = _replace_fragment(
        config,
        name,
        initial_pose=replace(fragment.initial_pose, position=position),
    )

    with pytest.raises(ValueError, match=message):
        validate_task_config(config)
