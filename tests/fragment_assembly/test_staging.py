import pytest

from fragment_assembly.staging import (
    AABB2D,
    PieceBounds,
    StagingSpec,
    TableBounds,
    compute_staging_poses,
)


def _spec(table=None, exclusions=()):
    return StagingSpec(
        table=table or TableBounds(-0.8, 0.8, -0.5, 0.5, 0.75),
        exclusions=tuple(exclusions),
        gap=0.04,
        grid_step=0.01,
    )


def _pieces():
    return {
        "shard_z": PieceBounds((-0.10, -0.05, 0.02), (0.10, 0.05, 0.18)),
        "fragment_alpha": PieceBounds((-0.06, -0.08, 0.00), (0.06, 0.08, 0.12)),
        "part_17": PieceBounds((-0.04, -0.04, 0.05), (0.04, 0.04, 0.11)),
    }


def test_staging_is_deterministic_for_arbitrary_names():
    first = compute_staging_poses(_pieces(), _spec())
    second = compute_staging_poses(dict(reversed(tuple(_pieces().items()))), _spec())

    assert first == second
    assert set(first) == {"shard_z", "fragment_alpha", "part_17"}


def test_staged_aabbs_are_separated_and_inside_table():
    pieces = _pieces()
    spec = _spec()
    poses = compute_staging_poses(pieces, spec)
    world = {name: pieces[name].world_aabb(pose) for name, pose in poses.items()}

    assert all(spec.table.contains(aabb) for aabb in world.values())
    for index, first in enumerate(sorted(world)):
        for second in sorted(world)[index + 1 :]:
            assert not world[first].expanded(spec.gap / 2).overlaps(
                world[second].expanded(spec.gap / 2)
            )


def test_lowest_voxel_face_touches_table():
    pieces = _pieces()
    spec = _spec()

    poses = compute_staging_poses(pieces, spec)

    for name, pose in poses.items():
        assert pose.position[2] + pieces[name].local_min[2] == pytest.approx(0.75)


def test_staging_avoids_both_robot_bases():
    exclusions = (
        AABB2D.from_center_extent((-0.55, 0.0), (0.34, 0.34)),
        AABB2D.from_center_extent((0.55, 0.0), (0.34, 0.34)),
    )
    pieces = _pieces()
    spec = _spec(exclusions=exclusions)

    poses = compute_staging_poses(pieces, spec)

    for name, pose in poses.items():
        aabb = pieces[name].world_aabb(pose).expanded(spec.gap)
        assert all(not aabb.overlaps(exclusion) for exclusion in exclusions)


def test_capacity_failure_names_first_unplaced_piece():
    pieces = {
        "large_a": PieceBounds((0.0, 0.0, 0.0), (0.18, 0.18, 0.10)),
        "large_b": PieceBounds((0.0, 0.0, 0.0), (0.18, 0.18, 0.10)),
    }
    spec = _spec(table=TableBounds(0.0, 0.2, 0.0, 0.2, 0.75))

    with pytest.raises(ValueError, match="could not place large_b.*table capacity"):
        compute_staging_poses(pieces, spec)


def test_bilateral_staging_uses_two_front_to_back_lanes_for_three_fragments():
    pieces = {
        "piece_0": PieceBounds((-0.10, -0.06, 0.00), (0.10, 0.06, 0.14)),
        "piece_1": PieceBounds((-0.08, -0.05, 0.02), (0.08, 0.05, 0.12)),
        "piece_2": PieceBounds((-0.05, -0.04, 0.00), (0.05, 0.04, 0.10)),
    }
    pad = AABB2D(-0.256, 0.256, -0.108, 0.308)
    spec = StagingSpec(
        table=TableBounds(-0.97, 0.97, -0.57, 0.57, 0.75),
        exclusions=(pad,),
        pad=pad,
        lane_centers_x=(-0.420, 0.420),
        gap=0.04,
        grid_step=0.01,
    )

    poses = compute_staging_poses(pieces, spec)
    world = {name: pieces[name].world_aabb(pose) for name, pose in poses.items()}

    assert world["piece_0"].center[0] == pytest.approx(-0.420)
    assert world["piece_1"].center[0] == pytest.approx(-0.420)
    assert world["piece_2"].center[0] == pytest.approx(0.420)
    assert world["piece_0"].center[1] != world["piece_1"].center[1]
    assert all(spec.table.contains(aabb) for aabb in world.values())
    assert all(not aabb.expanded(spec.gap).overlaps(pad) for aabb in world.values())
    assert all(
        pose.position[2] + pieces[name].local_min[2] == pytest.approx(0.75)
        for name, pose in poses.items()
    )


def test_bilateral_staging_reports_the_first_piece_that_exceeds_lane_capacity():
    pieces = {
        "piece_0": PieceBounds((-0.08, -0.09, 0.00), (0.08, 0.09, 0.10)),
        "piece_1": PieceBounds((-0.08, -0.09, 0.00), (0.08, 0.09, 0.10)),
        "piece_2": PieceBounds((-0.08, -0.09, 0.00), (0.08, 0.09, 0.10)),
    }
    pad = AABB2D(-0.05, 0.05, -0.05, 0.05)
    spec = StagingSpec(
        table=TableBounds(-0.50, 0.50, -0.14, 0.14, 0.75),
        exclusions=(pad,),
        pad=pad,
        lane_centers_x=(-0.30, 0.30),
        gap=0.04,
        grid_step=0.01,
    )

    with pytest.raises(ValueError, match="piece_1.*lane capacity"):
        compute_staging_poses(pieces, spec)


def test_bilateral_staging_moves_a_wide_lane_outward_to_preserve_pad_clearance():
    pieces = {
        "piece_0": PieceBounds((-0.128, -0.05, 0.00), (0.128, 0.05, 0.10)),
    }
    pad = AABB2D(-0.256, 0.256, -0.108, 0.308)
    spec = StagingSpec(
        table=TableBounds(-0.97, 0.97, -0.57, 0.57, 0.75),
        exclusions=(pad,),
        pad=pad,
        lane_centers_x=(-0.420, 0.420),
        gap=0.04,
        grid_step=0.01,
    )

    pose = compute_staging_poses(pieces, spec)["piece_0"]
    world = pieces["piece_0"].world_aabb(pose)

    assert world.center[0] == pytest.approx(-0.424)
    assert not world.expanded(spec.gap).overlaps(pad)


def test_bilateral_staging_centers_each_lane_aggregate_on_the_table_y_axis():
    pieces = {
        "piece_0": PieceBounds((-0.10, -0.08, 0.00), (0.10, 0.08, 0.10)),
        "piece_1": PieceBounds((-0.08, -0.12, 0.00), (0.08, 0.12, 0.10)),
        "piece_2": PieceBounds((-0.05, -0.05, 0.00), (0.05, 0.05, 0.10)),
    }
    pad = AABB2D(-0.256, 0.256, -0.108, 0.308)
    spec = StagingSpec(
        table=TableBounds(-0.97, 0.97, -0.57, 0.57, 0.75),
        exclusions=(pad,),
        pad=pad,
        lane_centers_x=(-0.420, 0.420),
        gap=0.04,
        grid_step=0.01,
    )

    poses = compute_staging_poses(pieces, spec)
    world = {name: pieces[name].world_aabb(pose) for name, pose in poses.items()}
    left_min = min(world[name].min_y for name in ("piece_0", "piece_1"))
    left_max = max(world[name].max_y for name in ("piece_0", "piece_1"))

    assert (left_min + left_max) / 2 == pytest.approx(spec.table.center[1])
    assert world["piece_2"].center[1] == pytest.approx(spec.table.center[1])
