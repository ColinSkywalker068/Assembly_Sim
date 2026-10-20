import numpy as np
import trimesh

from tasks.dual_arm_crag_skull.experiments.mesh_paths import blocking, orders, shrink, solid, upper_directions


def test_a_box_fills_exactly_the_cells_whose_centres_it_contains():
    box = trimesh.creation.box(extents=(10, 6, 4))  # faces at x = +-5, y = +-3, z = +-2, between cell centres
    inside, odd = solid(box, np.array([-6.0, -4.0, -3.0]), 1.0, (12, 8, 6))
    assert odd == 0 and inside.sum() == 10 * 6 * 4
    assert inside[1:11, 1:7, 1:5].all()


def test_a_sphere_is_filled_to_its_volume():
    ball = trimesh.creation.icosphere(subdivisions=4, radius=10)
    inside, odd = solid(ball, np.array([-11.0, -11.0, -11.0]), 0.5, (44, 44, 44))
    assert odd == 0 and abs(inside.sum() * 0.5 ** 3 / ball.volume - 1) < 0.01


def test_shrinking_peels_one_layer_including_at_the_array_edge():
    assert shrink(np.ones((5, 5, 5), bool), 1).sum() == 27


def cells(*boxes, step=0.25):
    """Points filling unit cells [lo, hi) of each box, `step` apart, and their labels."""
    points, labels = [], []
    for label, (lo, hi) in enumerate(boxes):
        axes = [np.arange(a + step / 2, b, step) for a, b in zip(lo, hi)]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        points.append(grid)
        labels.append(np.full(len(grid), label))
    return np.concatenate(points), np.concatenate(labels)


def test_a_fragment_is_blocked_only_by_what_lies_ahead_of_it():
    points, labels = cells(((0, 0, 0), (3, 3, 3)), ((0, 0, 5), (3, 3, 8)))
    up, side, down = (blocking(points, labels, 2, d, 0.5) for d in ((0, 0, 1), (1, 0, 0), (0, 0, -1)))
    assert up[0, 1] > 0 and up[1, 0] == 0
    assert not side.any() and down[1, 0] > 0 and down[0, 1] == 0


def test_points_in_different_tubes_never_block_each_other():
    # along +z the tubes are numbered by y and -x; these two sit at the far ends of neighbouring rows of tubes
    points, labels = cells(((0, 0, 0), (0.5, 0.5, 0.5)), ((3, 0.5, 2), (3.5, 1, 2.5)))
    assert not blocking(points, labels, 2, (0, 0, 1), 0.5).any()


def test_a_part_in_a_cup_under_a_lid_comes_out_only_after_the_lid():
    cup = [((0, 0, 0), (5, 5, 1)), ((0, 0, 1), (1, 5, 3)), ((4, 0, 1), (5, 5, 3)), ((1, 0, 1), (4, 1, 3)),
           ((1, 4, 1), (4, 5, 3))]
    part, lid = ((1, 1, 1), (4, 4, 3)), ((0, 0, 3), (5, 5, 4))
    points, labels = cells(*cup, part, lid)
    labels = np.where(labels < 5, 0, labels - 4)  # the cup's five boxes are one fragment: cup 0, part 1, lid 2
    dirs = upper_directions(400)
    counts = np.stack([blocking(points, labels, 3, d, 0.5) for d in dirs])
    assert orders(counts, dirs, anchor=0) == [[2, 1]]  # removal order: the lid, then the part
