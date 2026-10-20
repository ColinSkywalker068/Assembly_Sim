import numpy as np

from tasks.dual_arm_crag_skull.experiments.assembly_paths import DIRECTIONS, blocked, disassemble, free_directions


def cubes(*cells):
    return np.asarray(cells, dtype=int)


def test_a_neighbour_blocks_the_way_towards_it_only():
    moving, fixed = cubes((0, 0, 0)), cubes((1, 0, 0))
    assert blocked(moving, fixed, (1, 0, 0)) and blocked(moving, cubes((3, 0, 0)), (1, 0, 0))  # however far ahead
    assert not blocked(moving, fixed, (-1, 0, 0)) and not blocked(moving, fixed, (0, 1, 0))
    assert not blocked(moving, fixed, (0, 0, 1)) and not blocked(moving, cubes((1, 1, 0)), (1, 0, 0))  # other row


def test_a_diagonal_move_is_blocked_by_the_cubes_it_sweeps_through():
    moving = cubes((0, 0, 0))
    assert blocked(moving, cubes((1, 1, 0)), (1, 1, 0)) and blocked(moving, cubes((0, 1, 0)), (1, 1, 0))
    assert not blocked(moving, cubes((2, 0, 0)), (1, 1, 0))  # by the time x reaches it, y has passed it


def test_a_cube_held_from_above_and_on_every_side_has_no_way_out():
    cells = {'part': cubes((0, 0, 0)),
             'anchor': cubes((0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 1), (-1, -1, 1),
                             (1, -1, 1), (-1, 1, 1), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                             (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0))}
    assert free_directions(cells, 'part', {'part', 'anchor'}) == []
    assert disassemble(cells, 'anchor') == ([], ['part'])
    assert all(d[2] >= 0 for d in DIRECTIONS) and len(DIRECTIONS) == 17  # nothing goes into the table


def test_a_stack_comes_apart_and_a_part_under_a_lid_waits_for_it():
    cells = {'base': cubes((0, 0, 0), (1, 0, 0)), 'middle': cubes((0, 0, 1), (1, 0, 1)), 'top': cubes((0, 0, 2))}
    order, stuck = disassemble(cells, 'base')
    assert stuck == [] and {name for name, _ in order} == {'middle', 'top'}  # only collisions count, not gravity
    walls = [(x, y, z) for x in (-1, 1) for y in (-1, 0, 1) for z in (0, 1)] + [(0, y, z) for y in (-1, 1) for z in (0, 1)]
    cells = {'cup': cubes(*walls), 'part': cubes((0, 0, 0)), 'lid': cubes((0, 0, 1), (0, 0, 2))}
    order, stuck = disassemble(cells, 'cup')
    assert stuck == [] and [name for name, _ in order] == ['lid', 'part']  # the lid must go first
    assert free_directions(cells, 'part', set(cells)) == [] and free_directions(cells, 'part', {'cup', 'part'})
