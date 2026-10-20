import numpy as np
import pytest

from tasks.dual_arm_crag_skull.experiments.assembly_paths import disassemble, free_directions
from tasks.dual_arm_crag_skull.interlock_repair import straighten


def arrays(owned):
    return {name: np.asarray(cells, dtype=int) for name, cells in owned.items()}


HELD = {'part': [(0, 0, 0)],
        'anchor': [(0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 1), (-1, -1, 1), (1, -1, 1),
                   (-1, 1, 1), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 0), (-1, -1, 0), (1, -1, 0),
                   (-1, 1, 0)]}


def test_a_held_part_takes_the_cell_over_it_and_then_comes_out():
    assert free_directions(arrays(HELD), 'part', set(HELD)) == []
    owned, report = straighten(HELD, 'anchor')
    assert set().union(*map(set, owned.values())) == set().union(*map(set, HELD.values()))  # nothing added or lost
    assert sum(map(len, owned.values())) == sum(map(len, HELD.values()))
    assert disassemble(arrays(owned), 'anchor')[1] == []
    assert len(owned['part']) == 2 and report['flipped'] == 1  # it took the cube above it rather than vanish
    assert report['order'] == [{'fragment': 'part', 'direction': [0, 0, 1], 'flipped': 1}]


def test_an_assembly_that_comes_apart_is_left_alone():
    stack = {'base': [(0, 0, 0), (1, 0, 0)], 'middle': [(0, 0, 1), (1, 0, 1)], 'top': [(0, 0, 2)]}
    owned, report = straighten(stack, 'base')
    assert {n: sorted(c) for n, c in owned.items()} == {n: sorted(c) for n, c in stack.items()}
    assert report['flipped'] == 0 and len(report['order']) == 2


def test_a_cell_under_a_tall_overhang_goes_to_the_overhang_instead():
    # the part's left cube has three anchor cubes above it and two-thick walls on its sides: giving that cube away
    # (one flip) is cheaper than taking the column above it (three) or a wall (two)
    owned = {'part': [(0, 0, 0), (1, 0, 0)],
             'anchor': [(0, 0, 1), (0, 0, 2), (0, 0, 3), (-1, 0, 0), (-2, 0, 0), (2, 0, 0), (3, 0, 0),
                        (0, 1, 0), (0, -1, 0), (1, 1, 0), (1, -1, 0)]}
    assert free_directions(arrays(owned), 'part', set(owned)) == []
    owned_after, report = straighten(owned, 'anchor')
    assert sorted(owned_after['part']) == [(1, 0, 0)] and (0, 0, 0) in owned_after['anchor']
    assert report['changed'] == {'part': {'lost': 1, 'gained': 0}, 'anchor': {'lost': 0, 'gained': 1}}
    assert disassemble(arrays(owned_after), 'anchor')[1] == []


def test_an_unknown_anchor_is_refused():
    with pytest.raises(ValueError, match='anchor'):
        straighten(HELD, 'base')


def test_a_fragment_is_not_cut_in_two_when_another_way_out_costs_the_same():
    # straight up costs one flip but cuts the bar in two (its middle cube goes to the column above it); sideways
    # costs one flip too (it takes the wall cube at its end) and keeps it whole, even though giving the middle away
    # would join the anchor's column to its walls and leave fewer pieces in all
    owned = {'bar': [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
             'anchor': [(1, 0, 1), (1, 0, 2), (1, 0, 3), (-1, 0, 0), (3, 0, 0)]
                       + [(x, y, 0) for x in range(3) for y in (-1, 1)]}
    owned_after, report = straighten(owned, 'anchor')
    assert report['order'] == [{'fragment': 'bar', 'direction': [1, 0, 0], 'flipped': 1}]
    assert sorted(owned_after['bar']) == [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]


def test_a_stray_cube_joins_a_neighbour_that_keeps_the_order_clear():
    # p's cube at (1, 0, 2) touches nothing of p; the anchor touches it most, but owning it would cover q's way up,
    # so it goes to q instead
    owned = {'anchor': [(0, 0, 2), (1, 1, 2), (1, -1, 2)],
             'q': [(1, 0, 1), (2, 0, 1), (2, 0, 2)],
             'p': [(5, 0, 5), (5, 0, 6), (1, 0, 2)]}
    owned_after, report = straighten(owned, 'anchor')
    assert sorted(owned_after['p']) == [(5, 0, 5), (5, 0, 6)] and (1, 0, 2) in owned_after['q']
    assert report['islands'] == [{'fragment': 'p', 'cells': 1, 'to': 'q'}]
    assert disassemble(arrays(owned_after), 'anchor')[1] == []


def test_a_stray_cube_no_neighbour_can_take_stays():
    owned = {'anchor': [(0, 0, 0)], 'p': [(5, 0, 0), (9, 0, 0)]}
    owned_after, report = straighten(owned, 'anchor')
    assert sorted(owned_after['p']) == [(5, 0, 0), (9, 0, 0)] and report['islands'] == []


def layers(*grids):
    """{name: cells} from z layers bottom up, each rows of y with one letter per x."""
    owned = {}
    for z, grid in enumerate(grids):
        for y, row in enumerate(grid.split('/')):
            for x, name in enumerate(row):
                owned.setdefault(name, []).append((x, y, z))
    return owned


def test_the_cheapest_first_exit_is_not_taken_when_another_start_costs_less_in_all():
    # p is the cube in the middle of the block: it can come straight out first by taking the anchor cube over it, but
    # q still has to change a cube after it (two in all); q coming out first (one change) opens p's way along +y
    owned = layers('qqq/qqq/qqq', 'aaq/apa/aqa', 'aaa/aaa/qqa')
    owned_after, report = straighten(owned, 'a')
    assert report['flipped'] == 1 and [step['fragment'] for step in report['order']] == ['q', 'p']
    assert disassemble(arrays(owned_after), 'a')[1] == []
