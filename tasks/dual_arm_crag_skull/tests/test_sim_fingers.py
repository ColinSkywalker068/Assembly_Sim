"""The YAM fingers' inward reach from the tip meshes (sim/fingers.py): what the grasp planner knows of where the
fingers meet a fragment and what they sweep on the way down."""
import numpy as np
import pytest

from core.robots.linear_4310 import load_linear_4310
from tasks.dual_arm_crag_skull.sim.check_yam import BOX_SIDE_M, predicted_stop_width
from tasks.dual_arm_crag_skull.sim.fingers import yam_finger_reach

GRIPPER = load_linear_4310()
FINGERS = yam_finger_reach()


def lookup(table, heights, laterals):
    return FINGERS.at(table, np.asarray(heights, dtype=float), np.asarray(laterals, dtype=float))


def test_each_tip_is_on_its_own_side_of_the_closing_axis():
    assert sorted(FINGERS.side.values()) == [-1.0, 1.0]
    for tip, side in FINGERS.side.items():
        assert np.sign(GRIPPER.slide_axes[tip][0]) == side  # the TCP's closing axis is the mount's x


def test_the_pads_reach_just_past_the_middle_and_the_window_not_at_all():
    """At joint 0 the pads overlap 1.04 mm, so each reaches about 0.5 mm past the middle; 35 mm up on the middle line
    is the window, beside it the rails, about 1 mm behind the pad face."""
    for tip in FINGERS.side:
        pad = lookup(FINGERS.reach[tip], [0.004, 0.008, 0.012], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pad, -GRIPPER.closed_gap_m / 2, atol=1e-4)
        assert lookup(FINGERS.reach[tip], [0.035], [0.0])[0] == -np.inf
        rail = lookup(FINGERS.reach[tip], [0.035], [0.012])[0]
        assert np.isfinite(rail) and pad.min() - 0.002 < rail < pad.min() - 0.0005
        assert lookup(FINGERS.reach[tip], [-0.001, 0.2], [0.0, 0.0]).tolist() == [-np.inf, -np.inf]


def test_the_swept_reach_is_the_most_any_lower_section_reaches():
    for tip in FINGERS.side:
        reach, swept = FINGERS.reach[tip], FINGERS.swept[tip]
        assert (swept >= reach).all()
        assert (swept[1:] >= swept[:-1]).all()
        np.testing.assert_array_equal(swept[0], reach[0])
        # 35 mm up on the middle line nothing is there, but the pad below it came by on the way down
        assert lookup(swept, [0.035], [0.0])[0] == pytest.approx(lookup(reach, [0.008], [0.0])[0], abs=1e-6)


@pytest.mark.parametrize('depth, width', [(0.005, 0.040), (0.011, 0.040), (0.025, 0.040), (0.008, 0.020)])
def test_closing_on_a_box_stops_where_check_yam_predicts(depth, width):
    """The same mesh geometry, through the planner's grid: a box between the pads, 12 mm across."""
    stop = FINGERS.stop_width(*box_columns(depth, width, BOX_SIDE_M / 2))
    assert stop == pytest.approx(predicted_stop_width(GRIPPER, depth, width), abs=FINGERS.cell_m / 2)


def box_columns(depth, width, half):
    grid = np.arange(-half, half + 1e-9, FINGERS.cell_m / 2)
    h, u = (values.ravel() for values in np.meshgrid(depth + grid, grid, indexing='ij'))
    s = np.concatenate([np.full(len(h), width / 2), np.full(len(h), -width / 2)])
    return np.concatenate([h, h]), np.concatenate([u, u]), s


def test_below_the_fingertips_there_is_nothing_whatever_the_top_rows_hold():
    """Heights below 0 must not wrap around to the root's top rows, which hold material."""
    for tip, table in FINGERS.reach.items():
        rows = np.flatnonzero(np.isfinite(table).any(axis=1))
        top = int(rows.max())
        column = int(np.flatnonzero(np.isfinite(table[top]))[0])
        lateral = FINGERS.lateral_min_m + (column + 0.5) * FINGERS.cell_m
        assert np.isfinite(lookup(table, [(top + 0.5) * FINGERS.cell_m], [lateral])[0])
        wrapped = (top - table.shape[0] + 0.5) * FINGERS.cell_m  # the negative height whose row wraps onto `top`
        assert wrapped < 0 and lookup(table, [wrapped], [lateral])[0] == -np.inf
