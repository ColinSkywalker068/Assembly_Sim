"""DexAgent's CRX + YAM bench: the YAM fingers' convex collision pieces (baselines/dexagent/bench/yam_pieces.py),
against the official tip meshes."""
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from core.robots.linear_4310 import TIPS, load_linear_4310
from tasks.dual_arm_crag_skull.baselines.dexagent.bench import yam_pieces
from tasks.dual_arm_crag_skull.sim.yam_usd import ROOT_FROM_TIP_M, tip_points_in_link
from tasks.dual_arm_crag_skull.tests.test_sim_yam_usd import inner_extreme_cell, material_along_closing

GRIPPER = load_linear_4310()


def test_convex_silhouettes_are_exact_for_a_box():
    axes = np.eye(3)
    silhouettes = yam_pieces.Silhouettes(axes, [-0.01, -0.01, -0.01], [0.01, 0.01, 0.01], cell_m=0.001)
    box = np.array([[x, y, z] for x in (-0.004, 0.006) for y in (-0.002, 0.003) for z in (-0.005, 0.001)])
    blocks = silhouettes.convex_blocks(box)
    rows, columns, block = blocks[0, 1.0]  # +x over (y, z)
    centres_y = -0.01 + (np.arange(silhouettes.shape[1])[rows] + 0.5) * 0.001
    centres_z = -0.01 + (np.arange(silhouettes.shape[2])[columns] + 0.5) * 0.001
    inside = (np.abs(centres_y - 0.0005) < 0.0025)[:, None] & (np.abs(centres_z + 0.002) < 0.003)[None, :]
    np.testing.assert_allclose(block[inside], 0.006, atol=1e-12)
    assert np.isneginf(block[~inside]).all()
    np.testing.assert_allclose(blocks[0, -1.0][2][inside], 0.004, atol=1e-12)


def test_convex_silhouettes_are_where_each_cell_line_enters_and_leaves_a_random_polytope():
    """Against brute force: points every micrometre along each cell's centre line, inside the hull or not."""
    rng = np.random.default_rng(3)
    points = rng.normal(size=(30, 3)) * [0.004, 0.002, 0.006]
    axes = Rotation.from_euler('xyz', [0.3, -0.2, 0.7]).as_matrix()
    silhouettes = yam_pieces.Silhouettes(axes, [-0.02] * 3, [0.02] * 3, cell_m=0.0005)
    equations = ConvexHull(points).equations
    t = np.arange(-0.03, 0.03, 1e-6)
    for (axis, sign), (rows, columns, block) in silhouettes.convex_blocks(points).items():
        if sign < 0:
            continue
        i, j = (k for k in range(3) if k != axis)
        lower = silhouettes.convex_blocks(points)[axis, -1.0][2]
        met = 0
        for r, c in zip(*(index.ravel()[::23] for index in np.indices(block.shape))):
            centre = np.zeros(3)
            centre[i] = -0.02 + (np.arange(silhouettes.shape[i])[rows][r] + 0.5) * 0.0005
            centre[j] = -0.02 + (np.arange(silhouettes.shape[j])[columns][c] + 0.5) * 0.0005
            line = centre[None, :] + t[:, None] * np.eye(3)[axis]
            world = line @ axes  # rows of `axes` are the frame's unit vectors
            inside = ((world @ equations[:, :3].T + equations[:, 3]) <= 0).all(axis=1)
            if not inside.any():
                assert np.isneginf(block[r, c]) and np.isneginf(lower[r, c])
                continue
            met += 1
            assert block[r, c] == pytest.approx(t[inside].max(), abs=2e-6)
            assert -lower[r, c] == pytest.approx(t[inside].min(), abs=2e-6)
        assert met > 20


def test_each_tip_is_a_few_pieces_and_below_the_root_none_stands_out():
    for tip in TIPS:
        pieces = yam_pieces.tip_pieces(GRIPPER, tip)
        assert len(pieces) <= 14, len(pieces)
        finger = [p for p in pieces if p.heights_m[1] <= ROOT_FROM_TIP_M + 0.001]
        assert finger[0].heights_m[0] == pytest.approx(0.0, abs=1e-9)  # the pad, from the fingertip end
        # every finger piece within the tolerance, but for the 2 mm slab where the window begins (bridged by at most
        # a slab, as in Isaac)
        bridged = [p for p in finger if not p.bulge_m <= yam_pieces.MERGE_TOLERANCE_M]
        assert len(bridged) == 1 and bridged[0].heights_m[1] - bridged[0].heights_m[0] <= 0.0021, bridged
        assert 0.012 < bridged[0].heights_m[0] < 0.016


def test_tip_pieces_cover_the_mesh():
    """Every point of the tip's surface is inside some piece (to the slabs' 1e-6 m)."""
    for tip in TIPS:
        mesh = tip_points_in_link(GRIPPER, tip, spacing_m=0.001)
        outside = np.full(len(mesh), np.inf)
        for piece in yam_pieces.tip_pieces(GRIPPER, tip):
            equations = ConvexHull(piece.vertices).equations
            outside = np.minimum(outside, (mesh @ equations[:, :3].T + equations[:, 3]).max(axis=1))
        assert outside.max() <= 1e-6, outside.max()


def test_tip_pieces_reach_inward_as_far_as_the_mesh_and_leave_the_window_open():
    """Towards the other finger the pieces reach where the mesh does (MERGE_TOLERANCE_M) in every footprint up to
    ROOT_FROM_TIP_M, and nowhere the mesh has no material along the closing axis (the window)."""
    from tasks.dual_arm_crag_skull.sim.pick import surface_points
    end_z = GRIPPER.mount_T_tcp[2, 3]
    for tip, opening in (('tip_left', np.array([1.0, 0, 0])), ('tip_right', np.array([-1.0, 0, 0]))):
        frame = GRIPPER.link_frames[tip]
        to_mount = lambda points: points @ frame[:3, :3].T + frame[:3, 3]  # noqa: E731
        surfaces = []
        for piece in yam_pieces.tip_pieces(GRIPPER, tip):
            hull = ConvexHull(piece.vertices)
            surfaces.append(surface_points(piece.vertices, hull.simplices, 0.0005))
        hull_surface = to_mount(np.vstack(surfaces))
        dense = to_mount(tip_points_in_link(GRIPPER, tip, spacing_m=0.0005))
        heights, laterals = np.arange(0.0, 0.0941, 0.0005), np.arange(-0.03, 0.03001, 0.0005)
        hit, _, _ = material_along_closing(tip, heights, laterals)
        compared, empty = [], 0
        for depth in np.arange(0.002, ROOT_FROM_TIP_M, 0.004):
            for lateral in np.arange(-0.024, 0.0241, 0.004):
                rows = np.abs(heights - depth) <= 0.0015 + 1e-9
                cols = np.abs(laterals - lateral) <= 0.0015 + 1e-9
                cells = hit[np.ix_(rows, cols)]
                found = [inner_extreme_cell(points, opening, depth, lateral, end_z) for points in (hull_surface, dense)]
                if not cells.any() and abs(depth - 0.015) > 0.0025:  # off the slab where the window begins
                    empty += 1
                    assert found == [None, None], (tip, depth, lateral, found)
                elif cells.all():
                    compared.append(depth)
                    assert found[0] == pytest.approx(found[1], abs=yam_pieces.MERGE_TOLERANCE_M + 2.5e-4), (tip, depth,
                                                                                                       lateral)
        compared = np.array(compared)
        assert (compared <= 0.014).sum() >= 10 and (compared >= 0.02).sum() >= 20 and empty >= 20, (compared, empty)


def test_the_body_collides_through_its_meshs_convex_hull():
    from core.robots.linear_4310 import read_binary_stl
    pose = GRIPPER.mesh_poses['gripper']
    points = read_binary_stl(GRIPPER.mesh_files['gripper']) @ pose[:3, :3].T + pose[:3, 3]
    vertices = yam_pieces.body_vertices(GRIPPER)
    assert ConvexHull(vertices).volume == pytest.approx(ConvexHull(points).volume, rel=1e-9)
