"""The linear_4310 USD the Isaac scene uses, authored on an in-memory stage (pxr only, no Isaac)."""
import numpy as np
import pytest

pxr = pytest.importorskip('pxr')
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from core.robots.kinematics import T  # noqa: E402
from core.robots.linear_4310 import load_linear_4310  # noqa: E402
from tasks.dual_arm_crag_skull.s0.kinematics import yam_flange_T_mount  # noqa: E402
from tasks.dual_arm_crag_skull.sim.yam_usd import FingerDrive, author_linear_4310  # noqa: E402

GRIPPER = load_linear_4310()
FLANGE = '/World/Robots/Left/flange'
ROOT = '/World/Grippers/Left/YAM_Linear_4310'
WORLD_T_FLANGE = T([0.4, -0.3, 1.1], [0.9238795, 0.0, 0.3826834, 0.0])  # 45 deg about y


def matrix(prim):
    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float).T


def local_frame(position, rotation):
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(rotation))), dtype=float).T
    pose[:3, 3] = np.asarray(position, dtype=float)
    return pose


def joint_frames(stage, joint):
    """World poses of a joint's frame as seen from body0 and from body1."""
    bodies = [stage.GetPrimAtPath(rel.GetTargets()[0]) for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())]
    first = matrix(bodies[0]) @ local_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
    second = matrix(bodies[1]) @ local_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
    return first, second


@pytest.fixture
def scene():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    flange = UsdGeom.Xform.Define(stage, FLANGE)
    flange.AddTransformOp().Set(Gf.Matrix4d(WORLD_T_FLANGE.T.tolist()))
    UsdPhysics.RigidBodyAPI.Apply(flange.GetPrim())
    prims = author_linear_4310(stage, ROOT, flange_path=FLANGE, flange_T_mount=yam_flange_T_mount(), gripper=GRIPPER)
    return stage, prims


def test_links_are_rigid_bodies_at_the_flange_mounted_open_pose(scene):
    stage, prims = scene
    world_T_mount = WORLD_T_FLANGE @ yam_flange_T_mount()
    for link, path in prims.links.items():
        prim = stage.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        expected = world_T_mount @ GRIPPER.link_frames[link]
        if link != 'gripper':
            expected[:3, 3] += world_T_mount[:3, :3] @ GRIPPER.slide_axes[link] * GRIPPER.travel_m
        np.testing.assert_allclose(matrix(prim), expected, atol=1e-6, err_msg=link)


def test_meshes_sit_where_the_description_puts_them(scene):
    stage, prims = scene
    world_T_mount = WORLD_T_FLANGE @ yam_flange_T_mount()
    expected = {'gripper': GRIPPER.body_points, **GRIPPER.tips_at(GRIPPER.travel_m)}
    for link, path in prims.links.items():
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f'{path}/Mesh'))
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        world = points @ matrix(mesh.GetPrim())[:3, :3].T + matrix(mesh.GetPrim())[:3, 3]
        reference = expected[link] @ world_T_mount[:3, :3].T + world_T_mount[:3, 3]
        np.testing.assert_allclose(world.min(axis=0), reference.min(axis=0), atol=1e-6, err_msg=link)
        np.testing.assert_allclose(world.max(axis=0), reference.max(axis=0), atol=1e-6, err_msg=link)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        assert (counts == 3).all() and len(indices) == 3 * len(counts) and indices.max() < len(points)



def test_finger_joints_slide_open_from_the_pad_contact_and_are_driven(scene):
    stage, prims = scene
    world_T_mount = WORLD_T_FLANGE @ yam_flange_T_mount()
    drive = FingerDrive()
    for tip, path in prims.joints.items():
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(path))
        assert joint.GetAxisAttr().Get() == 'Z'
        assert joint.GetLowerLimitAttr().Get() == pytest.approx(GRIPPER.contact_joint_m)
        assert joint.GetUpperLimitAttr().Get() == pytest.approx(GRIPPER.travel_m)
        first, second = joint_frames(stage, joint)
        # Authored open: the body1 frame sits travel_m further along the joint axis than the body0 frame.
        np.testing.assert_allclose(first[:3, :3], second[:3, :3], atol=1e-6)
        np.testing.assert_allclose(second[:3, 3] - first[:3, 3], first[:3, 2] * GRIPPER.travel_m, atol=1e-6)
        np.testing.assert_allclose(first[:3, 2], world_T_mount[:3, :3] @ GRIPPER.slide_axes[tip], atol=1e-5)
        api = UsdPhysics.DriveAPI(joint.GetPrim(), 'linear')
        assert (api.GetTypeAttr().Get(), api.GetStiffnessAttr().Get(), api.GetDampingAttr().Get(),
                api.GetMaxForceAttr().Get()) == ('force', drive.stiffness_n_m, drive.damping_n_s_m, drive.max_force_n)
        assert api.GetTargetPositionAttr().Get() == pytest.approx(GRIPPER.travel_m)
        # the drive alone closes at (stiffness / damping) x error, 1.9 m/s from fully open: the joint caps its speed
        assert 'PhysxJointAPI' in joint.GetPrim().GetMetadata('apiSchemas').GetAddedOrExplicitItems()
        assert joint.GetPrim().GetAttribute('physxJoint:maxJointVelocity').Get() == pytest.approx(drive.max_speed_m_s)
    assert drive.max_speed_m_s < drive.stiffness_n_m / drive.damping_n_s_m * GRIPPER.travel_m / 10
    assert prims.dof_names == ('tip_left_joint', 'tip_right_joint')
    assert tuple(path.rsplit('/', 1)[1] for path in prims.joints.values()) == prims.dof_names


def test_the_mount_joint_holds_the_gripper_at_the_adapter_pose(scene):
    stage, prims = scene
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(prims.mount_joint))
    assert joint.GetBody0Rel().GetTargets()[0] == FLANGE and joint.GetBody1Rel().GetTargets()[0] == prims.links['gripper']
    first, second = joint_frames(stage, joint)
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_mass_properties_follow_the_mjcf(scene):
    stage, prims = scene
    for link, path in prims.links.items():
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        inertial = GRIPPER.inertials[link]
        assert mass.GetMassAttr().Get() == pytest.approx(inertial.mass_kg)
        np.testing.assert_allclose(mass.GetCenterOfMassAttr().Get(), inertial.center_of_mass_m, atol=1e-7)
        np.testing.assert_allclose(mass.GetDiagonalInertiaAttr().Get(), inertial.diagonal_inertia_kg_m2, rtol=1e-6)
        axes = mass.GetPrincipalAxesAttr().Get()
        np.testing.assert_allclose([axes.GetReal(), *axes.GetImaginary()], inertial.principal_axes_wxyz, atol=1e-6)


def test_drive_values_are_checked():
    for changes in ({'stiffness_n_m': 0.0}, {'damping_n_s_m': -1.0}, {'max_force_n': float('nan')},
                    {'max_speed_m_s': 0.0}, {'max_speed_m_s': float('inf')}):
        with pytest.raises(ValueError, match='drive'):
            FingerDrive(**changes)


def test_the_two_tips_never_collide_with_each_other(scene):
    """The MJCF excludes tip-tip contacts; the pads overlap 1 mm at q = 0."""
    stage, prims = scene
    left = stage.GetPrimAtPath(prims.links['tip_left'])
    assert left.HasAPI(UsdPhysics.FilteredPairsAPI)
    assert UsdPhysics.FilteredPairsAPI(left).GetFilteredPairsRel().GetTargets() == [prims.links['tip_right']]


def test_links_land_at_their_world_poses_under_a_moved_parent():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    flange = UsdGeom.Xform.Define(stage, FLANGE)
    flange.AddTransformOp().Set(Gf.Matrix4d(WORLD_T_FLANGE.T.tolist()))
    UsdPhysics.RigidBodyAPI.Apply(flange.GetPrim())
    parent = UsdGeom.Xform.Define(stage, '/World/Grippers/Left')
    parent.AddTranslateOp().Set(Gf.Vec3d(1.0, -2.0, 0.5))
    prims = author_linear_4310(stage, ROOT, flange_path=FLANGE, flange_T_mount=yam_flange_T_mount(), gripper=GRIPPER)
    expected = WORLD_T_FLANGE @ yam_flange_T_mount() @ GRIPPER.link_frames['gripper']
    np.testing.assert_allclose(matrix(stage.GetPrimAtPath(prims.links['gripper'])), expected, atol=1e-6)


def test_mesh_vertices_match_the_description_not_only_their_boxes(scene):
    """Centroid and covariance of the distinct vertices: a mesh mirrored inside its box would keep the box."""
    stage, prims = scene
    world_T_mount = WORLD_T_FLANGE @ yam_flange_T_mount()
    expected = {'gripper': GRIPPER.body_points, **GRIPPER.tips_at(GRIPPER.travel_m)}
    for link, path in prims.links.items():
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f'{path}/Mesh'))
        local = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        world = local @ matrix(mesh.GetPrim())[:3, :3].T + matrix(mesh.GetPrim())[:3, 3]
        reference = np.unique(expected[link], axis=0) @ world_T_mount[:3, :3].T + world_T_mount[:3, 3]
        assert len(world) == len(reference), link
        np.testing.assert_allclose(world.mean(axis=0), reference.mean(axis=0), atol=1e-6, err_msg=link)
        np.testing.assert_allclose(np.cov(world.T), np.cov(reference.T), atol=1e-8, err_msg=link)


def inside_some_hull(hulls, points, along, approach, tolerance):
    """Whether every point lies inside a hull whose slab holds it (each point is tested only against those)."""
    from scipy.spatial import ConvexHull
    inside = np.zeros(len(points), dtype=bool)
    for vertices, _ in hulls:
        span = vertices @ approach
        near = (along >= span.min() - 1e-9) & (along <= span.max() + 1e-9)
        equations = ConvexHull(vertices).equations
        inside[near] |= (points[near] @ equations[:, :3].T + equations[:, 3] <= tolerance).all(axis=1)
    return inside


def inner_extreme(points_mount, opening, depth_m, end_z, half_m=0.006):
    """How far towards the other finger a tip reaches within a 12 mm box footprint `depth_m` up from its end."""
    near = (np.abs(points_mount[:, 2] - (end_z - depth_m)) <= half_m) & (np.abs(points_mount[:, 1]) <= half_m)
    return float((points_mount[near] @ opening).max())


def material_along_closing(tip, heights, laterals):
    """Oracle independent of the hulls: whether a line along the closing axis at (height above the fingertip,
    lateral offset from the TCP) meets the tip's mesh, i.e. lies inside some triangle projected onto that plane."""
    from tasks.dual_arm_crag_skull.sim.yam_usd import tip_triangles_in_link
    frame = GRIPPER.link_frames[tip]
    approach = GRIPPER.mount_T_tcp[:3, 2]
    lateral = np.cross(approach, GRIPPER.slide_axes[tip] / np.linalg.norm(GRIPPER.slide_axes[tip]))
    triangles = tip_triangles_in_link(GRIPPER, tip) @ frame[:3, :3].T + frame[:3, 3] - GRIPPER.mount_T_tcp[:3, 3]
    h = (triangles @ approach).max() - triangles @ approach
    u = triangles @ lateral
    v0 = np.stack([h[:, 0], u[:, 0]], -1)
    e1, e2 = np.stack([h[:, 1], u[:, 1]], -1) - v0, np.stack([h[:, 2], u[:, 2]], -1) - v0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    keep = np.abs(det) > 1e-14
    v0, e1, e2, det = v0[keep], e1[keep], e2[keep], det[keep]
    grid = np.stack(np.meshgrid(heights, laterals, indexing='ij'), -1).reshape(-1, 2)
    hit = np.zeros(len(grid), dtype=bool)
    for first in range(0, len(grid), 256):
        p = grid[first:first + 256, None, :] - v0[None]
        s = (p[..., 0] * e2[:, 1] - p[..., 1] * e2[:, 0]) / det
        t = (e1[:, 0] * p[..., 1] - e1[:, 1] * p[..., 0]) / det
        hit[first:first + 256] = ((s >= -1e-9) & (t >= -1e-9) & (s + t <= 1 + 1e-9)).any(axis=1)
    return hit.reshape(len(heights), len(laterals)), approach, lateral


def test_the_tips_have_a_through_window_above_the_pad():
    """The finger is solid only near its end: from about 15 mm up, two rails with nothing between them along the
    closing axis (measured on the mesh, not on any collider)."""
    for tip in ('tip_left', 'tip_right'):
        hit, _, _ = material_along_closing(tip, np.array([0.005, 0.010, 0.020, 0.035, 0.050]), np.array([0.0]))
        assert hit[:2, 0].all() and not hit[2:, 0].any(), tip


def test_tip_colliders_cover_the_mesh_never_bridge_the_window_and_fit_physx_hulls():
    """Each 2 mm slab of a tip is split where the slab's own surface is disconnected (the window between the rails)
    and cut exactly into convex pieces of at most HULL_VERTEX_LIMIT vertices: together they cover the mesh, and no
    piece holds a point where the mesh has no material along the closing axis."""
    from scipy.spatial import ConvexHull
    from tasks.dual_arm_crag_skull.sim.yam_usd import (HULL_TOLERANCE_M, HULL_VERTEX_LIMIT, tip_points_in_link,
                                                       tip_slab_hulls)
    heights, laterals = np.arange(0.0, 0.094, 0.001), np.arange(-0.03, 0.0301, 0.001)
    for tip in ('tip_left', 'tip_right'):
        frame = GRIPPER.link_frames[tip]
        approach_link = frame[:3, :3].T @ GRIPPER.mount_T_tcp[:3, 2]
        hulls = tip_slab_hulls(GRIPPER, tip)
        assert max(len(vertices) for vertices, _ in hulls) <= HULL_VERTEX_LIMIT == 64
        # PhysX's GPU hulls also have at most 64 faces: a 60-point sphere (116 faces) was cooked down to 34 points
        assert max(len(np.unique(np.round(ConvexHull(vertices).equations, 7), axis=0)) for vertices, _ in hulls) <= 64
        for vertices, faces in hulls:  # faces index the hull's own vertices and bound it
            assert len(ConvexHull(vertices).vertices) == len(vertices)
            assert faces.min() == 0 and faces.max() == len(vertices) - 1
        mesh = tip_points_in_link(GRIPPER, tip, spacing_m=0.0015)
        assert inside_some_hull(hulls, mesh, mesh @ approach_link, approach_link, HULL_TOLERANCE_M + 1e-7).all()
        hit, approach, lateral = material_along_closing(tip, heights, laterals)
        # Points across the finger's thickness where the mesh has no material along the closing axis within 2 mm up
        # or down and 1 mm across (the slabs are 2 mm thick, the window's edges slope): a collider holding any of
        # them is solid where the real finger is open.
        closing = GRIPPER.slide_axes[tip] / np.linalg.norm(GRIPPER.slide_axes[tip])
        tip_points = GRIPPER.tip_points[tip] - GRIPPER.mount_T_tcp[:3, 3]
        depth = tip_points @ closing
        end = (tip_points @ approach).max()
        open_ = ~hit
        clear = open_.copy()
        for di in range(-2, 3):
            for dj in range(-1, 2):
                clear &= np.roll(np.roll(open_, di, axis=0), dj, axis=1)
        empty = [(h, u) for i, h in enumerate(heights) for j, u in enumerate(laterals)
                 if clear[i, j] and 2 <= i < len(heights) - 2 and hit[i].any()
                 and laterals[hit[i]].min() < u < laterals[hit[i]].max()]
        assert len(empty) > 300, len(empty)  # the window is large: the oracle does see it
        probes = np.array([[*(np.mean([depth.min(), depth.max()]) * closing)] for _ in empty])
        probes = probes + np.array([(end - h) * approach + u * lateral for h, u in empty])
        to_link = (probes + GRIPPER.mount_T_tcp[:3, 3] - frame[:3, 3]) @ frame[:3, :3]
        assert not inside_some_hull(hulls, to_link, to_link @ approach_link, approach_link, -1e-4).any(), tip


def test_tip_colliders_reach_inward_exactly_as_far_as_the_mesh():
    """Towards the other finger, where objects are held, the pieces reach as far as the mesh in every 2 mm footprint
    up to ROOT_FROM_TIP_M from the fingertip, and nowhere where it has no material (the window). Above that the
    finger's root is L-shaped across a slab and its hull bridges the corner (up to 48 mm further inward 74-80 mm up):
    documented in sim/yam_usd.py, nothing held here is that tall (the moving fragments stand at most 48 mm)."""
    from tasks.dual_arm_crag_skull.sim.yam_usd import ROOT_FROM_TIP_M
    from tasks.dual_arm_crag_skull.sim.pick import surface_points
    from tasks.dual_arm_crag_skull.sim.yam_usd import tip_points_in_link, tip_slab_hulls
    end_z = GRIPPER.mount_T_tcp[2, 3]
    for tip, opening in (('tip_left', np.array([1.0, 0, 0])), ('tip_right', np.array([-1.0, 0, 0]))):
        frame = GRIPPER.link_frames[tip]
        hulls = tip_slab_hulls(GRIPPER, tip)
        to_mount = lambda points: points @ frame[:3, :3].T + frame[:3, 3]  # noqa: E731
        hull_surface = to_mount(np.vstack([surface_points(vertices, faces, 0.0005) for vertices, faces in hulls]))
        dense = to_mount(tip_points_in_link(GRIPPER, tip, spacing_m=0.0005))
        heights, laterals = np.arange(0.0, 0.0941, 0.0005), np.arange(-0.03, 0.03001, 0.0005)
        hit, _, _ = material_along_closing(tip, heights, laterals)
        compared = []
        for depth in np.arange(0.002, ROOT_FROM_TIP_M, 0.004):
            for lateral in np.arange(-0.024, 0.0241, 0.004):
                # footprints wholly inside the material or wholly outside it, 0.5 mm around: their edges are where
                # the two samplings differ by chance
                rows = np.abs(heights - depth) <= 0.0015 + 1e-9
                cols = np.abs(laterals - lateral) <= 0.0015 + 1e-9
                cells = hit[np.ix_(rows, cols)]
                found = [inner_extreme_cell(points, opening, depth, lateral, end_z) for points in (hull_surface, dense)]
                if not cells.any():
                    assert found == [None, None], (tip, depth, lateral, found)
                elif cells.all():
                    compared.append(depth)
                    assert found[0] == pytest.approx(found[1], abs=2.5e-4), (tip, depth, lateral)
        compared = np.array(compared)  # both the solid pad and the rails beside the window were compared
        assert (compared <= 0.014).sum() >= 10 and (compared >= 0.02).sum() >= 20, compared


def inner_extreme_cell(points_mount, opening, depth_m, lateral_m, end_z, half_m=0.001):
    """How far towards the other finger a tip reaches within a 2 mm footprint centred depth_m up from its end and
    lateral_m across, or None where it has no surface there."""
    near = ((np.abs(points_mount[:, 2] - (end_z - depth_m)) <= half_m)
            & (np.abs(points_mount[:, 1] - lateral_m) <= half_m))
    return float((points_mount[near] @ opening).max()) if near.any() else None


def test_meshes_are_visual_only_and_hull_pieces_collide_as_physx_cooks_them(scene):
    """Every link collides only through invisible convex pieces of at most HULL_VERTEX_LIMIT points, with that
    limit and HULL_MIN_THICKNESS_M authored, so PhysX neither re-cooks nor thickens them."""
    from tasks.dual_arm_crag_skull.sim.yam_usd import HULL_MIN_THICKNESS_M, HULL_VERTEX_LIMIT
    stage, prims = scene
    for link, path in prims.links.items():
        assert not stage.GetPrimAtPath(f'{path}/Mesh').HasAPI(UsdPhysics.CollisionAPI), link
        pieces = list(stage.GetPrimAtPath(f'{path}/Colliders').GetChildren())
        assert pieces and all(piece.HasAPI(UsdPhysics.CollisionAPI) for piece in pieces), link
        assert {UsdPhysics.MeshCollisionAPI(piece).GetApproximationAttr().Get() for piece in pieces} == {'convexHull'}
        assert {UsdGeom.Imageable(piece).ComputeVisibility() for piece in pieces} == {UsdGeom.Tokens.invisible}
        for piece in pieces:
            assert 'PhysxConvexHullCollisionAPI' in piece.GetMetadata('apiSchemas').GetAddedOrExplicitItems()
            assert piece.GetAttribute('physxConvexHullCollision:hullVertexLimit').Get() == HULL_VERTEX_LIMIT
            assert piece.GetAttribute('physxConvexHullCollision:minThickness').Get() == pytest.approx(
                HULL_MIN_THICKNESS_M)
            assert len(UsdGeom.Mesh(piece).GetPointsAttr().Get()) <= HULL_VERTEX_LIMIT


def test_hull_pieces_are_authored_at_cooking_scale_with_explicit_contact_offsets(scene):
    """PhysX drops hull points within planeTolerance (0.7 mm) of a plane while cooking, in the mesh's own units: the
    pieces are authored COOKING_SCALE times larger under a 1 / COOKING_SCALE scale, so that is 7 um here, and land
    exactly where the hulls are. Small pieces get a small automatic contact offset; it is authored instead."""
    from tasks.dual_arm_crag_skull.sim.yam_usd import (COLLIDER_CONTACT_OFFSET_M, COOKING_SCALE, body_hulls,
                                                       tip_slab_hulls)
    stage, prims = scene
    expected = {'gripper': body_hulls(GRIPPER), 'tip_left': tip_slab_hulls(GRIPPER, 'tip_left'),
                'tip_right': tip_slab_hulls(GRIPPER, 'tip_right')}
    for link, path in prims.links.items():
        body = matrix(stage.GetPrimAtPath(path))
        pieces = list(stage.GetPrimAtPath(f'{path}/Colliders').GetChildren())
        assert len(pieces) == len(expected[link])
        for piece, (vertices, _) in zip(pieces, expected[link]):
            local = np.asarray(UsdGeom.Mesh(piece).GetPointsAttr().Get(), dtype=float)
            np.testing.assert_allclose(local, vertices * COOKING_SCALE, rtol=0, atol=1e-5)
            world = local @ matrix(piece)[:3, :3].T + matrix(piece)[:3, 3]
            np.testing.assert_allclose(world, vertices @ body[:3, :3].T + body[:3, 3], rtol=0, atol=1e-6)
            assert piece.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(COLLIDER_CONTACT_OFFSET_M)
            assert piece.GetAttribute('physxCollision:restOffset').Get() == 0.0
            assert 'PhysxCollisionAPI' in piece.GetMetadata('apiSchemas').GetAddedOrExplicitItems()


def test_hull_pieces_are_exact_halves_or_thinned_inside_the_hull_within_tolerance():
    """On a dense sphere (no piece is thin enough without cutting): every piece lies inside the exact hull, has at
    most the vertex limit, falls short of it by at most the tolerance, and the pieces together cover it."""
    from scipy.spatial import ConvexHull
    from tasks.dual_arm_crag_skull.sim.yam_usd import hull_pieces
    rng = np.random.default_rng(0)
    sphere = rng.normal(size=(3000, 3))
    sphere = 0.01 * sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    exact = ConvexHull(sphere)
    pieces = hull_pieces(sphere, limit=64, tolerance=2e-5)
    assert len(pieces) > 1
    for vertices, error in pieces:
        assert len(vertices) <= 64 and error <= 2e-5
        assert (vertices @ exact.equations[:, :3].T + exact.equations[:, 3] <= 1e-12).all()
    covered = np.zeros(len(sphere), dtype=bool)
    for vertices, error in pieces:
        planes = ConvexHull(vertices).equations
        covered |= (sphere @ planes[:, :3].T + planes[:, 3] <= error + 1e-12).all(axis=1)
    assert covered.all()
    total = sum(ConvexHull(vertices).volume for vertices, _ in pieces)
    assert exact.volume * (1 - 0.01) <= total <= exact.volume * (1 + 1e-9)


def test_the_body_collider_is_its_hull_within_tolerance():
    from scipy.spatial import ConvexHull
    from core.robots.linear_4310 import read_binary_stl
    from tasks.dual_arm_crag_skull.sim.yam_usd import HULL_TOLERANCE_M, HULL_VERTEX_LIMIT, body_hulls
    pose = GRIPPER.mesh_poses['gripper']
    points = read_binary_stl(GRIPPER.mesh_files['gripper']) @ pose[:3, :3].T + pose[:3, 3]
    exact = ConvexHull(points)
    hulls = body_hulls(GRIPPER)
    assert all(len(vertices) <= HULL_VERTEX_LIMIT for vertices, _ in hulls)
    for vertices, _ in hulls:
        assert (vertices @ exact.equations[:, :3].T + exact.equations[:, 3] <= 1e-9).all()
    inside = np.zeros(len(exact.vertices), dtype=bool)
    for vertices, _ in hulls:
        planes = ConvexHull(vertices).equations
        inside |= (points[exact.vertices] @ planes[:, :3].T + planes[:, 3] <= HULL_TOLERANCE_M + 1e-9).all(axis=1)
    assert inside.all()
