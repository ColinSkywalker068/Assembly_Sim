"""USD for the I2RT linear_4310 gripper on a CRX flange, from core.robots.linear_4310. pxr (UsdPhysics) only.

Three rigid bodies (the body and the two tips) at their world poses for the flange's current pose, each with one visual
mesh and invisible convex colliders. Each finger is solid only over its last PAD_M (15 mm), which collides as one convex
hull (tip_slab_hulls): its flat face toward the other finger exact, at most PAD_TOLERANCE_M short of the mesh at the
rounded edges. Above it, two rails run either side of a window with no material at all along the closing axis (+-2 mm
wide 16 mm up, +-9.5 mm at 35 mm, wider than +-15 mm at 50 mm), so the tip there is cut into SLAB_M thick slabs across
the approach axis, each split where its surface is (the rails), never bridged: a single hull would fill the window, and
PhysX's own decomposition stood 1.8 mm proud of the pads by default and, shrink-wrapped, let one finger sink 2.3 mm into
a box of known width on one arm only. The pad was slabs too until 2026-10-22: a slab's top 0.2 mm under a seam between
two stacked boxes of a held fragment let PhysX report a vertical penetration of slab thickness plus gap (2.2 mm) and
push 390 N through that internal face in one step, turning the fragment 35 deg in the grasp; slabs turned a fragment in
10 of 12 runs below 0.25 mm vertical seams, the one pad hull in none of 7 (contact reports, telemetry). Every collider
has at most HULL_VERTEX_LIMIT vertices and HULL_FACE_LIMIT faces, PhysX's GPU limits (the vertex limit authored too):
above them PhysX cooks a smaller hull of its own, and while cooking it drops points within 0.7 mm of a plane, so
colliders are authored at COOKING_SCALE (check_yam reads the cooked shapes back). A slab piece with more is thinned or
cut exactly (hull_pieces), inside its hull by at most HULL_TOLERANCE_M; the body's hull the same (body_hulls). Up to
ROOT_FROM_TIP_M from the fingertip the colliders reach inward exactly as far as the mesh; above, the finger's root is
L-shaped across a slab and its hull bridges the corner (up to 48 mm further inward 74-80 mm up): nothing that tall is
held here. The tips are authored fully open and never collide with each other (a filtered pair, as the MJCF excludes
them: the pads overlap 1 mm at q = 0). Each tip hangs on a prismatic joint along its slide axis, limited from pad
contact (contact_joint_m) to travel_m, with a linear position drive; the joint prim's name is the DOF name. The other
tip follows DRIVEN_TIP through a PhysX mimic joint (gearing -1: the same joint value), as the real gripper's fingers and
the MJCF's 1:1 equality constraint move together: when one pad touches first both stop, and a held fragment pushed
sideways cannot move one finger out and the other in (uncoupled, insertions in Isaac pushed one finger 2-3.5 mm out at
the same gap, 2026-10-21). The follower keeps its own drive: with the drive on the driven tip alone (twice as strong),
two pads on a kinematic box stood 1.1 mm a side off it (the two contacts and the mimic joint over-constrain the pair),
with both 0.37 mm. A fixed joint holds the body on the flange at flange_T_mount. The articulation the flange belongs to
solves with at least POSITION_ITERATIONS position iterations.

The drive values are PROVISIONAL: the real gripper's force and servo gains are not measured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from core.robots.kinematics import R_to_quat
from core.robots.linear_4310 import LINKS, TIPS

DRIVEN_TIP = 'tip_left'  # the coupling's reference: the other tip follows it, both keep a drive (module docstring)
# PhysxMimicJointAPI's instance name: it allows only rotX, rotY and rotZ (an unknown one, transZ, was ignored without
# a word), and a prismatic joint's single degree of freedom is implicit
MIMIC_AXIS = 'rotX'
# Position iterations of the articulation the gripper hangs on, at least: the mimic joint, the drives and a pad's
# contact are solved together, and at the arm asset's 32 a pad stood 0.13-0.37 mm off a kinematic box (0.03 at 64,
# 0.04 at 255; uncoupled, 0.00 at 32), 2026-10-21
POSITION_ITERATIONS = 64
SLAB_M = 0.002
TIP_SAMPLE_M = 0.0005
HULL_VERTEX_LIMIT = 64  # PhysX's GPU convex limit, physxConvexHullCollision:hullVertexLimit's default (authored too)
# PhysX's GPU hulls also have at most 64 faces: in Isaac a 60-point sphere (116 faces) was cooked down to 34 points
HULL_FACE_LIMIT = 64
HULL_TOLERANCE_M = 5e-5  # how far inside the exact hull a collider with at most HULL_VERTEX_LIMIT vertices may fall
# Each tip's solid end, one convex collider (the pad), and how far that hull, thinned to HULL_VERTEX_LIMIT vertices, may
# fall short of the exact one (0.43 / 0.47 mm on the left / right tip, at the rounded edges; the pad face is exact)
PAD_M = 0.015
PAD_TOLERANCE_M = 5e-4
# PhysX runs a GPU hull's contacts on the CPU when a face has more than this many vertices (PhysX 5 docs, GPU rigid
# bodies); the pad stays within it (11), many rail slabs do not yet
HULL_FACE_VERTEX_LIMIT = 32
HULL_MIN_THICKNESS_M = 1e-4  # authored physxConvexHullCollision:minThickness (default 1 mm; some walls are 0.4 mm)
# PhysX drops hull points within planeTolerance (0.7 mm) of a plane while cooking, in the mesh's own units; colliders
# are authored this many times larger under the inverse scale, which makes that 7 um
COOKING_SCALE = 100.0
# authored physxCollision:contactOffset: left automatic, it scales with a shape's size, and fingers closing at full
# drive force (1.7 mm per 240 Hz step) sank through boxes with the small pieces
COLLIDER_CONTACT_OFFSET_M = 0.002
# Up to here from the fingertip the tip colliders reach inward exactly as far as the mesh; above, the finger's root is
# L-shaped across a slab and its hull bridges the corner (up to 48 mm further inward 74-80 mm up, 3 mm at 90 mm).
ROOT_FROM_TIP_M = 0.057


@dataclass(frozen=True)
class FingerDrive:
    """Linear position drive of each finger joint and the joint's speed limit (PROVISIONAL). On its own the drive
    closes at (stiffness / damping) x the error, 1.9 m/s from fully open (measured 1.5 m/s in Isaac): fingers slamming
    into a box 12 mm per 120 Hz step sank through it. max_speed_m_s (PhysX's maxJointVelocity) makes a full stroke
    take about 0.5 s; the real gripper's speed is not measured."""

    stiffness_n_m: float = 2000.0
    damping_n_s_m: float = 50.0
    max_force_n: float = 20.0
    max_speed_m_s: float = 0.1

    def __post_init__(self):
        values = (self.stiffness_n_m, self.damping_n_s_m, self.max_force_n, self.max_speed_m_s)
        if not all(math.isfinite(value) for value in values) or not (self.stiffness_n_m > 0 and self.damping_n_s_m >= 0
                                                                       and self.max_force_n > 0
                                                                       and self.max_speed_m_s > 0):
            raise ValueError('finger drive needs finite stiffness > 0, damping >= 0, max force > 0 and max speed > 0')


@dataclass(frozen=True)
class YamPrims:
    root: str
    links: Mapping[str, str]
    joints: Mapping[str, str]
    mount_joint: str
    dof_names: tuple[str, ...]


def _world(stage, path):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f'no prim at {path}')
    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float).T


def _set_world(xformable, pose):
    """Give `xformable` the world pose `pose` through a local transform relative to its parent."""
    from pxr import Gf, Usd, UsdGeom
    parent = xformable.GetPrim().GetParent()
    world_T_parent = (np.asarray(UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                                 dtype=float).T if parent.IsA(UsdGeom.Xformable) else np.eye(4))
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(Gf.Matrix4d((np.linalg.inv(world_T_parent) @ np.asarray(pose, dtype=float)).T.tolist()))


def _quatf(rotation):
    from pxr import Gf
    w, x, y, z = (float(value) for value in R_to_quat(rotation))
    return Gf.Quatf(w, x, y, z)


def _set_local_frames(joint, body0_T_joint, body1_T_joint):
    from pxr import Gf
    joint.CreateLocalPos0Attr(Gf.Vec3f(*body0_T_joint[:3, 3].tolist()))
    joint.CreateLocalRot0Attr(_quatf(body0_T_joint[:3, :3]))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*body1_T_joint[:3, 3].tolist()))
    joint.CreateLocalRot1Attr(_quatf(body1_T_joint[:3, :3]))


def _frame_with_z(axis):
    """A rotation whose z column is `axis` (unit)."""
    z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def tip_triangles_in_link(gripper, tip):
    """(T, 3, 3) triangles of a tip's mesh in its link frame."""
    from core.robots.linear_4310 import read_binary_stl
    pose = gripper.mesh_poses[tip]
    return (read_binary_stl(gripper.mesh_files[tip]) @ pose[:3, :3].T + pose[:3, 3]).reshape(-1, 3, 3)


def tip_points_in_link(gripper, tip, spacing_m=TIP_SAMPLE_M):
    """Points covering a tip's mesh surface every `spacing_m`, in its link frame."""
    from .pick import surface_points
    triangles = tip_triangles_in_link(gripper, tip)
    return surface_points(triangles.reshape(-1, 3), np.arange(len(triangles) * 3).reshape(-1, 3), spacing_m)


def _clipped_points(triangles, along, start, stop):
    """The corners of every triangle clipped to start <= along <= stop: its corners inside the slab and where its
    edges cross the slab's two planes."""
    corners = triangles.reshape(-1, 3)
    values = along.reshape(-1)
    points = [corners[(values >= start) & (values <= stop)]]
    first, second = triangles, np.roll(triangles, -1, axis=1)
    a, b = along, np.roll(along, -1, axis=1)
    for plane in (start, stop):
        crossing = (a - plane) * (b - plane) < 0
        fraction = ((plane - a) / np.where(crossing, b - a, 1.0))[crossing]
        points.append(first[crossing] + fraction[:, None] * (second[crossing] - first[crossing]))
    return np.vstack(points)


def _slab_groups(triangles, along, start, stop):
    """Index arrays of the triangles reaching into the slab start <= along <= stop, one per piece of the slab's
    surface: two triangles belong together when they share an edge that reaches into the slab. Where a slab crosses
    the window between a tip's rails its surface is two pieces, one per rail, so no hull bridges the window."""
    _, ids = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    ids = np.asarray(ids).reshape(-1, 3)
    inside = np.flatnonzero((along.min(axis=1) <= stop) & (along.max(axis=1) >= start))
    parent = {int(t): int(t) for t in inside}

    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    owner = {}
    for t in (int(t) for t in inside):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if min(along[t, a], along[t, b]) > stop or max(along[t, a], along[t, b]) < start:
                continue
            edge = (min(ids[t, a], ids[t, b]), max(ids[t, a], ids[t, b]))
            if edge in owner:
                parent[root(t)] = root(owner[edge])
            else:
                owner[edge] = t
    groups = {}
    for t in (int(t) for t in inside):
        groups.setdefault(root(t), []).append(t)
    return [np.array(group) for group in groups.values()]


def _hull_edges(hull):
    """Point-index pairs of a hull's edges between faces that are not coplanar (no triangulation diagonals)."""
    faces = {}
    for index, simplex in enumerate(hull.simplices):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            faces.setdefault((min(simplex[a], simplex[b]), max(simplex[a], simplex[b])), []).append(index)
    normals = hull.equations[:, :3]
    return [edge for edge, (first, second) in faces.items() if normals[first] @ normals[second] < 1 - 1e-9]


def hull_faces(vertices):
    """How many faces the hull of `vertices` has, coplanar triangles counted as one."""
    from scipy.spatial import ConvexHull
    return len(np.unique(np.round(ConvexHull(vertices).equations, 7), axis=0))


def face_vertices(vertices):
    """The most vertices on one face of the hull of `vertices`, coplanar triangles counted as one face."""
    from scipy.spatial import ConvexHull
    hull, faces = ConvexHull(vertices), {}
    for simplex, plane in zip(hull.simplices, np.round(hull.equations, 7)):
        faces.setdefault(tuple(plane), set()).update(int(index) for index in simplex)
    return max(len(face) for face in faces.values())


def _thinned(points, limit, face_limit=HULL_FACE_LIMIT):
    """(vertices, error): at most `limit` of the hull vertices of `points`, with at most `face_limit` faces, dropping
    first those where the hull is flattest (distance from the plane through a vertex's hull neighbours, a tenth of the
    excess per pass, no two neighbours in one), and how far the hull's own vertices lie outside the thinned hull. A
    subset's hull lies inside the full one, so the thinned hull never protrudes: `error` is how far it falls short."""
    from scipy.spatial import ConvexHull
    full = points[ConvexHull(points).vertices]
    vertices = full
    while len(vertices) > limit or hull_faces(vertices) > face_limit:
        neighbours = [set() for _ in vertices]
        for a, b, c in ConvexHull(vertices).simplices:
            neighbours[a] |= {b, c}
            neighbours[b] |= {a, c}
            neighbours[c] |= {a, b}
        flatness = np.empty(len(vertices))
        for index, ring in enumerate(neighbours):
            ring = vertices[sorted(ring)]
            centre = ring.mean(axis=0)
            normal = np.linalg.svd(ring - centre, full_matrices=False)[2][-1]
            flatness[index] = abs((vertices[index] - centre) @ normal)
        budget = max(1, min(max(len(vertices) - limit, 1), len(vertices) // 10))
        dropped, blocked = [], set()
        for index in np.argsort(flatness, kind='stable'):
            if len(dropped) == budget:
                break
            if index not in blocked:
                dropped.append(index)
                blocked |= neighbours[index] | {index}
        vertices = np.delete(vertices, dropped, axis=0)
    planes = ConvexHull(vertices).equations
    return vertices, max(0.0, float((full @ planes[:, :3].T + planes[:, 3]).max()))


def _halves(points):
    """The two convex pieces of the hull of `points` either side of the plane across its longest extent through the
    median of its vertices: each keeps its own vertices and the points where the hull's edges cross the plane, so
    together they are exactly the hull."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    normal = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)[2][0]
    side = points @ normal - np.median(vertices @ normal)
    crossings = [points[i] + (points[j] - points[i]) * (side[i] / (side[i] - side[j]))
                 for i, j in _hull_edges(hull) if side[i] * side[j] < 0]
    own = side[hull.vertices]
    return [np.vstack([vertices[keep], *crossings]) for keep in (own <= 0, own >= 0)]


def hull_pieces(points, *, limit=HULL_VERTEX_LIMIT, tolerance=HULL_TOLERANCE_M):
    """[(vertices, error)] of convex pieces for the hull of `points`, each with at most `limit` vertices (all its
    hull's) and HULL_FACE_LIMIT faces, and inside the exact hull by at most `error` <= `tolerance`: thinned
    (_thinned) when that is close enough, otherwise cut in two exact halves (_halves) and each fitted the same way."""
    vertices, error = _thinned(points, limit)
    if error <= tolerance:
        return [(vertices, error)]
    return [piece for half in _halves(points) for piece in hull_pieces(half, limit=limit, tolerance=tolerance)]


def _hull_mesh(vertices):
    """(vertices, triangles) of the hull of `vertices`, every one of which is a hull vertex."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[hull.vertices] = np.arange(len(hull.vertices), dtype=np.int32)
    return vertices[hull.vertices], remap[hull.simplices]


_HULLS = {}


def tip_slab_hulls(gripper, tip, slab_m=SLAB_M, pad_m=PAD_M):
    """[(vertices, triangles)] of the convex colliders of a tip, in its link frame, the pad first. The tip's solid last
    `pad_m` is one hull, its exact hull thinned to at most HULL_VERTEX_LIMIT vertices (a subset, so inside it, short by
    at most PAD_TOLERANCE_M) with at most HULL_FACE_VERTEX_LIMIT on a face. Above it the tip is cut into `slab_m` thick
    slabs across the approach axis, counted from the pad (the slab at the root takes what is left). A slab's surface is
    split into its connected pieces (_slab_groups: the rails either side of the window), and each piece's hull is that
    of its triangles clipped to the slab (a slab's solid has its extreme points on its clipped surface), fitted by
    hull_pieces of at most HULL_VERTEX_LIMIT vertices so that PhysX cooks them as they are, each inside it by at most
    HULL_TOLERANCE_M. Neighbouring colliders share their cut plane, so together they cover the tip to within those
    tolerances."""
    from scipy.spatial import QhullError
    key = (str(gripper.mesh_files[tip]), np.asarray(gripper.mesh_poses[tip]).tobytes(),
           np.asarray(gripper.link_frames[tip]).tobytes(), np.asarray(gripper.mount_T_tcp).tobytes(), slab_m, pad_m)
    if key in _HULLS:
        return list(_HULLS[key])
    triangles = tip_triangles_in_link(gripper, tip)
    approach = gripper.link_frames[tip][:3, :3].T @ gripper.mount_T_tcp[:3, 2]
    along = triangles @ approach
    low, high = float(along.min()), float(along.max())
    base = high - pad_m
    pad = _slab_groups(triangles, along, base, high)
    if len(pad) != 1:
        raise ValueError(f'{tip}: its last {pad_m * 1000:g} mm is {len(pad)} pieces, not one solid pad')
    vertices, short = _thinned(_clipped_points(triangles[pad[0]], along[pad[0]], base, high), HULL_VERTEX_LIMIT)
    if short > PAD_TOLERANCE_M or face_vertices(vertices) > HULL_FACE_VERTEX_LIMIT:
        raise ValueError(f'{tip}: the pad hull falls {short * 1000:.3f} mm short of the mesh and has '
                         f'{face_vertices(vertices)} vertices on a face')
    hulls = [_hull_mesh(vertices)]
    for k in range(max(1, math.ceil((base - low) / slab_m - 1e-9))):
        start, stop = max(low, base - (k + 1) * slab_m), base - k * slab_m
        for group in _slab_groups(triangles, along, start, stop):
            try:
                pieces = hull_pieces(_clipped_points(triangles[group], along[group], start, stop))
            except (QhullError, ValueError):
                continue
            hulls.extend(_hull_mesh(vertices) for vertices, _ in pieces)
    _HULLS[key] = tuple(hulls)
    return hulls


def body_hulls(gripper):
    """[(vertices, triangles)] of the gripper body's collider in its link frame: its mesh's convex hull, fitted by
    hull_pieces (at most HULL_VERTEX_LIMIT vertices each, inside it by at most HULL_TOLERANCE_M)."""
    from core.robots.linear_4310 import read_binary_stl
    pose = gripper.mesh_poses['gripper']
    points = read_binary_stl(gripper.mesh_files['gripper']) @ pose[:3, :3].T + pose[:3, 3]
    return [_hull_mesh(vertices) for vertices, _ in hull_pieces(np.unique(points, axis=0))]


def _author_mesh(stage, path, vertices, faces, approximation, physics_material, *, visible=True, cooking_scale=1.0):
    """A mesh prim; with `approximation` it is also a collider of that kind. A convex hull collider gets PhysX's
    hull vertex limit (its pieces stay within it), HULL_MIN_THICKNESS_M and COLLIDER_CONTACT_OFFSET_M; its points are
    authored `cooking_scale` times larger under the inverse scale, so they land where `vertices` say."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=float).astype(np.float32) * cooking_scale))
    if cooking_scale != 1.0:
        UsdGeom.Xformable(mesh).AddScaleOp().Set(Gf.Vec3f(*([1.0 / cooking_scale] * 3)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, dtype=np.int32).reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr('none')
    if not visible:
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    if approximation is not None:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
        if physics_material is not None:
            from core.workcell.physics import bind_physics
            bind_physics(mesh.GetPrim(), physics_material)
        if approximation == 'convexHull':
            prim = mesh.GetPrim()
            prim.AddAppliedSchema('PhysxConvexHullCollisionAPI')
            prim.CreateAttribute('physxConvexHullCollision:hullVertexLimit', Sdf.ValueTypeNames.Int).Set(
                HULL_VERTEX_LIMIT)
            prim.CreateAttribute('physxConvexHullCollision:minThickness', Sdf.ValueTypeNames.Float).Set(
                HULL_MIN_THICKNESS_M)
            prim.AddAppliedSchema('PhysxCollisionAPI')
            prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(COLLIDER_CONTACT_OFFSET_M)
            prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.0)
    return mesh


def _author_mass(prim, inertial):
    from pxr import Gf, UsdPhysics
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(float(inertial.mass_kg))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*inertial.center_of_mass_m.tolist()))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertial.diagonal_inertia_kg_m2.tolist()))
    w, x, y, z = (float(value) for value in inertial.principal_axes_wxyz)
    mass.CreatePrincipalAxesAttr(Gf.Quatf(w, x, y, z))


def _at_least_position_iterations(stage, path, count):
    """Raise the position iterations of the articulation `path` belongs to (its nearest ancestor with the articulation
    root API) to at least `count`; ValueError if it belongs to none."""
    from pxr import Sdf, UsdPhysics
    prim = stage.GetPrimAtPath(path)
    while prim.IsValid() and not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim = prim.GetParent()
    if not prim.IsValid():
        raise ValueError(f'{path} is in no articulation: the coupled fingers need one')
    name = 'physxArticulation:solverPositionIterationCount'
    attribute = prim.GetAttribute(name)
    current = attribute.Get() if attribute and attribute.HasAuthoredValue() else None
    if current is None or current < count:
        prim.AddAppliedSchema('PhysxArticulationAPI')
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Int).Set(int(count))


def author_linear_4310(stage, root_path, *, flange_path, flange_T_mount, gripper, drive=FingerDrive(),
                       physics_material=None) -> YamPrims:
    """Author the gripper under `root_path` (outside the arm's prims: rigid bodies must not nest) and mount it on the
    rigid body at `flange_path`, at that flange's current world pose."""
    from core.robots.linear_4310 import read_binary_stl
    from pxr import Sdf, UsdGeom, UsdPhysics

    UsdGeom.Xform.Define(stage, root_path)
    world_T_mount = _world(stage, flange_path) @ np.asarray(flange_T_mount, dtype=float)
    links, poses = {}, {}
    for link in LINKS:
        pose = world_T_mount @ gripper.link_frames[link]
        if link in TIPS:
            pose[:3, 3] += world_T_mount[:3, :3] @ gripper.slide_axes[link] * gripper.travel_m
        path = f'{root_path}/{link}'
        xform = UsdGeom.Xform.Define(stage, path)
        _set_world(xform, pose)
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        _author_mass(xform.GetPrim(), gripper.inertials[link])
        mesh_pose = gripper.mesh_poses[link]
        points = read_binary_stl(gripper.mesh_files[link]) @ mesh_pose[:3, :3].T + mesh_pose[:3, 3]
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        _author_mesh(stage, f'{path}/Mesh', unique, np.asarray(inverse).reshape(-1, 3), None, physics_material)
        UsdGeom.Xform.Define(stage, f'{path}/Colliders')
        for index, (vertices, faces) in enumerate(tip_slab_hulls(gripper, link) if link in TIPS else body_hulls(gripper)):
            _author_mesh(stage, f'{path}/Colliders/Hull_{index:03d}', vertices, faces, 'convexHull', physics_material,
                         visible=False, cooking_scale=COOKING_SCALE)
        links[link], poses[link] = path, pose
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(links['tip_left'])).CreateFilteredPairsRel().SetTargets(
        [Sdf.Path(links['tip_right'])])
    joints = {}
    body_T_world = np.linalg.inv(poses['gripper'])
    for tip in TIPS:
        closed = poses[tip].copy()
        closed[:3, 3] -= world_T_mount[:3, :3] @ gripper.slide_axes[tip] * gripper.travel_m
        world_T_joint = np.eye(4)
        world_T_joint[:3, :3] = _frame_with_z(world_T_mount[:3, :3] @ gripper.slide_axes[tip])
        world_T_joint[:3, 3] = closed[:3, 3]
        path = f'{root_path}/{tip}_joint'
        joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(links['gripper'])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(links[tip])])
        joint.CreateAxisAttr('Z')
        joint.CreateLowerLimitAttr(float(gripper.contact_joint_m))
        joint.CreateUpperLimitAttr(float(gripper.travel_m))
        _set_local_frames(joint, body_T_world @ world_T_joint, np.linalg.inv(closed) @ world_T_joint)
        api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        api.CreateTypeAttr('force')
        api.CreateStiffnessAttr(float(drive.stiffness_n_m))
        api.CreateDampingAttr(float(drive.damping_n_s_m))
        api.CreateMaxForceAttr(float(drive.max_force_n))
        api.CreateTargetPositionAttr(float(gripper.travel_m))
        joint.GetPrim().AddAppliedSchema('PhysxJointAPI')
        joint.GetPrim().CreateAttribute('physxJoint:maxJointVelocity', Sdf.ValueTypeNames.Float).Set(
            float(drive.max_speed_m_s))
        joints[tip] = path
    for tip in TIPS:
        if tip != DRIVEN_TIP:  # follows the driven tip: joint value + gearing x the driven one's + offset = 0
            prim = stage.GetPrimAtPath(joints[tip])
            prim.AddAppliedSchema(f'PhysxMimicJointAPI:{MIMIC_AXIS}')
            prim.CreateRelationship(f'physxMimicJoint:{MIMIC_AXIS}:referenceJoint').SetTargets(
                [Sdf.Path(joints[DRIVEN_TIP])])
            prim.CreateAttribute(f'physxMimicJoint:{MIMIC_AXIS}:referenceJointAxis', Sdf.ValueTypeNames.Token).Set(
                MIMIC_AXIS)
            prim.CreateAttribute(f'physxMimicJoint:{MIMIC_AXIS}:gearing', Sdf.ValueTypeNames.Float).Set(-1.0)
            prim.CreateAttribute(f'physxMimicJoint:{MIMIC_AXIS}:offset', Sdf.ValueTypeNames.Float).Set(0.0)
    _at_least_position_iterations(stage, flange_path, POSITION_ITERATIONS)
    mount_path = f'{flange_path}/GripperMountJoint'
    mount = UsdPhysics.FixedJoint.Define(stage, mount_path)
    mount.CreateBody0Rel().SetTargets([Sdf.Path(flange_path)])
    mount.CreateBody1Rel().SetTargets([Sdf.Path(links['gripper'])])
    _set_local_frames(mount, np.asarray(flange_T_mount, dtype=float) @ gripper.link_frames['gripper'], np.eye(4))
    return YamPrims(root_path, links, joints, mount_path, tuple(path.rsplit('/', 1)[1] for path in joints.values()))
