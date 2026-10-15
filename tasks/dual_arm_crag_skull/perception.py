"""Metric RGB-D geometry; assumes aligned, undistorted optical-frame depth."""
import numpy as np


def rigid_transform(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('expected finite 4x4 transform')
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('invalid homogeneous row')
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError('transform must be rigid SE(3), not scale/reflection')
    return t


def depth_to_world(depth, mask, intrinsics, world_T_camera, depth_scale_m=1.0):
    """Depth is optical Z, not Euclidean ray distance. Returns Nx3 metres."""
    depth = np.asarray(depth, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    k = np.asarray(intrinsics, dtype=float)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError('depth/mask must have matching HxW dimensions')
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('invalid intrinsics')
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError('expected rectified zero-skew intrinsics')
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError('invalid depth scale')
    z = depth * depth_scale_m
    v, u = np.nonzero(mask & np.isfinite(z) & (z > 0))
    z = z[v, u]
    camera = np.column_stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z))
    t = rigid_transform(world_T_camera)
    return camera @ t[:3, :3].T + t[:3, 3]


def anchor_goals(assembly_poses, anchor_id, world_T_anchor):
    """Map predicted local-part -> assembly poses to one observed world anchor."""
    world_T_assembly = rigid_transform(world_T_anchor) @ np.linalg.inv(rigid_transform(assembly_poses[anchor_id]))
    return {key: world_T_assembly @ rigid_transform(value) for key, value in assembly_poses.items()}


def refine_pose(model_points, observed_world_points, initial_world_T_part,
                max_distance_m=.01, iterations=30):
    """Local ICP only: caller supplies correct part identity and coarse pose.

    Partial observation -> full-model nearest matches. Scores are residuals,
    not calibrated confidence; symmetry/occlusion require external rejection.
    """
    from scipy.spatial import cKDTree
    model = np.asarray(model_points, dtype=float)
    observed = np.asarray(observed_world_points, dtype=float)
    for points in (model, observed):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6 or not np.isfinite(points).all():
            raise ValueError('need finite Nx3 point sets, at least six points')
    if not np.isfinite(max_distance_m) or max_distance_m <= 0 or not 1 <= iterations <= 200:
        raise ValueError('invalid ICP limits')
    transform = rigid_transform(initial_world_T_part).copy()
    for _ in range(iterations):
        positioned = model @ transform[:3,:3].T + transform[:3,3]
        distance, indices = cKDTree(positioned).query(observed)
        valid = distance <= max_distance_m
        if valid.sum() < 6:
            raise ValueError('insufficient correspondences; reobserve or reinitialize')
        source, target = positioned[indices[valid]], observed[valid]
        source_center, target_center = source.mean(0), target.mean(0)
        h = (source-source_center).T @ (target-target_center)
        u, singular, vt = np.linalg.svd(h)
        if singular[1] < 1e-10:
            raise ValueError('degenerate ICP correspondences')
        correction = np.eye(3); correction[2,2] = np.linalg.det(vt.T @ u.T)
        rotation = vt.T @ correction @ u.T
        delta = np.eye(4); delta[:3,:3] = rotation
        delta[:3,3] = target_center-rotation @ source_center
        transform = delta @ transform
        if np.linalg.norm(delta-np.eye(4)) < 1e-7:
            break
    positioned = model @ transform[:3,:3].T + transform[:3,3]
    distance, _ = cKDTree(positioned).query(observed)
    valid = distance <= max_distance_m
    if valid.sum() < 6:
        raise ValueError('final registration has insufficient overlap')
    return {'world_T_part':transform, 'inlier_rmse_m':float(np.sqrt(np.mean(distance[valid]**2))),
            'observed_inlier_fraction':float(valid.mean()), 'requires_identity_and_symmetry_check':True}
