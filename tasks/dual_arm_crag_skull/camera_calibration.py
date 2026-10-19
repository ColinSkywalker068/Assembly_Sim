"""Pinhole intrinsics from what the renderer actually did, not from USD aperture attributes.

Two independent sources: the render product's projection matrix, and a least-squares fit from
one frame rendered as both optical-Z depth `z` and ray distance `r`. For pixel (u, v),
`(r / z)^2 - 1 = ((u - cx) / fx)^2 + ((v - cy) / fy)^2`, which is linear in 1/fx^2 and 1/fy^2.

K follows the OpenCV convention used by perception.depth_to_world: integer (u, v) are pixel
centres, so a centred W x H image has cx = W/2 - 0.5 and cy = H/2 - 0.5.
"""
import numpy as np


def intrinsics_from_projection(projection, width, height, *, max_offset=1e-6):
    """OpenGL-style projection (either memory order) -> K for a centred frustum."""
    p = np.asarray(projection, dtype=float).reshape(4, 4)
    if not np.isfinite(p).all() or p[0, 0] <= 0 or p[1, 1] <= 0:
        raise ValueError('projection must be finite with positive focal terms')
    if max(abs(p[0, 2]), abs(p[1, 2]), abs(p[2, 0]), abs(p[2, 1])) > max_offset:
        raise ValueError('off-centre (aperture offset) projections are not supported')
    fx, fy = p[0, 0] * width / 2, p[1, 1] * height / 2
    return np.array([[fx, 0, width / 2 - 0.5], [0, fy, height / 2 - 0.5], [0, 0, 1.0]])


def intrinsics_from_apertures(focal_length, horizontal_aperture, vertical_aperture, width, height):
    """What the USD attributes claim; only meaningful if the renderer honours vertical aperture."""
    fx = width * focal_length / horizontal_aperture
    fy = height * focal_length / vertical_aperture
    return np.array([[fx, 0, width / 2 - 0.5], [0, fy, height / 2 - 0.5], [0, 0, 1.0]])


def fit_focal_from_depths(optical_z, ray_distance, cx, cy, *, min_pixels=1000):
    """Least-squares (fx, fy) and the RMS residual of the fit, from aligned z and ray images."""
    z = np.asarray(optical_z, dtype=float)
    r = np.asarray(ray_distance, dtype=float)
    if z.shape != r.shape or z.ndim != 2:
        raise ValueError('depth images must be matching HxW arrays')
    v, u = np.indices(z.shape)
    du, dv = u - cx, v - cy
    valid = np.isfinite(z) & np.isfinite(r) & (z > 0) & (r >= z) & ((np.abs(du) > 1) | (np.abs(dv) > 1))
    if valid.sum() < min_pixels:
        raise ValueError(f'only {int(valid.sum())} usable pixels; need {min_pixels}')
    design = np.column_stack([du[valid] ** 2, dv[valid] ** 2])
    target = (r[valid] / z[valid]) ** 2 - 1
    (a, b), *_ = np.linalg.lstsq(design, target, rcond=None)
    if a <= 0 or b <= 0:
        raise ValueError('degenerate fit: depth images do not look like a pinhole camera')
    residual = float(np.sqrt(np.mean((design @ [a, b] - target) ** 2)))
    return float(1 / np.sqrt(a)), float(1 / np.sqrt(b)), residual


def rendered_world_T_usd_camera(params):
    """World pose of the rendered camera (USD axes: -z forward, +y up) from camera_params' row-vector view
    transform, i.e. what the renderer actually used."""
    view = np.asarray(params['cameraViewTransform'], dtype=float).reshape(4, 4)
    pose = np.linalg.inv(view.T)
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise RuntimeError('renderer view transform is not a rigid camera pose')
    return pose


def rendered_world_T_camera_optical(params):
    """The same pose with OpenCV axes (x right, y down, z forward)."""
    return rendered_world_T_usd_camera(params) @ np.diag([1.0, -1.0, -1.0, 1.0])
