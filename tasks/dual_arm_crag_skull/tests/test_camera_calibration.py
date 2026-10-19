import numpy as np
import pytest

from tasks.dual_arm_crag_skull.camera_calibration import (
    fit_focal_from_depths, intrinsics_from_apertures, intrinsics_from_projection, rendered_world_T_camera_optical,
    rendered_world_T_usd_camera)


def render_depths(fx, fy, width=320, height=240, seed=0):
    """Optical Z and ray distance of a tilted plane seen by an ideal pinhole camera."""
    v, u = np.indices((height, width))
    x, y = (u + 0.5 - width / 2) / fx, (v + 0.5 - height / 2) / fy
    rng = np.random.default_rng(seed)
    normal = np.array([0.1, -0.3, 1.0]) / np.linalg.norm([0.1, -0.3, 1.0])
    z = 2.0 / (normal[0] * x + normal[1] * y + normal[2]) + rng.normal(scale=1e-6, size=x.shape)
    ray = z * np.sqrt(1 + x ** 2 + y ** 2)
    return z, ray


def test_fit_recovers_non_square_pixels():
    z, ray = render_depths(400.0, 390.0)
    fx, fy, residual = fit_focal_from_depths(z, ray, 159.5, 119.5)
    assert fx == pytest.approx(400.0, rel=1e-3) and fy == pytest.approx(390.0, rel=1e-3)
    assert residual < 1e-4


def test_fit_rejects_ray_distance_mistaken_for_z():
    z, _ = render_depths(400.0, 400.0)
    with pytest.raises(ValueError, match='degenerate'):
        fit_focal_from_depths(z, z, 159.5, 119.5)


def test_projection_layout_independent_and_offset_rejected():
    projection = np.diag([2 * 400 / 320, 2 * 300 / 240, -1.0, 0.0])
    projection[2, 3], projection[3, 2] = -0.02, -1.0
    k = intrinsics_from_projection(projection, 320, 240)
    np.testing.assert_allclose(k, intrinsics_from_projection(projection.T.ravel(), 320, 240))
    assert (k[0, 0], k[1, 1]) == pytest.approx((400.0, 300.0))
    assert (k[0, 2], k[1, 2]) == (159.5, 119.5)  # OpenCV: integer pixel indices are pixel centres
    projection[0, 2] = 0.1
    with pytest.raises(ValueError, match='off-centre'):
        intrinsics_from_projection(projection, 320, 240)


def test_report_discrepancy_is_the_default_usd_vertical_aperture():
    # reports/2026-10-14.json agent camera: fx 2290.6228, fy 2354.3569 at 1920x1440. A 25 mm lens with
    # the USD default apertures (20.955 x 15.2908 mm, aspect 1.37) reproduces both numbers, while
    # the 4:3 render product implies a 15.71625 mm vertical aperture for square pixels.
    k = intrinsics_from_apertures(25.0, 20.955, 15.2908, 1920, 1440)
    assert (k[0, 0], k[1, 1]) == pytest.approx((2290.6227714035144, 2354.356853615721), rel=1e-6)
    square = intrinsics_from_apertures(25.0, 20.955, 20.955 * 1440 / 1920, 1920, 1440)
    assert square[1, 1] == pytest.approx(square[0, 0])


def test_principal_point_matches_depth_to_world_pixel_convention():
    from tasks.dual_arm_crag_skull.perception import depth_to_world
    width, height, fx = 320, 240, 400.0
    z, _ = render_depths(fx, fx, width, height)
    k = intrinsics_from_projection(np.diag([2 * fx / width, 2 * fx / height, -1.0, 0.0]), width, height)
    mask = np.zeros_like(z, dtype=bool)
    mask[0, 0] = mask[height - 1, width - 1] = True
    points = depth_to_world(z, mask, k, np.eye(4))
    # The ideal camera that rendered the plane: pixel centre (u + 0.5) relative to the image centre.
    expected_x = (np.array([0, width - 1]) + 0.5 - width / 2) / fx * z[[0, height - 1], [0, width - 1]]
    np.testing.assert_allclose(points[:, 0], expected_x, rtol=1e-12)


def view_params(world_T_usd):
    """camera_params as the renderer gives them: the row-vector view transform, flattened."""
    return {'cameraViewTransform': np.linalg.inv(world_T_usd).T.reshape(-1).tolist()}


def test_rendered_camera_pose_from_the_view_transform():
    angle = np.radians(30)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    pose[:3, 3] = [0.1, -2.0, 3.4]
    np.testing.assert_allclose(rendered_world_T_usd_camera(view_params(pose)), pose, atol=1e-12)
    optical = rendered_world_T_camera_optical(view_params(pose))
    np.testing.assert_allclose(optical[:3, 2], -pose[:3, 2], atol=1e-12)  # OpenCV z = forward = USD -z
    np.testing.assert_allclose(optical[:3, 1], -pose[:3, 1], atol=1e-12)
    np.testing.assert_allclose(optical[:3, 0], pose[:3, 0], atol=1e-12)
    skewed = pose.copy()
    skewed[0, 1] += 0.1
    with pytest.raises(RuntimeError, match='rigid'):
        rendered_world_T_usd_camera(view_params(skewed))
