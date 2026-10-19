"""Bounding-capsule geometry for coarse S0 rejection checks.

This is NOT continuous collision checking and NOT mesh-exact. Each link is replaced by one
capsule that encloses the vertices of its USD collision mesh, and distances are evaluated only
at sampled configurations, so a fast motion can still tunnel between samples and any concave
region is over-approximated. A pass means "no sampled capsule overlap", nothing more.
"""
from __future__ import annotations

import numpy as np


def _capsule_along(points, center, axis):
    offsets = points - center
    along = offsets @ axis
    radial_center = np.outer(along, axis)
    # Center the axis line on the radial bounding box so the radius is not inflated by an offset.
    lateral = offsets - radial_center
    basis = np.linalg.svd(np.eye(3) - np.outer(axis, axis))[0][:, :2]
    planar = lateral @ basis
    shift = basis @ ((planar.max(axis=0) + planar.min(axis=0)) / 2)
    center = center + shift
    offsets = points - center
    along = offsets @ axis
    radial = np.linalg.norm(offsets - np.outer(along, axis), axis=1)
    radius = float(radial.max())
    slack = np.sqrt(np.maximum(radius * radius - radial * radial, 0.0))
    # A point (along=t, radial=d) is covered iff t0 - slack <= t <= t1 + slack.
    start, stop = float((along + slack).min()), float((along - slack).max())
    if start > stop:
        start = stop = (start + stop) / 2
    return center + start * axis, center + stop * axis, radius


def fit_capsule(points):
    """Bounding capsule (p0, p1, radius) enclosing every point.

    Tries the three principal axes and the three coordinate axes and keeps the smallest volume;
    p0 == p1 (a sphere) when the point set is short along the chosen axis.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('need a finite (N>=2, 3) point set')
    center = points.mean(axis=0)
    candidates = [*np.linalg.svd(points - center, full_matrices=False)[2], *np.eye(3)]
    best = None
    for axis in candidates:
        p0, p1, radius = _capsule_along(points, center, axis / np.linalg.norm(axis))
        volume = np.pi * radius ** 2 * np.linalg.norm(p1 - p0) + 4 / 3 * np.pi * radius ** 3
        if best is None or volume < best[0]:
            best = (volume, p0, p1, radius)
    return best[1], best[2], best[3]


def segment_distances(p0, p1, q0, q1):
    """Vectorized closest distance between segments [p0, p1] and [q0, q1], shapes (..., 3).

    Ericson, Real-Time Collision Detection, section 5.1.9, including degenerate segments.
    """
    p0, p1, q0, q1 = (np.asarray(v, dtype=float) for v in (p0, p1, q0, q1))
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.sum(d1 * d1, axis=-1)
    e = np.sum(d2 * d2, axis=-1)
    f = np.sum(d2 * r, axis=-1)
    c = np.sum(d1 * r, axis=-1)
    b = np.sum(d1 * d2, axis=-1)
    tiny = 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = a * e - b * b
        s = np.where(denom > tiny, np.clip((b * f - c * e) / denom, 0.0, 1.0), 0.0)
        s = np.where(a <= tiny, 0.0, s)
        t = np.where(e > tiny, (b * s + f) / e, 0.0)
        # t outside [0, 1]: clamp it and recompute s for the clamped t.
        s_low = np.where(a > tiny, np.clip(-c / a, 0.0, 1.0), 0.0)
        s_high = np.where(a > tiny, np.clip((b - c) / a, 0.0, 1.0), 0.0)
        s = np.where(t < 0.0, s_low, np.where(t > 1.0, s_high, s))
        t = np.clip(t, 0.0, 1.0)
        # Second segment degenerate (a point): closest point on the first segment to it.
        s = np.where(e <= tiny, s_low, s)
    closest_p = p0 + s[..., None] * d1
    closest_q = q0 + t[..., None] * d2
    return np.linalg.norm(closest_p - closest_q, axis=-1)


def capsule_clearance(first, second):
    """Minimum signed clearance between two capsule sets (P0, P1, R) with shapes (K,3),(K,3),(K,).

    Returns (clearance_m, index_in_first, index_in_second); negative clearance means overlap.
    """
    a0, a1, ar = first
    b0, b1, br = second
    distance = segment_distances(a0[:, None, :], a1[:, None, :], b0[None, :, :], b1[None, :, :])
    clearance = distance - ar[:, None] - br[None, :]
    i, j = np.unravel_index(np.argmin(clearance), clearance.shape)
    return float(clearance[i, j]), int(i), int(j)


def lowest_point_z(capsules):
    """Lowest z of each capsule (P0, P1, R) -> (K,)."""
    p0, p1, radius = capsules
    return np.minimum(p0[:, 2], p1[:, 2]) - radius
