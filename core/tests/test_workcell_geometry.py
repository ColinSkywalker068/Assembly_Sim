import pytest

from core.workcell.geometry import assembly_pad_geometry


def test_assembly_pad_geometry_preserves_surface_and_collider_pose():
    geometry = assembly_pad_geometry(
        {"center": [0.0, 0.0, 0.75], "size": [0.512, 0.416, 0.0096]}
    )

    assert geometry.center == pytest.approx((0.0, 0.0, 0.75))
    assert geometry.size == pytest.approx((0.512, 0.416, 0.0096))
    assert geometry.top_z == pytest.approx(0.75)
    assert geometry.collider_center == pytest.approx((0.0, 0.0, 0.7452))
