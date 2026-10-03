import pytest

from core.workcell.geometry import assembly_pad_geometry, assembly_tape_segments


def test_assembly_region_geometry_preserves_center_and_footprint():
    geometry = assembly_pad_geometry(
        {"center": [0.0, 0.0, 0.75], "size": [0.55, 0.55, 0.0006]}
    )

    assert geometry.center == pytest.approx((0.0, 0.0, 0.75))
    assert geometry.size == pytest.approx((0.55, 0.55, 0.0006))
    assert geometry.top_z == pytest.approx(0.75)


def test_assembly_tape_segments_form_non_overlapping_outline_on_table():
    segments = assembly_tape_segments(
        {
            "center": [0.0, 0.0, 0.75],
            "size": [0.55, 0.55, 0.0006],
            "tape_width": 0.012,
        }
    )

    assert tuple(segment.name for segment in segments) == (
        "Top",
        "Bottom",
        "Left",
        "Right",
    )
    assert segments[0].center == pytest.approx((0.0, 0.269, 0.7503))
    assert segments[0].size == pytest.approx((0.55, 0.012, 0.0006))
    assert segments[1].center == pytest.approx((0.0, -0.269, 0.7503))
    assert segments[2].center == pytest.approx((-0.269, 0.0, 0.7503))
    assert segments[2].size == pytest.approx((0.012, 0.526, 0.0006))
    assert segments[3].center == pytest.approx((0.269, 0.0, 0.7503))
