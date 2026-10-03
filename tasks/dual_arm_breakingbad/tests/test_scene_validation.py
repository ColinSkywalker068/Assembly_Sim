from types import SimpleNamespace

from tasks.dual_arm_breakingbad.scene.validation import validate_fragment_collision_contract


def _handles(collider_paths):
    layout = SimpleNamespace(
        fragment=lambda name: {"name": name, "cells": [[0, 0, 0], [1, 0, 0]]}
    )
    fragment = SimpleNamespace(
        name="piece_1",
        root_path="/World/Fragments/piece_1",
        collider_paths=collider_paths,
    )
    return SimpleNamespace(layout=layout, fragments=(fragment,))


def test_collision_contract_accepts_only_merged_voxel_boxes():
    handles = _handles(("/World/Fragments/piece_1/Colliders/Box_000",))

    report = validate_fragment_collision_contract(handles)

    assert report == {"ok": True, "failures": []}


def test_collision_contract_rejects_unoccupied_support_volume():
    handles = _handles(
        (
            "/World/Fragments/piece_1/Colliders/Box_000",
            "/World/Fragments/piece_1/Colliders/SupportFootprint",
        ),
    )

    report = validate_fragment_collision_contract(handles)

    assert report["ok"] is False
    assert report["failures"][0]["fragment"] == "piece_1"
