import json
from types import SimpleNamespace

from validate_scene import validate_fragment_collision_contract


class _Config:
    def __init__(self, layout_path):
        self.layout_path = layout_path

    def resolve_repo_path(self, key):
        assert key == "layout_json"
        return self.layout_path


def _handles(tmp_path, collider_paths):
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "pieces": [
                    {
                        "name": "piece_1",
                        "cells": [[0, 0, 0], [1, 0, 0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fragment = SimpleNamespace(
        name="piece_1",
        root_path="/World/Fragments/piece_1",
        collider_paths=collider_paths,
    )
    return SimpleNamespace(config=_Config(layout_path), fragments=(fragment,))


def test_collision_contract_accepts_only_merged_voxel_boxes(tmp_path):
    handles = _handles(tmp_path, ("/World/Fragments/piece_1/Colliders/Box_000",))

    report = validate_fragment_collision_contract(handles)

    assert report == {"ok": True, "failures": []}


def test_collision_contract_rejects_unoccupied_support_volume(tmp_path):
    handles = _handles(
        tmp_path,
        (
            "/World/Fragments/piece_1/Colliders/Box_000",
            "/World/Fragments/piece_1/Colliders/SupportFootprint",
        ),
    )

    report = validate_fragment_collision_contract(handles)

    assert report["ok"] is False
    assert report["failures"][0]["fragment"] == "piece_1"
