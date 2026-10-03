"""Shared PhysX material helpers."""

from __future__ import annotations

from typing import Any

from core.workcell.config import WorkcellConfig


def author_physics_material(stage: Any, path: str, config: WorkcellConfig):
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(config.physics["static_friction"]))
    api.CreateDynamicFrictionAttr(float(config.physics["dynamic_friction"]))
    api.CreateRestitutionAttr(float(config.physics["restitution"]))
    return material


def bind_physics(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
