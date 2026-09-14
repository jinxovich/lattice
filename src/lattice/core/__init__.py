"""Базовые типы и инварианты, общие для всех слоёв."""

from lattice.core.grid import StructuredGrid, material_mask, stiffness_multiplier
from lattice.core.spec import (
    SCHEMA_VERSION,
    CaseSpec,
    DamageParams,
    Geometry,
    LoadProgram,
    MaterialParams,
    MeshSpec,
)
from lattice.core.state import State

__all__ = [
    "SCHEMA_VERSION",
    "CaseSpec",
    "DamageParams",
    "Geometry",
    "LoadProgram",
    "MaterialParams",
    "MeshSpec",
    "State",
    "StructuredGrid",
    "material_mask",
    "stiffness_multiplier",
]
