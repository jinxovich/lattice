"""Эталонный МКЭ-путь.

Слой, который умеет решать задачу «честно» и медленно. Всё, что он производит,
служит двум целям: обучающими данными для суррогата и точкой отсчёта, против
которой суррогат измеряется.

Суррогатному слою импорт отсюда запрещён: он обязан работать без ``skfem``,
иначе «ускорение» будет измеряться в присутствии того самого солвера, который
мы собирались обойти.
"""

from pdsopromat.solver.elasticity import (
    MechanicsSolution,
    assemble_body_force,
    assemble_stiffness,
    assemble_thermal_load,
    assemble_traction,
    equilibrium_residual,
    nodal_stress,
    solve_displacement,
)
from pdsopromat.solver.space import FemSpace

__all__ = [
    "FemSpace",
    "MechanicsSolution",
    "assemble_body_force",
    "assemble_stiffness",
    "assemble_thermal_load",
    "assemble_traction",
    "equilibrium_residual",
    "nodal_stress",
    "solve_displacement",
]
