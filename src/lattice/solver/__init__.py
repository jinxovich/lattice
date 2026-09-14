"""Эталонный МКЭ-путь.

Слой, который умеет решать задачу «честно» и медленно. Всё, что он производит,
служит двум целям: обучающими данными для суррогата и точкой отсчёта, против
которой суррогат измеряется.

Суррогатному слою импорт отсюда запрещён: он обязан работать без ``skfem``,
иначе «ускорение» будет измеряться в присутствии того самого солвера, который
мы собирались обойти.
"""

from lattice.solver.assembly import StiffnessAssembler
from lattice.solver.elasticity import (
    MechanicsSolution,
    assemble_body_force,
    assemble_stiffness,
    assemble_thermal_load,
    assemble_traction,
    equilibrium_residual,
    nodal_stress,
    solve_displacement,
)
from lattice.solver.linear import SolveReport, WarmSolver
from lattice.solver.screening import ScreeningOperator
from lattice.solver.space import FemSpace
from lattice.solver.thermal import ThermalOperator, ThermalStepper

__all__ = [
    "FemSpace",
    "MechanicsSolution",
    "ScreeningOperator",
    "SolveReport",
    "StiffnessAssembler",
    "ThermalOperator",
    "ThermalStepper",
    "WarmSolver",
    "assemble_body_force",
    "assemble_stiffness",
    "assemble_thermal_load",
    "assemble_traction",
    "equilibrium_residual",
    "nodal_stress",
    "solve_displacement",
]
