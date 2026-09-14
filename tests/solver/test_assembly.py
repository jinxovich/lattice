"""Сборка и решение задачи упругости: частные случаи с независимым ответом."""

from __future__ import annotations

import numpy as np
import pytest

from lattice.core import CaseSpec, MaterialParams
from lattice.core.grid import StructuredGrid
from lattice.solver import (
    FemSpace,
    assemble_stiffness,
    assemble_thermal_load,
    assemble_traction,
    equilibrium_residual,
    nodal_stress,
    solve_displacement,
)

TOL = 1e-9


@pytest.fixture
def space() -> FemSpace:
    return FemSpace.from_grid(StructuredGrid(nx=9, ny=9, width=1.0, height=1.0))


@pytest.fixture
def solid(space: FemSpace) -> np.ndarray:
    """Сплошной материал без отверстия и без повреждений."""
    return np.ones((space.grid.nx, space.grid.ny))


def symmetry_dofs(space: FemSpace) -> np.ndarray:
    """u_x = 0 на левой кромке, u_y = 0 на нижней — четвертьсимметричное закрепление."""
    left = space.vector.get_dofs(lambda x: x[0] < TOL).all("u^1")
    bottom = space.vector.get_dofs(lambda x: x[1] < TOL).all("u^2")
    return np.concatenate([left, bottom])


class TestNodeOrdering:
    def test_mesh_matches_the_grid_layout(self, space: FemSpace) -> None:
        expected_x, expected_y = space.grid.coordinates()

        assert np.allclose(space.mesh.p[0].reshape(9, 9), expected_x)
        assert np.allclose(space.mesh.p[1].reshape(9, 9), expected_y)

    def test_unknown_edge_is_rejected(self, space: FemSpace) -> None:
        with pytest.raises(ValueError, match="кромка"):
            space.facet_basis("diagonal")

    def test_broken_ordering_invariant_is_caught(self) -> None:
        # Сработать эта защита может только при смене версии skfem или конструктора
        # сетки. Проверяем, что она вообще способна сработать: иначе это мёртвый код,
        # создающий ложное ощущение безопасности.
        good = FemSpace.from_grid(StructuredGrid(nx=5, ny=9, width=1.0, height=2.0))
        transposed = FemSpace(
            grid=StructuredGrid(nx=9, ny=5, width=2.0, height=1.0),
            mesh=good.mesh,
            vector=good.vector,
            scalar=good.scalar,
        )

        with pytest.raises(RuntimeError, match="транспонированными"):
            transposed._assert_node_ordering()


class TestPatchTest:
    """Постоянное напряжённое состояние обязано воспроизводиться точно."""

    @pytest.mark.verification
    def test_uniform_traction_gives_uniform_displacement(
        self, space: FemSpace, solid: np.ndarray
    ) -> None:
        material = MaterialParams(youngs_modulus=1.0, poisson_ratio=0.3)
        traction = 1.0e-3

        stiffness = assemble_stiffness(space, material, solid)
        load = assemble_traction(space, "right", (traction, 0.0))
        dofs = solve_displacement(stiffness, load, symmetry_dofs(space))

        field = space.grid.as_vector_field(dofs)
        # Точное решение: u_x = (t/E)·x, u_y = −ν(t/E)·y. Q1 воспроизводит его без ошибки.
        assert np.allclose(field[-1, :, 0], traction / material.youngs_modulus, rtol=1e-10)
        assert np.allclose(
            field[:, -1, 1], -material.poisson_ratio * traction / material.youngs_modulus,
            rtol=1e-10,
        )

    @pytest.mark.verification
    def test_recovered_stress_matches_the_applied_traction(
        self, space: FemSpace, solid: np.ndarray
    ) -> None:
        spec = CaseSpec(material=MaterialParams(youngs_modulus=1.0, poisson_ratio=0.3))
        traction = 1.0e-3

        stiffness = assemble_stiffness(space, spec.material, solid)
        load = assemble_traction(space, "right", (traction, 0.0))
        dofs = solve_displacement(stiffness, load, symmetry_dofs(space))

        temperature = np.full((9, 9), spec.material.reference_temperature)
        stress = nodal_stress(space, spec, space.grid.as_vector_field(dofs), solid, temperature)

        assert np.allclose(stress[..., 0], traction, rtol=1e-9)
        assert np.allclose(stress[..., 1], 0.0, atol=1e-12)
        assert np.allclose(stress[..., 2], 0.0, atol=1e-12)


class TestThermalLoading:
    @pytest.mark.verification
    def test_fully_constrained_heating_reproduces_the_analytic_stress(
        self, space: FemSpace, solid: np.ndarray
    ) -> None:
        material = MaterialParams(
            youngs_modulus=2.0e11, poisson_ratio=0.3, thermal_expansion=1.2e-5
        )
        spec = CaseSpec(material=material)
        delta_t = 100.0
        temperature = np.full((9, 9), material.reference_temperature + delta_t)

        stiffness = assemble_stiffness(space, material, solid)
        load = assemble_thermal_load(space, material, solid, temperature)
        # Закрепление по всему контуру: расширяться некуда.
        clamped = space.vector.get_dofs(
            lambda x: (x[0] < TOL) | (x[0] > 1.0 - TOL) | (x[1] < TOL) | (x[1] > 1.0 - TOL)
        ).all()
        dofs = solve_displacement(stiffness, load, clamped)

        stress = nodal_stress(
            space, spec, space.grid.as_vector_field(dofs), solid, temperature
        )
        expected = (
            -material.youngs_modulus
            * material.thermal_expansion
            * delta_t
            / (1.0 - material.poisson_ratio)
        )

        interior = stress[2:-2, 2:-2, 0]
        assert np.allclose(interior, expected, rtol=1e-6)

    def test_unconstrained_heating_produces_no_stress(
        self, space: FemSpace, solid: np.ndarray
    ) -> None:
        # Тело, которому дали расшириться, напряжений не несёт независимо от ΔT.
        material = MaterialParams()
        spec = CaseSpec(material=material)
        temperature = np.full((9, 9), material.reference_temperature + 250.0)

        stiffness = assemble_stiffness(space, material, solid)
        load = assemble_thermal_load(space, material, solid, temperature)
        dofs = solve_displacement(stiffness, load, symmetry_dofs(space))

        stress = nodal_stress(
            space, spec, space.grid.as_vector_field(dofs), solid, temperature
        )

        assert np.max(np.abs(stress)) < 1.0e-3 * material.youngs_modulus * 1e-6


class TestDegradation:
    def test_softer_material_deforms_more(self, space: FemSpace, solid: np.ndarray) -> None:
        material = MaterialParams(youngs_modulus=1.0)
        damaged = np.full_like(solid, 0.5)

        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        dofs = symmetry_dofs(space)

        intact = solve_displacement(assemble_stiffness(space, material, solid), load, dofs)
        weakened = solve_displacement(assemble_stiffness(space, material, damaged), load, dofs)

        # Однородная деградация вдвое удваивает перемещения ровно.
        assert np.allclose(weakened, 2.0 * intact, rtol=1e-9)

    def test_localised_damage_concentrates_deformation(
        self, space: FemSpace, solid: np.ndarray
    ) -> None:
        material = MaterialParams(youngs_modulus=1.0)
        weakened = solid.copy()
        weakened[4, :] = 0.05

        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        dofs = symmetry_dofs(space)

        intact = space.grid.as_vector_field(
            solve_displacement(assemble_stiffness(space, material, solid), load, dofs)
        )
        soft = space.grid.as_vector_field(
            solve_displacement(assemble_stiffness(space, material, weakened), load, dofs)
        )

        assert soft[-1, :, 0].mean() > intact[-1, :, 0].mean()


class TestEquilibriumResidual:
    def test_exact_solution_has_vanishing_residual(
        self, space: FemSpace, solid: np.ndarray
    ) -> None:
        material = MaterialParams(youngs_modulus=1.0)
        stiffness = assemble_stiffness(space, material, solid)
        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        dofs = symmetry_dofs(space)

        exact = solve_displacement(stiffness, load, dofs)

        assert equilibrium_residual(stiffness, load, exact, dofs) < 1e-10

    def test_perturbed_solution_is_detected(self, space: FemSpace, solid: np.ndarray) -> None:
        # Это и есть индикатор для быстрого режима суррогата: один матвек
        # отличает правдоподобное поле от равновесного.
        material = MaterialParams(youngs_modulus=1.0)
        stiffness = assemble_stiffness(space, material, solid)
        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        dofs = symmetry_dofs(space)
        exact = solve_displacement(stiffness, load, dofs)

        rng = np.random.default_rng(0)
        perturbed = exact * (1.0 + 0.01 * rng.normal(size=exact.shape))

        assert equilibrium_residual(stiffness, load, perturbed, dofs) > 1e-3

    def test_zero_load_falls_back_to_absolute_norm(
        self, space: FemSpace, solid: np.ndarray
    ) -> None:
        material = MaterialParams(youngs_modulus=1.0)
        stiffness = assemble_stiffness(space, material, solid)
        load = np.zeros(stiffness.shape[0])
        dofs = symmetry_dofs(space)

        assert equilibrium_residual(stiffness, load, np.zeros_like(load), dofs) == 0.0
