"""Кешированная пересборка матрицы жёсткости против эталонной сборки skfem.

Приём опирается на линейность слабой формы по множителю жёсткости и на то, что
структура разреженной матрицы постоянна. Обе предпосылки проверяются здесь
сравнением с честной сборкой — до машинной точности, а не «примерно похоже».
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from lattice.core import MaterialParams
from lattice.core.grid import StructuredGrid
from lattice.physics.elasticity import plane_stress_moduli
from lattice.solver import FemSpace, assemble_stiffness
from lattice.solver.assembly import StiffnessAssembler
from lattice.solver.elasticity import stiffness_form


@pytest.fixture
def space() -> FemSpace:
    return FemSpace.from_grid(StructuredGrid(nx=17, ny=17, width=1.0, height=1.0))


@pytest.fixture
def material() -> MaterialParams:
    return MaterialParams(youngs_modulus=1.0, poisson_ratio=0.3)


def reference_element_constant(
    space: FemSpace, material: MaterialParams, element_multiplier: np.ndarray
) -> sp.csr_matrix:
    """Честная сборка skfem с множителем, постоянным на элементе.

    Значения передаются прямо в точки квадратуры, поэтому это в точности та
    дискретизация, которую воспроизводит кеш, — и расхождение обязано быть
    на уровне округления, а не «в пределах пары процентов».
    """
    lam, mu = plane_stress_moduli(material)
    n_quadrature = np.asarray(space.vector.global_coordinates()).shape[2]
    at_quadrature = np.broadcast_to(
        element_multiplier[:, np.newaxis], (element_multiplier.size, n_quadrature)
    )
    return stiffness_form.assemble(
        space.vector, multiplier=at_quadrature, lam=lam, mu=mu
    )


def relative_difference(left: sp.csr_matrix, right: sp.csr_matrix) -> float:
    return float(sp.linalg.norm(left - right) / sp.linalg.norm(right))


class TestExactness:
    def test_reproduces_the_reference_assembly_to_machine_precision(
        self, space: FemSpace, material: MaterialParams
    ) -> None:
        assembler = StiffnessAssembler.build(space, material)
        rng = np.random.default_rng(0)
        multiplier = 0.2 + 0.8 * rng.random((17, 17))

        fast = assembler.assemble(multiplier)
        reference = reference_element_constant(
            space, material, assembler.element_multiplier(multiplier)
        )

        assert relative_difference(fast, reference) < 1e-13

    def test_constant_multiplier_matches_the_nodal_assembly_exactly(
        self, space: FemSpace, material: MaterialParams
    ) -> None:
        # При постоянном множителе поэлементное усреднение и поузловая интерполяция
        # совпадают тождественно, поэтому здесь кеш обязан сойтись с эталонным
        # путём из solver.elasticity без всяких оговорок о дискретизации.
        assembler = StiffnessAssembler.build(space, material)
        multiplier = np.full((17, 17), 0.37)

        assert relative_difference(
            assembler.assemble(multiplier), assemble_stiffness(space, material, multiplier)
        ) < 1e-13

    def test_scales_linearly_with_a_uniform_multiplier(
        self, space: FemSpace, material: MaterialParams
    ) -> None:
        assembler = StiffnessAssembler.build(space, material)

        single = assembler.assemble(np.ones((17, 17)))
        doubled = assembler.assemble(np.full((17, 17), 2.0))

        assert relative_difference(doubled, 2.0 * single) < 1e-13

    def test_element_ordering_is_not_scrambled(
        self, space: FemSpace, material: MaterialParams
    ) -> None:
        # Раскладка coo_data такова, что индекс элемента меняется быстрее всех.
        # Перепутать tile и repeat здесь означало бы получить правдоподобную,
        # симметричную, но соответствующую совсем другому полю матрицу —
        # ошибку, которую по норме или по спектру не увидеть.
        assembler = StiffnessAssembler.build(space, material)
        multiplier = np.ones((17, 17))
        multiplier[:8, :] = 0.01

        fast = assembler.assemble(multiplier)
        reference = reference_element_constant(
            space, material, assembler.element_multiplier(multiplier)
        )

        assert relative_difference(fast, reference) < 1e-13


class TestProperties:
    def test_result_stays_symmetric(self, space: FemSpace, material: MaterialParams) -> None:
        assembler = StiffnessAssembler.build(space, material)
        rng = np.random.default_rng(1)

        matrix = assembler.assemble(0.1 + rng.random((17, 17)))

        assert relative_difference(matrix, matrix.T.tocsr()) < 1e-13

    def test_sparsity_pattern_is_reused(
        self, space: FemSpace, material: MaterialParams
    ) -> None:
        # Структура постоянна, поэтому повторные сборки обязаны давать одинаковое
        # число ненулей. Иначе отображение позиций пришлось бы строить заново,
        # и весь выигрыш исчез бы.
        assembler = StiffnessAssembler.build(space, material)
        rng = np.random.default_rng(2)

        first = assembler.assemble(0.5 * np.ones((17, 17)))
        second = assembler.assemble(0.1 + rng.random((17, 17)))

        assert first.data.size == second.data.size
        assert np.array_equal(first.indptr, second.indptr)
        assert np.array_equal(first.indices, second.indices)

    def test_wrong_shape_is_rejected(
        self, space: FemSpace, material: MaterialParams
    ) -> None:
        assembler = StiffnessAssembler.build(space, material)

        with pytest.raises(ValueError, match="формы"):
            assembler.assemble(np.ones((16, 17)))

    def test_element_multiplier_averages_the_four_corners(
        self, space: FemSpace, material: MaterialParams
    ) -> None:
        assembler = StiffnessAssembler.build(space, material)
        multiplier = np.full((17, 17), 0.25)

        assert np.allclose(assembler.element_multiplier(multiplier), 0.25)
