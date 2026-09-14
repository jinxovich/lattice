"""Структурная решётка: соответствие поля и DOF-вектора, маска материала.

Главный инвариант здесь — что переход «поле ↔ вектор степеней свободы» есть
представление, а не копия. На нём держится решение отказаться от согласованной
сетки: пока это верно, шва между МКЭ и входом CNN просто не существует.
"""

from __future__ import annotations

import numpy as np
import pytest

from lattice.core import Geometry
from lattice.core.grid import StructuredGrid, material_mask, stiffness_multiplier


@pytest.fixture
def grid() -> StructuredGrid:
    return StructuredGrid(nx=33, ny=33, width=0.1, height=0.1)


class TestLayout:
    def test_coordinates_have_grid_shape(self, grid: StructuredGrid) -> None:
        xx, yy = grid.coordinates()

        assert xx.shape == (33, 33)
        assert yy.shape == (33, 33)

    def test_x_varies_along_first_axis(self, grid: StructuredGrid) -> None:
        # Порядок узлов skfem — y-быстрейший, поэтому первая ось решётки это x.
        xx, yy = grid.coordinates()

        assert xx[0, 0] == pytest.approx(0.0)
        assert xx[-1, 0] == pytest.approx(0.1)
        assert yy[0, -1] == pytest.approx(0.1)
        assert np.allclose(xx[:, 0], xx[:, -1])

    def test_scalar_field_shares_memory_with_dofs(self, grid: StructuredGrid) -> None:
        dofs = np.arange(grid.n_nodes, dtype=np.float64)

        field = grid.as_scalar_field(dofs)

        assert field.shape == (33, 33)
        assert np.shares_memory(field, dofs)

    def test_vector_field_shares_memory_with_dofs(self, grid: StructuredGrid) -> None:
        dofs = np.arange(2 * grid.n_nodes, dtype=np.float64)

        field = grid.as_vector_field(dofs)

        assert field.shape == (33, 33, 2)
        assert np.shares_memory(field, dofs)

    def test_scalar_round_trip_is_lossless(self, grid: StructuredGrid) -> None:
        dofs = np.random.default_rng(0).normal(size=grid.n_nodes)

        assert np.array_equal(grid.as_dofs(grid.as_scalar_field(dofs)), dofs)

    def test_vector_round_trip_is_lossless(self, grid: StructuredGrid) -> None:
        dofs = np.random.default_rng(1).normal(size=2 * grid.n_nodes)

        assert np.array_equal(grid.as_dofs(grid.as_vector_field(dofs)), dofs)

    def test_wrong_dof_count_is_rejected(self, grid: StructuredGrid) -> None:
        with pytest.raises(ValueError, match="степеней свободы"):
            grid.as_scalar_field(np.zeros(grid.n_nodes + 1))

    def test_cell_size_matches_spacing(self, grid: StructuredGrid) -> None:
        hx, hy = grid.cell_size

        assert hx == pytest.approx(0.1 / 32)
        assert hy == pytest.approx(0.1 / 32)

    @pytest.mark.parametrize(("nx", "ny"), [(1, 8), (8, 1), (0, 0)])
    def test_degenerate_grid_is_rejected(self, nx: int, ny: int) -> None:
        # Один узел по стороне означает нулевой шаг ячейки: деление на ноль
        # всплыло бы позже, внутри сборки, где причину уже не видно.
        with pytest.raises(ValueError, match="2 узла"):
            StructuredGrid(nx=nx, ny=ny, width=0.1, height=0.1)


class TestMaterialMask:
    def test_hole_centre_is_void(self, grid: StructuredGrid) -> None:
        mask = material_mask(grid, Geometry(width=0.1, height=0.1, hole_radius=0.02))

        assert mask[16, 16] == pytest.approx(0.0)

    def test_corners_are_solid(self, grid: StructuredGrid) -> None:
        mask = material_mask(grid, Geometry(width=0.1, height=0.1, hole_radius=0.02))

        assert mask[0, 0] == pytest.approx(1.0)
        assert mask[-1, -1] == pytest.approx(1.0)

    def test_sharp_mask_is_strictly_binary(self, grid: StructuredGrid) -> None:
        mask = material_mask(
            grid, Geometry(width=0.1, height=0.1, hole_radius=0.02), interface_fraction=0.0
        )

        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_smoothing_produces_a_transition_band(self, grid: StructuredGrid) -> None:
        mask = material_mask(
            grid, Geometry(width=0.1, height=0.1, hole_radius=0.02), interface_fraction=0.3
        )

        interior = mask[(mask > 0.01) & (mask < 0.99)]
        assert interior.size > 0, "сглаживание не создало переходной полосы"

    def test_transition_width_is_physical_not_mesh_bound(self) -> None:
        # Ширина полосы обязана задаваться геометрией, а не шагом сетки. Иначе
        # измельчение меняет саму задачу, и непрерывного предела, к которому
        # сходиться, не существует — сеточная сходимость становится недостижима.
        geometry = Geometry(width=0.1, height=0.1, hole_radius=0.02)
        fraction = 0.25
        expected = fraction * geometry.hole_radius

        widths = []
        for n in (65, 129, 257):
            fine = StructuredGrid(nx=n, ny=n, width=0.1, height=0.1)
            mask = material_mask(fine, geometry, interface_fraction=fraction)
            # Профиль вдоль оси x от центра: считаем протяжённость полосы перехода.
            profile = mask[n // 2 :, n // 2]
            band = np.count_nonzero((profile > 0.01) & (profile < 0.99))
            widths.append(band * (0.1 / (n - 1)))

        assert all(abs(w - expected) / expected < 0.15 for w in widths), widths

    def test_mask_stays_within_unit_interval(self, grid: StructuredGrid) -> None:
        mask = material_mask(grid, Geometry(width=0.1, height=0.1, hole_radius=0.03))

        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    @pytest.mark.verification
    def test_void_area_converges_to_the_analytic_circle(self) -> None:
        # Количественная проверка маски: доля пустоты обязана сходиться к πR²/(WH).
        # Если сходимости нет, все последующие утверждения о концентрации напряжений
        # опираются на геометрию, отличную от заявленной.
        geometry = Geometry(width=0.1, height=0.1, hole_radius=0.02)
        expected = np.pi * 0.02**2 / (0.1 * 0.1)

        errors = []
        for n in (65, 129, 257):
            fine = StructuredGrid(nx=n, ny=n, width=0.1, height=0.1)
            void_fraction = 1.0 - material_mask(fine, geometry, interface_fraction=0.1).mean()
            errors.append(abs(void_fraction - expected) / expected)

        assert errors[-1] < 0.01, f"грубая маска на 257 узлах: {errors}"
        assert errors[-1] < errors[0], f"ошибка не убывает при измельчении: {errors}"


class TestStiffnessMultiplier:
    def test_solid_material_is_untouched(self) -> None:
        assert stiffness_multiplier(np.array([1.0]), floor=1e-6) == pytest.approx(1.0)

    def test_void_keeps_the_floor_instead_of_zero(self) -> None:
        # Ноль сделал бы матрицу жёсткости вырожденной: узлы внутри отверстия
        # остались бы без единого уравнения.
        assert stiffness_multiplier(np.array([0.0]), floor=1e-6) == pytest.approx(1e-6)

    def test_is_monotone_in_the_mask(self) -> None:
        mask = np.linspace(0.0, 1.0, 11)

        multiplier = stiffness_multiplier(mask, floor=1e-4)

        assert np.all(np.diff(multiplier) > 0.0)
