"""Концентрация напряжений у отверстия — проверка погружённой границы.

Проверяется самое рискованное решение проекта: геометрия задаётся маской на
структурной сетке, а не согласованной сеткой. Цена приёма — размытая граница
отверстия, и замерить её нужно именно там, где она опаснее всего: у концентратора,
где зарождается повреждение.

Два урока, ради которых стоит читать этот файл целиком.

**Мерить нужно в фиксированных физических точках.** Первая редакция брала максимум
по узлам сплошного материала, и «сходимости» не было: 3.53 → 3.50 → 3.52 → 3.76.
Причина не в солвере — набор таких узлов смещается в пространстве при смене сетки,
а пик Кирша крайне локален (уже на 1.1R он падает с 3.0 до 2.44). Сравнивались
значения в разных точках.

**Ширина перехода обязана быть физической.** Пока она задавалась в ячейках, маска
описывала геометрию, зависящую от сетки: при измельчении полоса сжималась, маска
стремилась к ступеньке, и углы пикселей давали искусственную концентрацию.
Непрерывной задачи, к которой могло бы сходиться решение, при этом не существует.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from pdsopromat.core import CaseSpec, Geometry, MaterialParams
from pdsopromat.core.grid import StructuredGrid, material_mask, stiffness_multiplier
from pdsopromat.solver import (
    FemSpace,
    assemble_stiffness,
    assemble_traction,
    nodal_stress,
    solve_displacement,
)

TOL = 1e-9
FAR_FIELD_STRESS = 1.0e-3
STIFFNESS_FLOOR = 1.0e-6


def kirsch_profile(r_over_radius: float) -> float:
    """σ_xx(0, y)/σ_∞ на оси, перпендикулярной нагрузке, для бесконечной пластины.

    σ_xx = σ_∞/2·[2 + (R/y)² + 3(R/y)⁴]. При y = R даёт 3 — ту самую тройку.
    """
    q = 1.0 / r_over_radius
    return 0.5 * (2.0 + q**2 + 3.0 * q**4)


def finite_width_factor(diameter_over_width: float) -> float:
    """K_t по брутто-сечению для полосы конечной ширины: [2+(1−d/W)³]/(1−d/W).

    При d/W → 0 стремится к 3. Нужна, чтобы отличить ошибку солвера от честного
    влияния близких границ — без неё любое отклонение от тройки выглядит дефектом.
    """
    ligament = 1.0 - diameter_over_width
    return (2.0 + ligament**3) / ligament


def solve_plate(
    nx: int,
    ny: int,
    radius: float,
    *,
    width: float = 1.0,
    height: float = 1.0,
    interface_fraction: float = 0.15,
) -> tuple[StructuredGrid, np.ndarray, np.ndarray]:
    """Пластина с центральным отверстием под двусторонним растяжением.

    Обе осевые линии — плоскости симметрии задачи, поэтому закрепление по ним
    точно и не требует точечных пиннингов, которые сами создают возмущение поля.
    Возвращает решётку, нормированное поле σ_xx/σ_∞ и маску.
    """
    geometry = Geometry(
        width=width, height=height, thickness=1.0e-3, hole_radius=radius
    )
    grid = StructuredGrid(nx=nx, ny=ny, width=width, height=height)
    space = FemSpace.from_grid(grid)

    mask = material_mask(grid, geometry, interface_fraction=interface_fraction)
    multiplier = stiffness_multiplier(mask, STIFFNESS_FLOOR)
    material = MaterialParams(youngs_modulus=1.0, poisson_ratio=0.3)

    stiffness = assemble_stiffness(space, material, multiplier)
    load = assemble_traction(space, "right", (FAR_FIELD_STRESS, 0.0)) + assemble_traction(
        space, "left", (-FAR_FIELD_STRESS, 0.0)
    )
    constrained = np.concatenate(
        [
            space.vector.get_dofs(lambda x: np.abs(x[0] - 0.5 * width) < TOL).all("u^1"),
            space.vector.get_dofs(lambda x: np.abs(x[1] - 0.5 * height) < TOL).all("u^2"),
        ]
    )
    displacement = solve_displacement(stiffness, load, constrained)

    temperature = np.full((nx, ny), material.reference_temperature)
    stress = nodal_stress(
        space,
        CaseSpec(material=material, geometry=geometry),
        space.grid.as_vector_field(displacement),
        multiplier,
        temperature,
    )
    return grid, stress[..., 0] / FAR_FIELD_STRESS, mask


def sample_on_axis(
    grid: StructuredGrid, field: np.ndarray, distance: float
) -> float:
    """Значение поля на оси y, на заданном расстоянии от центра отверстия."""
    interpolate = RegularGridInterpolator((grid.x, grid.y), field)
    return float(interpolate((0.5 * grid.width, 0.5 * grid.height + distance)))


@pytest.mark.verification
@pytest.mark.slow
class TestKirsch:
    def test_far_field_recovers_the_applied_traction(self) -> None:
        # Базовая проверка постановки: вдали от отверстия возмущение затухает,
        # и поле обязано вернуться к приложенному растяжению.
        radius = 0.08
        grid, stress, _ = solve_plate(97, 289, radius, height=3.0)

        assert sample_on_axis(grid, stress, 1.2) == pytest.approx(1.0, abs=0.02)

    def test_profile_matches_the_analytic_solution(self) -> None:
        # Область вытянута (H/W = 3), чтобы приблизиться к условиям бесконечной
        # пластины: на квадрате близкие границы сами дают заметный вклад.
        radius = 0.08
        grid, stress, _ = solve_plate(129, 385, radius, height=3.0)

        # Остаточное отклонение объясняется конечной шириной: при d/W = 0.16
        # поправка даёт +2.9% к тройке, что и наблюдается.
        expected_bias = finite_width_factor(2.0 * radius) / 3.0

        for ratio in (1.3, 2.0, 3.0, 5.0):
            measured = sample_on_axis(grid, stress, ratio * radius)
            analytic = kirsch_profile(ratio) * expected_bias

            assert measured == pytest.approx(analytic, rel=0.03), f"на r={ratio}R"

    def test_converges_at_fixed_physical_points(self) -> None:
        # Главный тест файла. Точки фиксированы в пространстве, поэтому сравниваются
        # действительно одни и те же величины — в отличие от максимума по узлам,
        # чей набор смещается вместе с сеткой.
        radius = 0.15
        samples = []

        for n in (97, 129, 161):
            grid, stress, _ = solve_plate(n, n, radius)
            samples.append(
                [sample_on_axis(grid, stress, k * radius) for k in (1.5, 2.0)]
            )

        for column in range(2):
            values = [row[column] for row in samples]
            spread = (max(values) - min(values)) / float(np.mean(values))
            assert spread < 0.01, f"столбец {column} не сошёлся: {values}"

    def test_interface_width_barely_affects_the_solid_region(self) -> None:
        # Погружённая граница размывает поле вблизи себя, но вдали от неё
        # результат не должен зависеть от того, насколько широка полоса.
        radius = 0.15
        measured = []

        for fraction in (0.10, 0.20, 0.30):
            grid, stress, _ = solve_plate(129, 129, radius, interface_fraction=fraction)
            measured.append(sample_on_axis(grid, stress, 1.5 * radius))

        spread = (max(measured) - min(measured)) / float(np.mean(measured))
        assert spread < 0.02, f"ширина перехода протекает в сплошной материал: {measured}"

    def test_peak_at_the_interface_is_overestimated_by_a_sharp_mask(self) -> None:
        # Документированное ограничение приёма, а не дефект. Ступенчатая маска даёт
        # искусственную концентрацию на углах пикселей поверх настоящей, и мерить
        # напряжение прямо на границе нельзя ни в том, ни в другом варианте.
        #
        # Практическое следствие для движка: движущую силу повреждаемости берём
        # из НЕЛОКАЛЬНОГО поля. Скрининг на длине l_c сглаживает именно ту зону
        # шириной в переходную полосу, где значения ненадёжны — регуляризация,
        # введённая ради сеточной объективности, попутно подавляет и этот артефакт.
        radius = 0.15

        _, sharp, sharp_mask = solve_plate(129, 129, radius, interface_fraction=0.0)
        _, graded, graded_mask = solve_plate(129, 129, radius, interface_fraction=0.2)

        sharp_peak = sharp[sharp_mask > 0.99].max()
        graded_peak = graded[graded_mask > 0.99].max()

        assert sharp_peak > 1.1 * graded_peak, f"резкая {sharp_peak}, сглаженная {graded_peak}"
