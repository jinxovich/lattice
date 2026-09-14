"""Нестационарная теплопроводность: стационар, аналитика слоя, сохранение тепла.

Центральный тест — сверка с рядом Фурье для одномерного слоя. У задачи есть
замкнутое решение, поэтому проверяется не правдоподобие поля, а совпадение чисел.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdsopromat.core import MaterialParams
from pdsopromat.core.grid import StructuredGrid
from pdsopromat.solver import FemSpace
from pdsopromat.solver.thermal import BACKWARD_EULER, CRANK_NICOLSON, ThermalOperator

TOL = 1e-9
NX, NY = 33, 5


@pytest.fixture
def material() -> MaterialParams:
    """Единичная температуропроводность: α = k/(ρc) = 1, числа остаются читаемыми."""
    return MaterialParams(
        conductivity=1.0, density=1.0, specific_heat=1.0, reference_temperature=1.0
    )


@pytest.fixture
def space() -> FemSpace:
    return FemSpace.from_grid(StructuredGrid(nx=NX, ny=NY, width=1.0, height=0.25))


@pytest.fixture
def operator(space: FemSpace, material: MaterialParams) -> ThermalOperator:
    return ThermalOperator.build(space, material)


def side_dofs(space: FemSpace) -> np.ndarray:
    """Только левая и правая кромки: верх и низ остаются теплоизолированными.

    Естественное граничное условие даёт нулевой поток, поэтому задача становится
    одномерной по x — той самой, для которой известен ряд.
    """
    return np.asarray(
        space.scalar.get_dofs(lambda x: (x[0] < TOL) | (x[0] > 1.0 - TOL)).all()
    )


def slab_series(x: np.ndarray, time: float, terms: int = 40) -> np.ndarray:
    """Решение ∂θ/∂t = ∂²θ/∂x² при θ(x,0)=1, θ(0,t)=θ(1,t)=0.

    θ = Σ_{n нечётные} (4/nπ)·sin(nπx)·exp(−(nπ)²t)
    """
    total = np.zeros_like(x)
    for n in range(1, 2 * terms, 2):
        total += (4.0 / (n * np.pi)) * np.sin(n * np.pi * x) * np.exp(-((n * np.pi) ** 2) * time)
    return total


class TestValidation:
    def test_non_positive_step_is_rejected(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        with pytest.raises(ValueError, match="положительным"):
            operator.stepper(np.ones((NX, NY)), 0.0, side_dofs(space))

    def test_theta_outside_the_unit_interval_is_rejected(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        with pytest.raises(ValueError, match="θ"):
            operator.stepper(np.ones((NX, NY)), 0.01, side_dofs(space), theta=1.5)

    def test_wrong_field_shape_is_rejected(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        stepper = operator.stepper(np.ones((NX, NY)), 0.01, side_dofs(space))

        with pytest.raises(ValueError, match="формы"):
            stepper.advance(np.zeros((NX + 1, NY)), 0.0)


class TestSteadyState:
    def test_uniform_boundary_drives_the_field_to_that_value(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        stepper = operator.stepper(np.ones((NX, NY)), 0.05, side_dofs(space))
        field = np.zeros((NX, NY))

        for _ in range(200):
            field = stepper.advance(field, 7.0)

        assert np.allclose(field, 7.0, rtol=1e-6)

    def test_field_already_at_the_boundary_value_does_not_move(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        # Стационар обязан быть неподвижной точкой схемы. Дрейф означал бы
        # рассогласование масс-матрицы и кондуктивной части.
        stepper = operator.stepper(np.ones((NX, NY)), 0.01, side_dofs(space))
        field = np.full((NX, NY), 3.0)

        assert np.allclose(stepper.advance(field, 3.0), 3.0, rtol=1e-10)

    def test_crank_nicolson_leaves_residue_that_backward_euler_does_not(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        # Причина, по которой умолчанием стоит неявный Эйлер, а не более точная
        # схема второго порядка. Множитель усиления Кранка–Николсон при λΔt ≫ 1
        # стремится к −1, поэтому жёсткие сеточные моды не подавляются, а ползут
        # знакопеременно. Физическое время установления здесь давно прошло.
        #
        # Тест закрепляет находку: если кто-то вернёт θ=0.5 умолчанием, станет ясно,
        # что именно он покупает. Ошибка мала, но немонотонность в связке, где
        # напряжения входят в закон повреждаемости в степени r, недопустима.
        constrained = side_dofs(space)
        coarse_step = 0.05

        def settle(theta: float) -> float:
            stepper = operator.stepper(
                np.ones((NX, NY)), coarse_step, constrained, theta=theta
            )
            field = np.zeros((NX, NY))
            for _ in range(200):
                field = stepper.advance(field, 7.0)
            return float(np.max(np.abs(field - 7.0)))

        assert settle(BACKWARD_EULER) < 1e-9
        assert settle(CRANK_NICOLSON) > 1e-5


class TestAnalyticSlab:
    @pytest.mark.verification
    @pytest.mark.parametrize("theta", [CRANK_NICOLSON, BACKWARD_EULER])
    def test_matches_the_fourier_series(
        self, operator: ThermalOperator, space: FemSpace, theta: float
    ) -> None:
        end_time, step = 0.05, 5.0e-4
        stepper = operator.stepper(np.ones((NX, NY)), step, side_dofs(space), theta=theta)

        field = np.ones((NX, NY))
        for _ in range(round(end_time / step)):
            field = stepper.advance(field, 0.0)

        expected = slab_series(space.grid.x, end_time)
        measured = field[:, NY // 2]

        assert np.max(np.abs(measured - expected)) < 0.01, (
            f"максимальное расхождение {np.max(np.abs(measured - expected)):.4f}"
        )

    @pytest.mark.verification
    def test_centre_decay_follows_the_leading_mode(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        # На больших временах остаётся только первая гармоника, и центр обязан
        # затухать как exp(−π²t). Проверка показателя, а не одного значения.
        step = 1.0e-3
        stepper = operator.stepper(np.ones((NX, NY)), step, side_dofs(space))

        field = np.ones((NX, NY))
        centre = []
        for index in range(1, 201):
            field = stepper.advance(field, 0.0)
            if index % 50 == 0:
                centre.append(field[NX // 2, NY // 2])

        times = np.array([50, 100, 150, 200]) * step
        slope = np.polyfit(times, np.log(centre), 1)[0]

        assert slope == pytest.approx(-(np.pi**2), rel=0.02)


class TestMaximumPrinciple:
    """Диффузия не может вывести температуру за пределы граничных и начальных значений.

    Нарушение здесь — не погрешность, а неверный ответ: нефизичный выброс входит
    в закон повреждаемости в степени r ≈ 5 и портит предсказанный ресурс.
    """

    @pytest.mark.verification
    def test_lumped_capacity_keeps_the_field_within_bounds(
        self, space: FemSpace, material: MaterialParams
    ) -> None:
        operator = ThermalOperator.build(space, material, lumped=True)
        stepper = operator.stepper(np.ones((NX, NY)), 2.0e-4, side_dofs(space))

        field = np.zeros((NX, NY))
        for _ in range(40):
            field = stepper.advance(field, 1.0)
            assert field.min() >= -1e-12, f"провал ниже начального: {field.min()}"
            assert field.max() <= 1.0 + 1e-12

    def test_consistent_capacity_undershoots(
        self, space: FemSpace, material: MaterialParams
    ) -> None:
        # Причина, по которой диагонализация стоит умолчанием. Согласованная матрица
        # связывает соседние узлы, и на резком фронте появляется провал ниже минимума.
        # Тест закрепляет находку: возврат lumped=False должен быть осознанным.
        #
        # Шаг взят мелким намеренно: немонотонность проявляется при Δt < h²/(6α),
        # то есть ниже 1.6e-4 для этой сетки. Это контринтуитивно — обычно мелкий шаг
        # считают безопаснее, — и потому особенно легко пропустить.
        operator = ThermalOperator.build(space, material, lumped=False)
        stepper = operator.stepper(np.ones((NX, NY)), 5.0e-5, side_dofs(space))

        field = np.zeros((NX, NY))
        for _ in range(40):
            field = stepper.advance(field, 1.0)

        assert field.min() < -1e-6, f"провала не возникло: {field.min()}"


class TestConservation:
    def test_insulated_domain_keeps_its_heat(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        # Без закреплений на границе поток наружу нулевой, поэтому полное
        # теплосодержание ∫ρc·T обязано сохраняться, а поле — выравниваться.
        stepper = operator.stepper(np.ones((NX, NY)), 0.01, np.array([], dtype=np.int64))
        xx, _ = space.grid.coordinates()
        field = 1.0 + np.sin(np.pi * xx)

        def heat(values: np.ndarray) -> float:
            return float(np.sum(operator.capacity @ values.reshape(-1)))

        initial = heat(field)
        for _ in range(100):
            field = stepper.advance(field, 0.0)

        assert heat(field) == pytest.approx(initial, rel=1e-9)
        assert field.std() < 1e-3, "поле не выровнялось"


class TestDegradation:
    def test_damaged_material_conducts_more_slowly(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        # Двусторонняя связь: повреждение влияет на температурное поле, а не только
        # наоборот. Без неё связка вырождается в одностороннюю цепочку.
        constrained = side_dofs(space)
        intact = operator.stepper(np.ones((NX, NY)), 1.0e-3, constrained)
        damaged = operator.stepper(np.full((NX, NY), 0.2), 1.0e-3, constrained)

        hot, cool = np.ones((NX, NY)), np.ones((NX, NY))
        for _ in range(50):
            hot = damaged.advance(hot, 0.0)
            cool = intact.advance(cool, 0.0)

        assert hot[NX // 2, NY // 2] > cool[NX // 2, NY // 2]

    def test_zero_conductivity_freezes_the_interior(
        self, operator: ThermalOperator, space: FemSpace
    ) -> None:
        stepper = operator.stepper(np.full((NX, NY), 1e-12), 1.0e-3, side_dofs(space))
        field = np.ones((NX, NY))

        for _ in range(50):
            field = stepper.advance(field, 0.0)

        assert field[NX // 2, NY // 2] == pytest.approx(1.0, rel=1e-6)
