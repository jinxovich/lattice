"""Нелокальный скрининг: сохранение, масштаб сглаживания, независимость от сетки.

Последнее — главное. Весь смысл регуляризации в том, чтобы ширина зоны локализации
задавалась физической длиной l_c, а не шагом сетки. Если это не выполняется,
предсказанный ресурс остаётся сеточным артефактом, и сравнивать с ним суррогат
бессмысленно.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdsopromat.core.grid import StructuredGrid
from pdsopromat.solver import FemSpace
from pdsopromat.solver.screening import ScreeningOperator


def build(n: int = 41, length: float = 0.05, size: float = 1.0) -> ScreeningOperator:
    return ScreeningOperator.build(
        FemSpace.from_grid(StructuredGrid(nx=n, ny=n, width=size, height=size)), length
    )


def spike(n: int, amplitude: float = 1.0) -> np.ndarray:
    """Единичный всплеск в центре — модель одноэлементной локализации."""
    field = np.zeros((n, n))
    field[n // 2, n // 2] = amplitude
    return field


def gaussian(n: int, width: float, size: float = 1.0) -> np.ndarray:
    """Источник фиксированной физической ширины, нормированный на единичный пик.

    В отличие от всплеска в одну ячейку, это функция, а не мера: при измельчении
    сетки она сходится к самой себе, и отклик на неё можно осмысленно сравнивать
    между сетками.
    """
    grid = StructuredGrid(nx=n, ny=n, width=size, height=size)
    xx, yy = grid.coordinates()
    radius_squared = (xx - 0.5 * size) ** 2 + (yy - 0.5 * size) ** 2
    return np.exp(-0.5 * radius_squared / width**2)


def second_moment(operator: ScreeningOperator, field: np.ndarray, size: float = 1.0) -> float:
    """⟨r²⟩ = ∫r²·f dx / ∫f dx относительно центра области.

    Мера ширины, у которой для этого фильтра есть точный аналитический ответ —
    в отличие от полуширины по полувысоте. Последняя для узкого источника
    определяется логарифмической частью K₀ и с l_c связана лишь слабо, поэтому
    как критерий не годится.
    """
    xx, yy = operator.grid.coordinates()
    radius_squared = (xx - 0.5 * size) ** 2 + (yy - 0.5 * size) ** 2
    return operator.integrate(radius_squared * field) / operator.integrate(field)


class TestConstruction:
    def test_non_positive_length_is_rejected(self) -> None:
        space = FemSpace.from_grid(StructuredGrid(nx=9, ny=9, width=1.0, height=1.0))

        with pytest.raises(ValueError, match="положительной"):
            ScreeningOperator.build(space, 0.0)

    def test_wrong_field_shape_is_rejected(self) -> None:
        operator = build(n=9)

        with pytest.raises(ValueError, match="формы"):
            operator.apply(np.zeros((8, 9)))


class TestInvariants:
    def test_constant_field_passes_through_untouched(self) -> None:
        # Лапласиан постоянного поля равен нулю, поэтому оператор обязан
        # воспроизвести константу точно. Отклонение означает ошибку сборки.
        operator = build()
        constant = np.full((41, 41), 7.0)

        assert np.allclose(operator.apply(constant), 7.0, rtol=1e-10)

    def test_total_quantity_is_conserved(self) -> None:
        # Условие Неймана обнуляет граничный вклад лапласиана: скрининг
        # перераспределяет величину, но не создаёт и не уничтожает её.
        operator = build()
        field = spike(41, amplitude=100.0)

        assert operator.integrate(operator.apply(field)) == pytest.approx(
            operator.integrate(field), rel=1e-9
        )

    def test_symmetry_is_preserved(self) -> None:
        operator = build()

        smoothed = operator.apply(spike(41))

        assert np.allclose(smoothed, smoothed[::-1, :], atol=1e-12)
        assert np.allclose(smoothed, smoothed[:, ::-1], atol=1e-12)
        assert np.allclose(smoothed, smoothed.T, atol=1e-12)

    def test_is_linear(self) -> None:
        operator = build()
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(41, 41)), rng.normal(size=(41, 41))

        combined = operator.apply(3.0 * a - 2.0 * b)

        assert np.allclose(combined, 3.0 * operator.apply(a) - 2.0 * operator.apply(b))


class TestSmoothing:
    def test_peak_is_flattened(self) -> None:
        operator = build()
        field = spike(41)

        smoothed = operator.apply(field)

        assert smoothed.max() < field.max()
        assert np.count_nonzero(smoothed > 1e-6 * smoothed.max()) > 1

    def test_longer_length_smooths_more(self) -> None:
        field = spike(41)

        mild = build(length=0.02).apply(field)
        strong = build(length=0.10).apply(field)

        assert strong.max() < mild.max()

    def test_short_length_approaches_the_identity(self) -> None:
        # При l_c → 0 оператор обязан вырождаться в тождественный: регуляризация
        # отключается, и остаётся исходное локальное поле.
        operator = build(n=41, length=1.0e-6)
        rng = np.random.default_rng(1)
        field = rng.normal(size=(41, 41))

        assert np.allclose(operator.apply(field), field, atol=1e-6)


class TestMeshIndependence:
    """Ради этого регуляризация и вводится.

    Источником служит гауссиан фиксированной физической ширины, а не всплеск
    в одну ячейку. Причина не в удобстве: функция Грина двумерного оператора
    Гельмгольца есть K₀(r/l_c), у неё логарифмическая особенность в нуле, поэтому
    у отклика на дельта-источник нет сеточно-независимого пика в принципе. Это
    свойство самой функции Грина, а не дефект регуляризации, но мерить на таком
    источнике сходимость нельзя.
    """

    @pytest.mark.verification
    def test_added_variance_equals_four_l_squared(self) -> None:
        # Точное свойство оператора. В Фурье фильтр есть 1/(1+l²k²), откуда
        # второй момент его функции Грина равен ровно 4l². Значит фильтрация
        # добавляет к дисперсии поля 4l² и ничего больше.
        #
        # Это несопоставимо сильнее проверок «стало глаже»: величина предсказана
        # аналитически с точностью до множителя, и ошибка в сборке масс-матрицы
        # или лапласиана сдвинет её сразу.
        size, source_width, length = 1.0, 0.05, 0.03
        operator = build(n=81, length=length, size=size)
        source = gaussian(81, source_width, size)

        added = second_moment(operator, operator.apply(source), size) - second_moment(
            operator, source, size
        )

        assert added == pytest.approx(4.0 * length**2, rel=0.02)

    @pytest.mark.verification
    def test_added_variance_is_mesh_independent(self) -> None:
        # Ради этого регуляризация и вводится: масштаб зоны задаёт l_c, а не шаг.
        size, source_width, length = 1.0, 0.05, 0.03

        added = []
        for n in (41, 81, 161):
            operator = build(n=n, length=length, size=size)
            source = gaussian(n, source_width, size)
            added.append(
                second_moment(operator, operator.apply(source), size)
                - second_moment(operator, source, size)
            )

        spread = (max(added) - min(added)) / float(np.mean(added))
        assert spread < 0.02, f"добавленная дисперсия следует за сеткой: {added}"

    @pytest.mark.verification
    def test_doubling_the_length_quadruples_the_added_variance(self) -> None:
        # Контроль, что предыдущие тесты не проходят тривиально: зависимость
        # от l_c квадратичная, а не какая угодно.
        size, source_width, n = 1.0, 0.05, 81
        source = gaussian(n, source_width, size)

        def added_variance(length: float) -> float:
            operator = build(n=n, length=length, size=size)
            return second_moment(operator, operator.apply(source), size) - second_moment(
                operator, source, size
            )

        assert added_variance(0.06) / added_variance(0.03) == pytest.approx(4.0, rel=0.03)

    @pytest.mark.verification
    def test_response_converges_under_refinement(self) -> None:
        # Источник разрешён на всех трёх сетках (σ/h от 2 до 8), поэтому
        # последовательность откликов осмысленна и обязана сходиться.
        length, source_width, size = 0.06, 0.05, 1.0

        peaks = [
            build(n=n, length=length, size=size).apply(gaussian(n, source_width, size)).max()
            for n in (41, 81, 161)
        ]

        first_step = abs(peaks[1] - peaks[0])
        second_step = abs(peaks[2] - peaks[1])
        assert second_step < 0.5 * first_step, f"отклик не сходится: {peaks}"
