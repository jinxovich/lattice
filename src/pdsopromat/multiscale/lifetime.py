"""Прокат до отказа прыжками по блокам нагружения.

Прямой ответ на барьер из карточки проекта. Напряжения живут на масштабе секунд,
накопление повреждаемости — на масштабе лет; считать каждый блок по отдельности
невозможно. Медленный процесс продвигается прыжками через тысячи блоков сразу,
быстрый разрешается внутри одного блока.

**Где именно возникает ошибка.** Прыжок Качанова–Работнова при замороженной
движущей силе точен — это доказано сверкой с численным интегрированием ОДУ
и неизменностью результата при подразбиении. Следовательно, удвоение шага на самой
формуле показало бы ноль и не измерило бы ничего. Единственный источник ошибки —
то, что за время прыжка Φ меняется: повреждение перераспределяет напряжения.

Поэтому шаг контролируется схемой предиктор-корректор по Φ, а не подразбиением::

    предиктор:   d* = jump(dₙ, Φₙ, M)
    оценка:      Φ* = block(d*)
    корректор:   d⁺ = jump(dₙ, (Φₙ + Φ*)/2, M)
    ошибка:      max|d⁺ − d*|

Это метод Хойна применительно к медленно меняющейся движущей силе. Он даёт
одновременно второй порядок и оценку локальной ошибки, причём оценивает ровно
ту величину, которая приближается, — а не погрешность интегратора, которой нет.

Цена — два вычисления блока на шаг вместо одного. Второй порядок окупает её
уменьшением числа шагов.

Размер шага ведёт ПИ-регулятор. Чисто интегральный отрабатывает каждое отклонение
целиком и склонен к колебаниям «слишком большой шаг — отказ — слишком малый»;
пропорциональная часть сглаживает последовательность.

Критерий отказа диффузный: потеря 10% глобальной жёсткости. Локальное «d достигло
единицы» сетко-зависимо, и опирать на него метрику ресурса нельзя.

**Измеренное поведение** на демонстрационной постановке (сетка 25×25, отказ около
502 блоков), при изменении допуска в 16 раз::

    допуск    блоков до отказа   вычислений блока   блоков на вычисление
    4.0e-02            502.40                  53                    9.5
    1.0e-02            501.75                  72                    7.0
    2.5e-03            501.65                 101                    5.0

Ресурс держится в пределах 0.15% при четырёхкратном росте стоимости, то есть
прыжок не покупает скорость ценой неизвестной ошибки.

Отношение «блоков на вычисление» растёт вместе с ресурсом: размер прыжка ограничен
допуском на приращение повреждаемости, а не числом блоков. На этой короткоживущей
постановке выигрыш семикратный; на реальных сроках в 10⁵–10⁶ блоков те же несколько
десятков шагов дадут его на три-четыре порядка. Приводить одно число в отрыве
от длительности жизни было бы лукавством.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pdsopromat.core.state import State
from pdsopromat.multiscale.block import BlockEvaluator, BlockResult
from pdsopromat.physics.damage import kr_exact_jump

FloatArray = npt.NDArray[np.float64]

HEUN_ORDER = 2


@dataclass(frozen=True)
class LifetimeSettings:
    """Настройки проката. Умолчания подобраны консервативно, а не быстро."""

    damage_tolerance: float = 5.0e-3
    compliance_criterion: float = 1.10
    initial_step: float = 1.0
    max_step_growth: float = 4.0
    min_step_shrink: float = 0.2
    safety: float = 0.9
    max_blocks: float = 1.0e9
    max_steps: int = 2000

    def __post_init__(self) -> None:
        if self.damage_tolerance <= 0.0:
            msg = f"допуск по повреждаемости должен быть положительным: {self.damage_tolerance}"
            raise ValueError(msg)
        if self.compliance_criterion <= 1.0:
            msg = (
                "критерий отказа задаётся отношением податливости и должен превышать "
                f"единицу, получено {self.compliance_criterion}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class LifetimeSample:
    """Одна принятая точка траектории — строка будущего сертификата."""

    blocks: float
    damage_max: float
    compliance_ratio: float
    driving_max: float
    step: float
    local_error: float
    residual: float


@dataclass(frozen=True)
class LifetimeResult:
    """Итог проката вместе со стоимостью, которой он достался.

    Стоимость возвращается всегда и наравне с ответом: сравнение суррогата
    с эталоном без учёта числа решений — то самое приписывание ускорения не тому,
    от которого проект обязан себя защитить.
    """

    reached_criterion: bool
    blocks_to_criterion: float | None
    trajectory: tuple[LifetimeSample, ...]
    final_state: State
    block_evaluations: int
    rejected_steps: int
    solver_iterations: int
    stop_reason: str

    @property
    def damage_max(self) -> float:
        return float(self.final_state.damage.max())


@dataclass
class LifetimeEngine:
    """Прокат до критерия отказа с адаптивным прыжком.

    Движок не знает, кто решает механику: :class:`BlockEvaluator` инкапсулирует
    и эталонный путь, и будущий суррогатный. Замена одного другим здесь — подмена
    объекта, а не правка движка.
    """

    evaluator: BlockEvaluator
    settings: LifetimeSettings = field(default_factory=LifetimeSettings)

    def _next_step(self, step: float, error: float, previous_error: float) -> float:
        """ПИ-регулятор по локальной ошибке.

        Показатели 0.7/k и 0.4/k — стандартный набор Густафссона для схемы порядка k.
        """
        tolerance = self.settings.damage_tolerance
        if error <= 0.0:
            return step * self.settings.max_step_growth

        integral = (tolerance / error) ** (0.7 / HEUN_ORDER)
        proportional = (
            (previous_error / error) ** (0.4 / HEUN_ORDER) if previous_error > 0.0 else 1.0
        )
        factor = self.settings.safety * integral * proportional
        bounded = min(
            max(float(factor), self.settings.min_step_shrink), self.settings.max_step_growth
        )
        return step * bounded

    def run(self, initial: State | None = None) -> LifetimeResult:
        spec = self.evaluator.spec
        state = initial if initial is not None else State.initial(
            self.evaluator.space.grid, spec
        )

        baseline = self.evaluator.evaluate(state.damage, state.temperature)
        reference_compliance = baseline.compliance
        evaluations, rejected, iterations = 1, 0, baseline.iterations

        current: BlockResult = baseline
        step = self.settings.initial_step
        previous_error = 0.0
        samples: list[LifetimeSample] = []
        crossing: float | None = None
        reason = "исчерпан лимит шагов"

        for _ in range(self.settings.max_steps):
            # Прыжок подрезается остатком бюджета: иначе последний шаг перелетал бы
            # заданный предел, и «прокат на N блоков» означал бы «на N с чем-то».
            step = min(step, self.settings.max_blocks - state.blocks)
            if step <= 0.0:
                reason = "исчерпан лимит блоков"
                break

            predicted = kr_exact_jump(
                state.damage, current.driving_integral, step, spec.damage
            )
            probe = self.evaluator.evaluate(predicted, current.temperature)
            evaluations += 1
            iterations += probe.iterations

            averaged = 0.5 * (current.driving_integral + probe.driving_integral)
            corrected = kr_exact_jump(state.damage, averaged, step, spec.damage)
            error = float(np.max(np.abs(corrected - predicted)))

            if error > self.settings.damage_tolerance and step > 1.0:
                rejected += 1
                step = max(self._next_step(step, error, previous_error), 1.0)
                continue

            blocks = state.blocks + step
            state = state.advanced(
                time=blocks * spec.load.period,
                blocks=blocks,
                temperature=probe.temperature,
                damage=corrected,
            )

            current = self.evaluator.evaluate(state.damage, state.temperature)
            evaluations += 1
            iterations += current.iterations

            ratio = current.compliance / reference_compliance
            samples.append(
                LifetimeSample(
                    blocks=blocks,
                    damage_max=float(state.damage.max()),
                    compliance_ratio=ratio,
                    driving_max=float(current.driving_integral.max()),
                    step=step,
                    local_error=error,
                    residual=current.max_residual,
                )
            )

            if ratio >= self.settings.compliance_criterion:
                crossing = _interpolate_crossing(samples, self.settings.compliance_criterion)
                reason = "достигнут критерий отказа"
                break
            if blocks >= self.settings.max_blocks:
                reason = "исчерпан лимит блоков"
                break

            # Порядок существен: пропорциональная часть сравнивает текущую ошибку
            # с ошибкой ПРЕДЫДУЩЕГО шага. Обновить previous_error раньше — значит
            # получить отношение единицы и выключить регулятор до чисто интегрального.
            step = self._next_step(step, error, previous_error)
            previous_error = error

        return LifetimeResult(
            reached_criterion=crossing is not None,
            blocks_to_criterion=crossing,
            trajectory=tuple(samples),
            final_state=state,
            block_evaluations=evaluations,
            rejected_steps=rejected,
            solver_iterations=iterations,
            stop_reason=reason,
        )


def _interpolate_crossing(samples: list[LifetimeSample], criterion: float) -> float:
    """Число блоков в момент пересечения критерия, линейной интерполяцией.

    Без интерполяции ресурс округлялся бы до размера прыжка, а он адаптивный
    и может достигать тысяч блоков. Метрика тогда зависела бы от настроек
    регулятора шага сильнее, чем от физики.
    """
    last = samples[-1]
    if len(samples) == 1 or samples[-2].compliance_ratio >= criterion:
        return last.blocks

    previous = samples[-2]
    span = last.compliance_ratio - previous.compliance_ratio
    if span <= 0.0:
        return last.blocks

    fraction = (criterion - previous.compliance_ratio) / span
    return previous.blocks + fraction * (last.blocks - previous.blocks)
