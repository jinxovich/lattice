"""Прокат до отказа: сходимость по допуску, адаптация шага, учёт стоимости.

Главный тест — сходимость предсказанного ресурса при ужесточении допуска. Он
отвечает на единственный вопрос, который имеет значение: не «работает ли прыжок»,
а «к чему он сходится и насколько мы близки». Без этого ускорение оказывается
разменом на неизвестную ошибку.
"""

from __future__ import annotations

import numpy as np
import pytest

from lattice.core import (
    CaseSpec,
    DamageParams,
    Geometry,
    LoadProgram,
    MaterialParams,
    MeshSpec,
    State,
    StructuredGrid,
)
from lattice.multiscale import BlockEvaluator
from lattice.multiscale.lifetime import (
    LifetimeEngine,
    LifetimeSample,
    LifetimeSettings,
    _interpolate_crossing,
)
from lattice.solver import FemSpace

RESOLUTION = 25


def make_spec() -> CaseSpec:
    """Постановка, доводимая до отказа за десятки блоков.

    Опорное напряжение занижено намеренно: при паспортных значениях ресурс
    измерялся бы миллионами блоков, и тест шёл бы часами, ничего дополнительно
    не проверяя. Физика от этого не меняется — меняется масштаб времени.
    """
    return CaseSpec(
        material=MaterialParams(),
        damage=DamageParams(
            exponent_r=4.0,
            exponent_k=2.0,
            reference_stress=2.2e8,
            # Постоянная ползучести порядка половины суток. При t_ref = 1 с движущая сила
            # за блок выходит порядка сотен, и разрушение наступает быстрее одного
            # блока — прыгать становится не через что, и движок нечем проверять.
            reference_time=5.0e4,
            nonlocal_length=5.0e-3,
        ),
        geometry=Geometry(width=0.1, height=0.1, thickness=1.0e-3, hole_radius=0.015),
        load=LoadProgram(
            period=200.0,
            far_field_stress=7.0e7,
            temperature_cold=293.15,
            temperature_hot=523.15,
            hold_fraction=0.6,
        ),
        mesh=MeshSpec(resolution=RESOLUTION, interface_fraction=0.3),
    )


@pytest.fixture
def engine() -> LifetimeEngine:
    spec = make_spec()
    space = FemSpace.from_grid(
        StructuredGrid(nx=RESOLUTION, ny=RESOLUTION, width=0.1, height=0.1)
    )
    evaluator = BlockEvaluator.build(space, spec, samples=3, thermal_steps_per_sample=2)
    return LifetimeEngine(evaluator)


class TestSettingsValidation:
    def test_non_positive_tolerance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="допуск"):
            LifetimeSettings(damage_tolerance=0.0)

    def test_criterion_below_unity_is_rejected(self) -> None:
        # Критерий — отношение податливостей, поэтому единица означает «уже отказ».
        with pytest.raises(ValueError, match="единицу"):
            LifetimeSettings(compliance_criterion=0.9)


class TestStepController:
    """Краевые случаи регулятора и интерполяции — без прокатов, на чистых функциях."""

    def test_zero_error_grows_the_step_to_the_limit(self, engine: LifetimeEngine) -> None:
        # Нулевая ошибка означает, что Φ за шаг не изменилась вовсе. Делить на неё
        # нельзя, а наращивать шаг — можно и нужно, до разрешённого потолка.
        grown = engine._next_step(step=10.0, error=0.0, previous_error=1.0e-3)

        assert grown == pytest.approx(10.0 * engine.settings.max_step_growth)

    def test_single_sample_crossing_returns_its_own_position(self) -> None:
        # Интерполировать не по чему: критерий перейдён на первом же шаге.
        sample = LifetimeSample(
            blocks=7.0,
            damage_max=0.5,
            compliance_ratio=1.2,
            driving_max=1.0,
            step=7.0,
            local_error=0.0,
            residual=0.0,
        )

        assert _interpolate_crossing([sample], 1.1) == pytest.approx(7.0)

    def test_flat_compliance_falls_back_to_the_last_point(self) -> None:
        # Вырожденный случай: податливость не выросла между точками, наклона нет.
        # Делить на ноль нельзя, поэтому берётся последняя точка.
        def sample(blocks: float, ratio: float) -> LifetimeSample:
            return LifetimeSample(
                blocks=blocks,
                damage_max=0.5,
                compliance_ratio=ratio,
                driving_max=1.0,
                step=1.0,
                local_error=0.0,
                residual=0.0,
            )

        assert _interpolate_crossing([sample(3.0, 1.05), sample(9.0, 1.05)], 1.1) == (
            pytest.approx(9.0)
        )

    def test_exhausted_budget_stops_before_the_first_jump(
        self, engine: LifetimeEngine
    ) -> None:
        # Состояние приходит из интерфейса двойника и может уже исчерпать заданный
        # бюджет. Прыгать назад нельзя, поэтому движок обязан остановиться сразу.
        spec = make_spec()
        fresh = State.initial(engine.evaluator.space.grid, spec)
        spent = fresh.advanced(time=5.0 * spec.load.period, blocks=5.0)
        engine.settings = LifetimeSettings(max_blocks=2.0)

        result = engine.run(spent)

        assert not result.reached_criterion
        assert result.trajectory == ()
        assert "блоков" in result.stop_reason

    def test_step_budget_stops_the_rollout(self, engine: LifetimeEngine) -> None:
        # Отдельный предохранитель от бесконечного цикла: если регулятор почему-то
        # перестал наращивать шаг, прокат обязан закончиться, а не висеть.
        engine.settings = LifetimeSettings(max_steps=1)

        result = engine.run()

        assert not result.reached_criterion
        assert len(result.trajectory) == 1
        assert "шагов" in result.stop_reason


class TestRollout:
    @pytest.mark.slow
    def test_reaches_the_failure_criterion(self, engine: LifetimeEngine) -> None:
        result = engine.run()

        assert result.reached_criterion, result.stop_reason
        assert result.blocks_to_criterion is not None
        assert result.blocks_to_criterion > 0.0
        assert result.damage_max > 0.0

    @pytest.mark.slow
    def test_trajectory_is_monotone(self, engine: LifetimeEngine) -> None:
        # Повреждаемость термодинамически необратима, податливость только растёт.
        # Немонотонность означала бы ошибку знака или перепутанные аргументы прыжка.
        result = engine.run()
        damage = [sample.damage_max for sample in result.trajectory]
        compliance = [sample.compliance_ratio for sample in result.trajectory]

        assert np.all(np.diff(damage) >= -1e-12)
        assert np.all(np.diff(compliance) >= -1e-9)

    @pytest.mark.slow
    def test_step_grows_while_damage_is_small(self, engine: LifetimeEngine) -> None:
        # Ради этого движок и нужен: пока перераспределение напряжений медленное,
        # регулятор обязан наращивать прыжок, а не идти по одному блоку.
        result = engine.run()
        steps = [sample.step for sample in result.trajectory]

        assert max(steps) > 5.0 * steps[0], f"шаг не вырос: {steps[:10]}"

    @pytest.mark.slow
    def test_reports_its_own_cost(self, engine: LifetimeEngine) -> None:
        # Стоимость возвращается наравне с ответом. Сравнение суррогата с эталоном
        # без числа решений — то самое приписывание ускорения не тому, от которого
        # проект обязан себя защитить.
        result = engine.run()

        assert result.block_evaluations > 0
        assert result.solver_iterations > 0
        assert result.block_evaluations >= 2 * len(result.trajectory)

    @pytest.mark.slow
    def test_respects_the_block_budget(self, engine: LifetimeEngine) -> None:
        engine.settings = LifetimeSettings(max_blocks=3.0)

        result = engine.run()

        assert not result.reached_criterion
        assert result.final_state.blocks == pytest.approx(3.0)
        assert "блоков" in result.stop_reason


class TestToleranceConvergence:
    @pytest.mark.slow
    @pytest.mark.verification
    def test_predicted_life_is_insensitive_to_the_tolerance(
        self, engine: LifetimeEngine
    ) -> None:
        # Единственный вопрос, который имеет значение: во что обходится прыжок
        # по точности. Проверяется не порядок сходимости, а разброс ответа при
        # изменении допуска в 16 раз — на измеренном уровне 0.15% разности между
        # соседними допусками уже шум, и утверждение о монотонном убывании было бы
        # хрупким, а не строгим.
        lives = []
        for tolerance in (4.0e-2, 1.0e-2, 2.5e-3):
            engine.settings = LifetimeSettings(damage_tolerance=tolerance)
            result = engine.run()
            assert result.reached_criterion, f"допуск {tolerance}: {result.stop_reason}"
            assert result.blocks_to_criterion is not None
            lives.append(result.blocks_to_criterion)

        spread = (max(lives) - min(lives)) / float(np.mean(lives))
        assert spread < 0.01, f"ресурс зависит от допуска: {lives}"

    @pytest.mark.slow
    def test_tighter_tolerance_costs_more_evaluations(
        self, engine: LifetimeEngine
    ) -> None:
        # Обратная сторона сходимости: точность не бесплатна. Эта пара чисел —
        # первая точка будущей кривой «точность против стоимости».
        engine.settings = LifetimeSettings(damage_tolerance=2.0e-2)
        loose = engine.run()
        engine.settings = LifetimeSettings(damage_tolerance=2.5e-3)
        tight = engine.run()

        assert tight.block_evaluations > loose.block_evaluations


class TestInitialState:
    @pytest.mark.slow
    def test_starting_damaged_shortens_the_remaining_life(
        self, engine: LifetimeEngine
    ) -> None:
        # Ровно этот запрос будет приходить из интерфейса двойника: остаточный
        # ресурс при известном текущем состоянии, а не от нуля.
        spec = make_spec()
        grid = engine.evaluator.space.grid
        fresh = State.initial(grid, spec)
        worn = fresh.advanced(
            time=0.0, blocks=0.0, damage=np.full((RESOLUTION, RESOLUTION), 0.25)
        )

        engine.settings = LifetimeSettings(damage_tolerance=2.0e-2)
        intact = engine.run()
        used = engine.run(worn)

        assert used.reached_criterion
        assert intact.blocks_to_criterion is not None
        assert used.blocks_to_criterion is not None
        assert used.blocks_to_criterion < intact.blocks_to_criterion
