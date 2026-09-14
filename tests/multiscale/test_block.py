"""Блок нагружения: связка трёх процессов при замороженном повреждении.

Главный количественный тест — степенная зависимость движущей силы от нагрузки.
Φ пропорционально χ^r, поэтому удвоение растяжения обязано умножить Φ на 2^r.
Это проверяет разом всю цепочку: сборку, решение, восстановление напряжений,
эквивалентное напряжение, скрининг и интегрирование по времени. Ошибка в любом
звене сдвинет показатель.
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
    StructuredGrid,
)
from lattice.multiscale import BlockEvaluator
from lattice.solver import FemSpace

RESOLUTION = 33


def make_spec(*, far_field: float = 6.0e7, thermal_span: float = 0.01) -> CaseSpec:
    """Постановка с управляемым вкладом тепла.

    ``thermal_span`` близкий к нулю выключает температурные напряжения, оставляя
    чистую механику, — так проверяется степенная зависимость без примеси.
    """
    return CaseSpec(
        material=MaterialParams(),
        damage=DamageParams(exponent_r=5.0, reference_stress=1.0e8, nonlocal_length=4.0e-3),
        geometry=Geometry(width=0.1, height=0.1, thickness=1.0e-3, hole_radius=0.015),
        load=LoadProgram(
            # Период подобран по числу Фурье: при α ≈ 1.2·10⁻⁵ м²/с и полутолщине
            # 0.05 м выравнивание занимает ≈ 205 с. Более длинный блок прогревал бы
            # тело равномерно, а равномерное расширение напряжений не даёт.
            period=200.0,
            far_field_stress=far_field,
            temperature_cold=293.15,
            temperature_hot=293.15 + thermal_span,
            hold_fraction=0.6,
        ),
        mesh=MeshSpec(resolution=RESOLUTION, interface_fraction=0.3),
    )


@pytest.fixture
def space() -> FemSpace:
    return FemSpace.from_grid(
        StructuredGrid(nx=RESOLUTION, ny=RESOLUTION, width=0.1, height=0.1)
    )


def build(space: FemSpace, spec: CaseSpec) -> BlockEvaluator:
    return BlockEvaluator.build(space, spec, samples=3, thermal_steps_per_sample=2)


class TestUndamagedBlock:
    def test_driving_force_is_non_negative_everywhere(self, space: FemSpace) -> None:
        spec = make_spec()
        evaluator = build(space, spec)

        result = evaluator.evaluate(
            np.zeros((RESOLUTION, RESOLUTION)), evaluator.initial_temperature()
        )

        assert np.all(result.driving_integral >= 0.0)
        assert np.isfinite(result.driving_integral).all()

    def test_driving_force_peaks_beside_the_hole(self, space: FemSpace) -> None:
        # Повреждение обязано зарождаться у концентратора, а не у кромки нагружения.
        # Если максимум уходит на границу, перепутаны закрепления или знак нагрузки.
        spec = make_spec()
        evaluator = build(space, spec)

        result = evaluator.evaluate(
            np.zeros((RESOLUTION, RESOLUTION)), evaluator.initial_temperature()
        )
        peak = np.unravel_index(
            np.argmax(result.driving_integral), result.driving_integral.shape
        )

        centre = RESOLUTION // 2
        distance = np.hypot(peak[0] - centre, peak[1] - centre) * (0.1 / (RESOLUTION - 1))
        assert distance < 3.0 * spec.geometry.hole_radius, f"максимум в {peak}"

    def test_solves_converge(self, space: FemSpace) -> None:
        evaluator = build(space, make_spec())

        result = evaluator.evaluate(
            np.zeros((RESOLUTION, RESOLUTION)), evaluator.initial_temperature()
        )

        assert result.max_residual < 1e-7
        assert result.solves == 4
        assert result.iterations > 0


class TestPowerLaw:
    @pytest.mark.verification
    @pytest.mark.parametrize("exponent", [3.0, 5.0])
    def test_driving_force_scales_as_stress_to_the_r(
        self, space: FemSpace, exponent: float
    ) -> None:
        # Вся цепочка разом: сборка, решение, восстановление напряжений, Хейхёрст,
        # скрининг, интегрирование по времени. Все они линейны по нагрузке, поэтому
        # нелинейность обязана прийти ровно из степени r и ниоткуда больше.
        base = make_spec(far_field=4.0e7)
        spec = base.model_copy(
            update={"damage": base.damage.model_copy(update={"exponent_r": exponent})}
        )
        doubled = spec.model_copy(
            update={"load": spec.load.model_copy(update={"far_field_stress": 8.0e7})}
        )

        undamaged = np.zeros((RESOLUTION, RESOLUTION))
        single = build(space, spec).evaluate(
            undamaged, np.full((RESOLUTION, RESOLUTION), spec.load.temperature_cold)
        )
        twice = build(space, doubled).evaluate(
            undamaged, np.full((RESOLUTION, RESOLUTION), spec.load.temperature_cold)
        )

        ratio = float(twice.driving_integral.max() / single.driving_integral.max())
        assert ratio == pytest.approx(2.0**exponent, rel=0.02)


class TestDamageResponse:
    def test_damage_softens_the_structure(self, space: FemSpace) -> None:
        # Податливость — метрика отказа. Она обязана расти с повреждением,
        # иначе критерий «потеря 10% жёсткости» не сработает никогда.
        evaluator = build(space, make_spec())
        temperature = evaluator.initial_temperature()

        intact = evaluator.evaluate(np.zeros((RESOLUTION, RESOLUTION)), temperature)
        damaged = evaluator.evaluate(np.full((RESOLUTION, RESOLUTION), 0.3), temperature)

        assert damaged.compliance > intact.compliance

    def test_damage_raises_the_driving_force(self, space: FemSpace) -> None:
        # Положительная обратная связь: ослабленный материал перегружается сильнее,
        # и повреждение ускоряется. Ради этого эффекта задача и связана.
        evaluator = build(space, make_spec())
        temperature = evaluator.initial_temperature()

        localised = np.zeros((RESOLUTION, RESOLUTION))
        centre = RESOLUTION // 2
        localised[centre - 6 : centre + 7, centre - 1 : centre + 2] = 0.5

        intact = evaluator.evaluate(np.zeros((RESOLUTION, RESOLUTION)), temperature)
        weakened = evaluator.evaluate(localised, temperature)

        assert weakened.driving_integral.max() > intact.driving_integral.max()


class TestThermalCoupling:
    def test_thermal_cycle_adds_to_the_driving_force(self, space: FemSpace) -> None:
        # Прямая проверка того, что тепло действительно участвует. Если температурная
        # нагрузка не собирается или собирается с нулевым коэффициентом, оба варианта
        # совпадут — и связка окажется двухпроцессной вопреки заявленному.
        undamaged = np.zeros((RESOLUTION, RESOLUTION))

        cold = make_spec(thermal_span=0.01)
        hot = make_spec(thermal_span=400.0)

        flat = build(space, cold)
        cycled = build(space, hot)

        without = flat.evaluate(undamaged, flat.initial_temperature())
        with_heat = cycled.evaluate(undamaged, cycled.initial_temperature())

        assert with_heat.driving_integral.max() > 1.5 * without.driving_integral.max()

    def test_interior_lags_behind_the_surface(self, space: FemSpace) -> None:
        # Это и есть источник температурных напряжений. Поверхность к концу блока
        # возвращается на холодный уровень — так задано граничным условием, — а
        # внутренность не успевает остыть за период порядка времени выравнивания.
        # Возникает градиент, а вместе с ним стеснённое расширение.
        #
        # Если отставания нет, значит период выбран много больше L²/α, тело греется
        # равномерно, и термомеханическая связь вырождается: равномерно расширяющееся
        # тело напряжений не несёт.
        spec = make_spec(thermal_span=300.0)
        evaluator = BlockEvaluator.build(space, spec, samples=5, thermal_steps_per_sample=8)

        result = evaluator.evaluate(
            np.zeros((RESOLUTION, RESOLUTION)), evaluator.initial_temperature()
        )

        centre = RESOLUTION // 2
        surface = result.temperature[0, centre]
        # Точка в перемычке между отверстием и кромкой. Сам центр брать нельзя:
        # он лежит внутри отверстия, где проводимость почти нулевая, а теплоёмкость
        # полная, поэтому там температура стоит на начальном значении — и это верно.
        ligament = result.temperature[centre + 8, centre]

        assert surface == pytest.approx(spec.load.temperature_cold, abs=1e-6)
        assert ligament > surface + 0.2 * (
            spec.load.temperature_hot - spec.load.temperature_cold
        )
