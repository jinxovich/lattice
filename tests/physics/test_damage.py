"""Верификация закона Качанова–Работнова и эквивалентного напряжения Хейхёрста.

Центральный тест здесь — сверка замкнутой формы прыжка с численным интегрированием
исходного ОДУ. Он входит в набор верификации: если он зелёный, движок разделения
масштабов опирается не на приближение, а на точное решение при замороженном напряжении.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import solve_ivp

from pdsopromat.core import DamageParams
from pdsopromat.physics.damage import (
    blocks_to_damage,
    damage_driving_integral,
    damage_rate,
    hayhurst_equivalent_stress,
    kr_exact_jump,
    max_principal_plane_stress,
    von_mises_plane_stress,
)


def stress(xx: float = 0.0, yy: float = 0.0, xy: float = 0.0) -> np.ndarray:
    """Точечное напряжённое состояние в порядке [σ_xx, σ_yy, σ_xy]."""
    return np.array([xx, yy, xy], dtype=np.float64)


class TestVonMises:
    def test_uniaxial_tension_equals_applied_stress(self) -> None:
        assert von_mises_plane_stress(stress(xx=100.0)) == pytest.approx(100.0)

    def test_uniaxial_compression_equals_magnitude(self) -> None:
        # Мизес знака не различает — именно поэтому один он не годится
        # как движущая сила повреждаемости.
        assert von_mises_plane_stress(stress(xx=-100.0)) == pytest.approx(100.0)

    def test_pure_shear(self) -> None:
        assert von_mises_plane_stress(stress(xy=100.0)) == pytest.approx(100.0 * np.sqrt(3.0))

    def test_equibiaxial_tension(self) -> None:
        assert von_mises_plane_stress(stress(xx=100.0, yy=100.0)) == pytest.approx(100.0)

    def test_is_vectorised_over_a_field(self) -> None:
        field = np.zeros((4, 5, 3))
        field[..., 0] = 100.0

        assert von_mises_plane_stress(field).shape == (4, 5)
        assert np.allclose(von_mises_plane_stress(field), 100.0)


class TestMaxPrincipal:
    def test_uniaxial_tension(self) -> None:
        assert max_principal_plane_stress(stress(xx=100.0)) == pytest.approx(100.0)

    def test_uniaxial_compression_gives_zero(self) -> None:
        # Второе главное отрицательно, первое — ровно ноль.
        assert max_principal_plane_stress(stress(xx=-100.0)) == pytest.approx(0.0)

    def test_pure_shear_gives_shear_magnitude(self) -> None:
        assert max_principal_plane_stress(stress(xy=50.0)) == pytest.approx(50.0)

    def test_equibiaxial_compression_stays_in_plane(self) -> None:
        # Формула даёт первое ГЛАВНОЕ В ПЛОСКОСТИ, то есть −100, хотя истинное
        # первое главное трёхмерного состояния равно нулю (внеплоскостное).
        # Разницу закрывает скобка Маколея внутри Хейхёрста — см. соответствующий тест.
        assert max_principal_plane_stress(stress(xx=-100.0, yy=-100.0)) == pytest.approx(-100.0)


class TestHayhurst:
    def test_alpha_zero_reduces_to_von_mises(self) -> None:
        s = stress(xx=80.0, yy=-30.0, xy=20.0)

        assert hayhurst_equivalent_stress(s, alpha=0.0) == pytest.approx(von_mises_plane_stress(s))

    def test_alpha_one_reduces_to_bracketed_max_principal(self) -> None:
        s = stress(xx=80.0, yy=-30.0, xy=20.0)

        assert hayhurst_equivalent_stress(s, alpha=1.0) == pytest.approx(
            max_principal_plane_stress(s)
        )

    def test_alpha_one_gives_no_driving_force_in_uniaxial_compression(self) -> None:
        # Физический смысл скобки Маколея: сжатие не раскрывает поры.
        assert hayhurst_equivalent_stress(stress(xx=-100.0), alpha=1.0) == pytest.approx(0.0)

    def test_alpha_one_gives_no_driving_force_in_equibiaxial_compression(self) -> None:
        # Здесь главное в плоскости отрицательно — работает именно скобка,
        # а не случайное совпадение с нулём, как в одноосном случае.
        s = stress(xx=-100.0, yy=-100.0)

        assert hayhurst_equivalent_stress(s, alpha=1.0) == pytest.approx(0.0)

    def test_mixed_alpha_lies_between_the_two_measures(self) -> None:
        s = stress(xx=120.0, yy=40.0, xy=15.0)
        chi = hayhurst_equivalent_stress(s, alpha=0.5)

        low = min(von_mises_plane_stress(s), max_principal_plane_stress(s))
        high = max(von_mises_plane_stress(s), max_principal_plane_stress(s))
        assert low <= chi <= high


class TestExactJump:
    """Точность замкнутой формы — предпосылка всего движка разделения масштабов."""

    @pytest.mark.verification
    def test_matches_numerical_ode_integration(self) -> None:
        params = DamageParams(
            exponent_r=5.0, exponent_k=3.0, reference_stress=1.0, reference_time=1.0
        )
        chi = 0.5  # χ/σ_ref = 0.5
        period = 1.0
        n_blocks = 4.0

        phi = damage_driving_integral(
            np.array([chi]), weights=np.array([period]), params=params
        )

        def rate(_t: float, state: np.ndarray) -> np.ndarray:
            return np.atleast_1d(damage_rate(np.array(chi), state[0], params))

        numerical = solve_ivp(
            rate,
            t_span=(0.0, n_blocks * period),
            y0=np.array([0.0]),
            rtol=1e-12,
            atol=1e-14,
            dense_output=True,
        )

        closed_form = kr_exact_jump(0.0, phi, n_blocks, params)
        assert closed_form == pytest.approx(float(numerical.y[0, -1]), rel=1e-8)

    def test_is_exact_under_subdivision(self) -> None:
        # Один прыжок на M обязан совпасть с двумя по M/2 до машинной точности.
        # Иначе удвоение шага в PI-контроллере мерило бы ошибку интегратора,
        # а не то, ради чего оно делается — изменение Φ.
        params = DamageParams()
        phi, total = 1.0e-4, 200.0

        single = kr_exact_jump(0.1, phi, total, params)
        halved = kr_exact_jump(kr_exact_jump(0.1, phi, total / 2, params), phi, total / 2, params)

        assert single == pytest.approx(halved, rel=1e-14)

    def test_inverse_lands_exactly_on_the_target(self) -> None:
        params = DamageParams()
        phi = 2.0e-5
        target = 0.4

        n_blocks = blocks_to_damage(0.0, target, phi, params)

        assert kr_exact_jump(0.0, phi, n_blocks, params) == pytest.approx(target, rel=1e-12)

    def test_clamps_at_the_damage_cap_instead_of_producing_nan(self) -> None:
        # Скобка под корнем уходит в минус — это разрушение внутри прыжка.
        params = DamageParams(max_damage=0.99)

        result = kr_exact_jump(0.0, phi=1.0e-3, n_blocks=1.0e9, params=params)

        assert np.isfinite(result)
        assert result == pytest.approx(params.max_damage)

    def test_zero_driving_force_leaves_damage_untouched(self) -> None:
        params = DamageParams()

        assert kr_exact_jump(0.3, phi=0.0, n_blocks=1.0e6, params=params) == pytest.approx(0.3)

    def test_is_vectorised_over_a_damage_field(self) -> None:
        params = DamageParams()
        field = np.array([0.0, 0.1, 0.5])
        phi = np.array([1.0e-5, 2.0e-5, 1.0e-6])

        result = kr_exact_jump(field, phi, 100.0, params)

        assert result.shape == (3,)
        assert np.all(result >= field)


class TestErrorAmplification:
    """δN/N = r·δχ/χ — соотношение, задающее бюджет точности всему ML-контуру."""

    @pytest.mark.verification
    @pytest.mark.parametrize("r", [3.0, 5.0, 8.0])
    def test_life_error_is_r_times_the_stress_error(self, r: float) -> None:
        params = DamageParams(exponent_r=r, reference_stress=1.0, reference_time=1.0)
        weights = np.array([1.0])
        relative_stress_error = 1.0e-4  # мал, чтобы проверялась производная, а не кривизна

        perturbed_stress = np.array([0.5 * (1.0 + relative_stress_error)])
        baseline = blocks_to_damage(
            0.0, 0.5, damage_driving_integral(np.array([0.5]), weights, params), params
        )
        perturbed = blocks_to_damage(
            0.0, 0.5, damage_driving_integral(perturbed_stress, weights, params), params
        )

        amplification = abs(perturbed - baseline) / baseline / relative_stress_error
        assert amplification == pytest.approx(r, rel=1e-3)

    def test_two_percent_stress_error_costs_ten_percent_of_life_at_r5(self) -> None:
        # Числовая проверка утверждения из плана, на котором держится
        # требование ≲1.5% по напряжениям в хот-споте.
        params = DamageParams(exponent_r=5.0, reference_stress=1.0, reference_time=1.0)
        weights = np.array([1.0])

        def life(chi: float) -> float:
            return float(
                blocks_to_damage(
                    0.0, 0.5, damage_driving_integral(np.array([chi]), weights, params), params
                )
            )

        relative_life_error = abs(life(0.5 * 1.02) - life(0.5)) / life(0.5)

        assert relative_life_error == pytest.approx(0.10, abs=0.01)


class TestInvariants:
    @settings(max_examples=200, deadline=None)
    @given(
        d0=st.floats(min_value=0.0, max_value=0.95),
        phi=st.floats(min_value=0.0, max_value=1.0e-2),
        n_blocks=st.floats(min_value=0.0, max_value=1.0e6),
    )
    def test_damage_never_decreases_and_stays_bounded(
        self, d0: float, phi: float, n_blocks: float
    ) -> None:
        params = DamageParams()

        result = kr_exact_jump(d0, phi, n_blocks, params)

        assert np.isfinite(result)
        assert result >= d0 - 1e-12
        assert result <= params.max_damage + 1e-12

    @settings(max_examples=100, deadline=None)
    @given(
        xx=st.floats(min_value=-3.0e8, max_value=3.0e8),
        yy=st.floats(min_value=-3.0e8, max_value=3.0e8),
        xy=st.floats(min_value=-3.0e8, max_value=3.0e8),
        alpha=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_equivalent_stress_is_non_negative(
        self, xx: float, yy: float, xy: float, alpha: float
    ) -> None:
        chi = hayhurst_equivalent_stress(stress(xx, yy, xy), alpha=alpha)

        assert np.isfinite(chi)
        assert chi >= 0.0


class TestDrivingIntegral:
    def test_constant_stress_reproduces_the_closed_form(self) -> None:
        params = DamageParams(exponent_r=4.0, reference_stress=2.0, reference_time=10.0)
        chi, period = 1.0, 5.0

        phi = damage_driving_integral(np.array([chi]), weights=np.array([period]), params=params)

        assert phi == pytest.approx((chi / 2.0) ** 4.0 * period / 10.0)

    def test_weights_sum_to_the_block_period(self) -> None:
        # Контракт квадратуры: веса имеют размерность времени и в сумме дают t_c.
        params = DamageParams(reference_stress=1.0, reference_time=1.0, exponent_r=1.0)
        chi_samples = np.array([1.0, 1.0, 1.0])
        weights = np.array([0.25, 0.5, 0.25])

        assert damage_driving_integral(chi_samples, weights, params) == pytest.approx(1.0)

    def test_integrates_over_a_field(self) -> None:
        params = DamageParams()
        chi_samples = np.full((3, 8, 8), 1.0e8)
        weights = np.array([0.25, 0.5, 0.25])

        assert damage_driving_integral(chi_samples, weights, params).shape == (8, 8)

    def test_quadrature_mismatch_fails_loudly(self) -> None:
        # tensordot по несовпадающим осям иначе либо упал бы невнятно, либо —
        # при совпадении длин по случайности — молча проинтегрировал не то.
        params = DamageParams()

        with pytest.raises(ValueError, match="квадратуры"):
            damage_driving_integral(np.ones((3, 8)), np.array([0.5, 0.5]), params)
