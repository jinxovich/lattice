"""Определяющие соотношения плоского напряжённого состояния с деградацией и теплом.

Проверяются не формулы, а физические частные случаи, у которых есть независимый
аналитический ответ: одноосное растяжение, свободное и стеснённое тепловое
расширение, чистый сдвиг. Сверка «код против своей же выкладки» ничего не доказала бы.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdsopromat.core import DamageParams, MaterialParams
from pdsopromat.physics.elasticity import (
    degradation_multiplier,
    plane_stress_moduli,
    stress_from_strain,
    thermal_strain,
)


def strain(xx: float = 0.0, yy: float = 0.0, xy: float = 0.0) -> np.ndarray:
    """Тензорные компоненты деформации [ε_xx, ε_yy, ε_xy] — без инженерного сдвига."""
    return np.array([xx, yy, xy], dtype=np.float64)


@pytest.fixture
def material() -> MaterialParams:
    return MaterialParams(youngs_modulus=2.0e11, poisson_ratio=0.3, thermal_expansion=1.2e-5)


class TestModuli:
    def test_shear_modulus_matches_the_standard_relation(self, material: MaterialParams) -> None:
        _, mu = plane_stress_moduli(material)

        expected = material.youngs_modulus / (2.0 * (1.0 + material.poisson_ratio))
        assert mu == pytest.approx(expected)

    def test_lame_lambda_is_the_plane_stress_variant(self, material: MaterialParams) -> None:
        # Плоское напряжённое состояние даёт λ = Eν/(1−ν²), а не трёхмерное
        # Eν/((1+ν)(1−2ν)). Подстановка трёхмерного значения — классическая ошибка,
        # которая проходит все тесты на сдвиг и ломает только объёмный отклик.
        lam, _ = plane_stress_moduli(material)

        E, nu = material.youngs_modulus, material.poisson_ratio
        assert lam == pytest.approx(E * nu / (1.0 - nu**2))


class TestUniaxial:
    def test_uniaxial_strain_state_gives_uniaxial_stress(self, material: MaterialParams) -> None:
        # Растяжение с свободным поперечным сужением: ε = [e, −νe, 0] → σ = [Ee, 0, 0].
        e = 1.0e-4
        eps = strain(xx=e, yy=-material.poisson_ratio * e)

        sigma = stress_from_strain(eps, material)

        assert sigma[0] == pytest.approx(material.youngs_modulus * e)
        assert sigma[1] == pytest.approx(0.0, abs=1.0)

    def test_equibiaxial_strain(self, material: MaterialParams) -> None:
        e = 1.0e-4
        E, nu = material.youngs_modulus, material.poisson_ratio

        sigma = stress_from_strain(strain(xx=e, yy=e), material)

        assert sigma[0] == pytest.approx(E * e / (1.0 - nu))
        assert sigma[1] == pytest.approx(E * e / (1.0 - nu))

    def test_pure_shear_uses_the_shear_modulus(self, material: MaterialParams) -> None:
        _, mu = plane_stress_moduli(material)
        gamma = 1.0e-4

        sigma = stress_from_strain(strain(xy=0.5 * gamma), material)

        # Тензорная компонента ε_xy = γ/2, поэтому σ_xy = 2μ·ε_xy = μγ.
        assert sigma[2] == pytest.approx(mu * gamma)
        assert sigma[0] == pytest.approx(0.0, abs=1.0)


class TestThermal:
    def test_free_expansion_produces_no_stress(self, material: MaterialParams) -> None:
        # Тело, которому дали расшириться, напряжений не несёт. Если этот тест красный,
        # знак теплового члена перепутан, и вся термомеханика считает наоборот.
        delta_t = 200.0
        eps_th = thermal_strain(material, delta_t)

        sigma = stress_from_strain(eps_th, material, delta_temperature=delta_t)

        assert np.allclose(sigma, 0.0, atol=1.0)

    def test_fully_constrained_heating_gives_compression(self, material: MaterialParams) -> None:
        delta_t = 200.0
        E, nu, alpha = (
            material.youngs_modulus,
            material.poisson_ratio,
            material.thermal_expansion,
        )

        sigma = stress_from_strain(strain(), material, delta_temperature=delta_t)

        # Аналитика для стеснённого нагрева: σ = −EαΔT/(1−ν) по обеим осям.
        expected = -E * alpha * delta_t / (1.0 - nu)
        assert sigma[0] == pytest.approx(expected)
        assert sigma[1] == pytest.approx(expected)
        assert sigma[2] == pytest.approx(0.0, abs=1.0)

    def test_thermal_strain_is_isotropic_and_shear_free(self, material: MaterialParams) -> None:
        eps_th = thermal_strain(material, 100.0)

        assert eps_th[0] == pytest.approx(eps_th[1])
        assert eps_th[2] == pytest.approx(0.0)

    def test_cooling_below_reference_gives_tension(self, material: MaterialParams) -> None:
        sigma = stress_from_strain(strain(), material, delta_temperature=-150.0)

        assert sigma[0] > 0.0


class TestDegradation:
    def test_multiplier_scales_stress_linearly(self, material: MaterialParams) -> None:
        eps = strain(xx=1.0e-4, yy=-0.5e-4, xy=0.3e-4)

        full = stress_from_strain(eps, material)
        halved = stress_from_strain(eps, material, multiplier=0.5)

        assert np.allclose(halved, 0.5 * full)

    def test_multiplier_also_scales_the_thermal_term(self, material: MaterialParams) -> None:
        # Тепловой член обязан деградировать вместе с упругим: он порождается
        # той же жёсткостью. Разделение дало бы напряжения в полностью
        # разрушенном материале.
        full = stress_from_strain(strain(), material, delta_temperature=200.0)
        halved = stress_from_strain(strain(), material, delta_temperature=200.0, multiplier=0.5)

        assert np.allclose(halved, 0.5 * full)

    def test_fully_degraded_material_carries_no_stress(self, material: MaterialParams) -> None:
        eps = strain(xx=1.0e-3)

        sigma = stress_from_strain(eps, material, multiplier=0.0)

        assert np.allclose(sigma, 0.0)


class TestDegradationMultiplier:
    """Маска и повреждаемость входят в жёсткость одним числом: материала нет
    ни там, где отверстие, ни там, где он разрушен."""

    def test_intact_solid_is_unchanged(self) -> None:
        params = DamageParams()

        multiplier = degradation_multiplier(np.array([1.0]), np.array([0.0]), params)

        assert multiplier == pytest.approx(1.0, rel=1e-5)

    def test_void_collapses_to_the_floor(self) -> None:
        params = DamageParams(residual_stiffness=1.0e-6)

        multiplier = degradation_multiplier(np.array([0.0]), np.array([0.0]), params)

        assert multiplier == pytest.approx(1.0e-6)

    def test_fully_damaged_solid_collapses_to_the_floor(self) -> None:
        params = DamageParams(residual_stiffness=1.0e-6)

        multiplier = degradation_multiplier(np.array([1.0]), np.array([1.0]), params)

        assert multiplier == pytest.approx(1.0e-6)

    def test_never_reaches_zero(self) -> None:
        # Ноль сделал бы матрицу жёсткости вырожденной. Полка — не косметика:
        # без неё солвер падал бы посреди многочасового прогона DoE.
        params = DamageParams()
        mask = np.zeros((8, 8))
        damage = np.ones((8, 8))

        assert np.all(degradation_multiplier(mask, damage, params) > 0.0)

    def test_is_monotone_in_damage(self) -> None:
        params = DamageParams()
        damage = np.linspace(0.0, 0.99, 12)

        multiplier = degradation_multiplier(np.ones_like(damage), damage, params)

        assert np.all(np.diff(multiplier) < 0.0)

    def test_combines_both_sources_multiplicatively(self) -> None:
        params = DamageParams(residual_stiffness=1.0e-8)

        both = degradation_multiplier(np.array([0.5]), np.array([0.5]), params)

        assert both == pytest.approx(0.25, rel=1e-6)


class TestVectorisation:
    def test_operates_on_a_field(self, material: MaterialParams) -> None:
        eps = np.zeros((16, 16, 3))
        eps[..., 0] = 1.0e-4

        sigma = stress_from_strain(eps, material)

        assert sigma.shape == (16, 16, 3)
        assert np.allclose(sigma[..., 0], material.youngs_modulus * 1.0e-4 / (1.0 - 0.3**2))

    def test_multiplier_may_vary_over_the_field(self, material: MaterialParams) -> None:
        # Именно так входит поле повреждённости: свой множитель в каждой точке.
        eps = np.zeros((4, 4, 3))
        eps[..., 0] = 1.0e-4
        multiplier = np.linspace(0.0, 1.0, 16).reshape(4, 4)

        sigma = stress_from_strain(eps, material, multiplier=multiplier)

        assert np.allclose(sigma[..., 0] / sigma[-1, -1, 0], multiplier)

    def test_temperature_may_vary_over_the_field(self, material: MaterialParams) -> None:
        eps = np.zeros((4, 4, 3))
        delta_t = np.full((4, 4), 100.0)
        delta_t[0, 0] = 0.0

        sigma = stress_from_strain(eps, material, delta_temperature=delta_t)

        assert sigma[0, 0, 0] == pytest.approx(0.0, abs=1.0)
        assert sigma[1, 1, 0] < 0.0
