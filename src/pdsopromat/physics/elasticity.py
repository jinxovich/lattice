"""Определяющие соотношения плоского напряжённого состояния.

Контракт C4 в действии: эти функции зовут оба пути. МКЭ считает деформацию
в точках квадратуры, суррогат — конечными разностями по предсказанному полю
перемещений, но напряжение из деформации оба получают здесь. Отдельная копия
формулы в суррогатном слое означала бы, что две ветки расходятся необнаружимо:
на картинке поля выглядят одинаково, а ресурс отличается вдвое.

Компоненты всюду тензорные: ``[ε_xx, ε_yy, ε_xy]`` и ``[σ_xx, σ_yy, σ_xy]``,
без инженерного сдвига γ = 2ε_xy. Смешение двух конвенций даёт ровно двойку
в сдвиговом члене — ошибку, которая проходит все тесты на одноосное растяжение.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pdsopromat.core.grid import stiffness_multiplier
from pdsopromat.core.spec import DamageParams, MaterialParams

FloatArray = npt.NDArray[np.float64]
FloatLike = float | FloatArray


def degradation_multiplier(
    mask: FloatArray, damage: FloatArray, params: DamageParams
) -> FloatArray:
    """Совокупный множитель жёсткости: ersatz-маска × целостность ``(1−d̄)``.

    Обе причины падения жёсткости входят одним числом, потому что в определяющем
    соотношении они неразличимы: материала нет ни там, где отверстие, ни там, где
    он разрушен. Множитель прижат к полке снизу — матрица жёсткости обязана
    остаться положительно определённой даже при полностью разрушенной зоне.
    """
    ersatz = stiffness_multiplier(mask, params.residual_stiffness)
    integrity = np.clip(1.0 - damage, params.residual_stiffness, 1.0)
    return ersatz * integrity


def plane_stress_moduli(material: MaterialParams) -> tuple[float, float]:
    """Параметры Ламе для плоского напряжённого состояния: ``(λ, μ)``.

    Модуль сдвига совпадает с трёхмерным, а вот ``λ = Eν/(1−ν²)`` отличается от
    трёхмерного ``Eν/((1+ν)(1−2ν))``. Подстановка трёхмерного значения —
    классическая ошибка: она не меняет сдвиговый отклик и ломает только объёмный,
    поэтому тесты на чистый сдвиг остаются зелёными.
    """
    E = material.youngs_modulus
    nu = material.poisson_ratio
    lam = E * nu / (1.0 - nu**2)
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def thermal_strain(material: MaterialParams, delta_temperature: FloatLike) -> FloatArray:
    """Свободная тепловая деформация ``α·ΔT`` — изотропная, без сдвига.

    ``ΔT`` отсчитывается от ``reference_temperature``, при которой тело
    не напряжено.
    """
    dilation = material.thermal_expansion * np.asarray(delta_temperature, dtype=np.float64)
    return np.stack([dilation, dilation, np.zeros_like(dilation)], axis=-1)


def stress_from_strain(
    strain: FloatArray,
    material: MaterialParams,
    *,
    delta_temperature: FloatLike = 0.0,
    multiplier: FloatLike = 1.0,
) -> FloatArray:
    """σ = m·[λ·tr(ε−ε_th)·I + 2μ(ε−ε_th)] для плоского напряжённого состояния.

    ``multiplier`` — совокупный множитель жёсткости: произведение ``(1−d̄)``
    на ersatz-множитель маски. Он умножает **и** упругий, **и** тепловой член,
    потому что тепловое напряжение порождается той же жёсткостью. Разделить их —
    значит получить ненулевые напряжения в полностью разрушенном материале.

    ``delta_temperature`` и ``multiplier`` могут быть полями: их формы
    транслируются на ведущие оси ``strain``.
    """
    lam, mu = plane_stress_moduli(material)

    mechanical = np.asarray(strain, dtype=np.float64) - thermal_strain(
        material, delta_temperature
    )
    trace = mechanical[..., 0] + mechanical[..., 1]

    volumetric = lam * trace
    sigma = 2.0 * mu * mechanical
    sigma[..., 0] += volumetric
    sigma[..., 1] += volumetric

    return np.asarray(multiplier, dtype=np.float64)[..., np.newaxis] * sigma
