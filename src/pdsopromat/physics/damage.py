"""Накопление повреждаемости по Качанову–Работнову с эквивалентным напряжением Хейхёрста.

Контракт C4: этот модуль — единственный источник истины по определяющим соотношениям.
И МКЭ-путь, и суррогатный путь вызывают отсюда одни и те же функции. Тест утверждает
побитовое совпадение приращения повреждаемости при одинаковых входах: без этого
два пути тихо разъезжаются, и отладка уходит в нейросеть, где причины нет.

Напряжение всюду хранится в порядке ``[σ_xx, σ_yy, σ_xy]`` по последней оси, поэтому
одни и те же функции работают и на точке, и на поле ``(H, W, 3)``, и на массиве
значений в точках квадратуры.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pdsopromat.core.spec import DamageParams

FloatArray = npt.NDArray[np.float64]
FloatLike = float | FloatArray


def von_mises_plane_stress(sigma: FloatArray) -> FloatArray:
    """Эквивалентное напряжение Мизеса для плоского напряжённого состояния.

    Знака не различает: под сжатием даёт ту же величину, что под растяжением.
    Поэтому один Мизес как движущая сила повреждаемости физически неверен —
    он «повреждает» сжатую зону, где поры не раскрываются.
    """
    s_xx = sigma[..., 0]
    s_yy = sigma[..., 1]
    s_xy = sigma[..., 2]
    return np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * s_xy**2)


def max_principal_plane_stress(sigma: FloatArray) -> FloatArray:
    """Первое главное напряжение **в плоскости** (круг Мора).

    Внимание: у трёхмерного состояния при плоском напряжённом состоянии есть третье
    главное, равное нулю. При всестороннем сжатии в плоскости эта функция вернёт
    отрицательное значение, тогда как истинное первое главное равно нулю. Разницу
    закрывает скобка Маколея внутри :func:`hayhurst_equivalent_stress`, поэтому
    отдельной трёхмерной ветки здесь нет.
    """
    s_xx = sigma[..., 0]
    s_yy = sigma[..., 1]
    s_xy = sigma[..., 2]
    centre = 0.5 * (s_xx + s_yy)
    radius = np.hypot(0.5 * (s_xx - s_yy), s_xy)
    return centre + radius


def hayhurst_equivalent_stress(sigma: FloatArray, alpha: float) -> FloatArray:
    """χ = α⟨σ_I⟩ + (1−α)σ_vM.

    Скобки Маколея на первом главном — то, что не даёт сжатию накапливать
    повреждение через этот член. Мизесовский член знака по-прежнему не различает,
    и это соответствует исходной модели: под сжатием ползучее повреждение
    подавлено, но не обнулено.

    ``alpha`` передаётся явно, а не берётся из :class:`DamageParams`, чтобы
    функцию можно было звать при построении графиков чувствительности к α,
    не собирая ради этого полный набор параметров.
    """
    tensile_principal = np.maximum(max_principal_plane_stress(sigma), 0.0)
    return alpha * tensile_principal + (1.0 - alpha) * von_mises_plane_stress(sigma)


def _integrity(damage: FloatLike, params: DamageParams) -> FloatArray:
    """(1−d), прижатое к нижней полке — иначе матрица жёсткости вырождается."""
    return np.clip(1.0 - np.asarray(damage, dtype=np.float64), params.residual_stiffness, 1.0)


def damage_rate(chi_bar: FloatLike, damage: FloatLike, params: DamageParams) -> FloatArray:
    """ḋ = (1/t_ref)·(χ̄/σ_ref)^r·(1−d)^(−k), 1/с.

    Вход — уже **нелокальное** (сглаженное скринингом) эквивалентное напряжение.
    Подстановка локального χ сюда формально пройдёт, но вернёт сетко-зависимый
    результат, ради избавления от которого скрининг и вводился.

    Эта функция нужна для верификации и для явного интегрирования на малых шагах.
    Движок разделения масштабов её не использует: он идёт через :func:`kr_exact_jump`.
    """
    chi = np.asarray(chi_bar, dtype=np.float64)
    driving = (chi / params.reference_stress) ** params.exponent_r
    return driving / params.reference_time * _integrity(damage, params) ** (-params.exponent_k)


def damage_driving_integral(
    chi_samples: FloatArray, weights: FloatArray, params: DamageParams
) -> FloatArray:
    """Φ = ∫_блок (1/t_ref)(χ̄(t)/σ_ref)^r dt — безразмерная движущая сила за один блок.

    ``chi_samples`` имеет форму ``(n_q, ...)``: первая ось — точки квадратуры по времени
    внутри блока, остальные — пространственное поле. ``weights`` имеет размерность
    времени и в сумме даёт период блока ``t_c``.

    Повреждаемость за блок считается при **замороженном** ``d``. Это законно, поскольку
    приращение за один блок мало; всё, что дальше делает движок, — прыгает по блокам,
    оставляя Φ постоянной, и именно там возникает контролируемая ошибка экстраполяции.
    """
    if chi_samples.shape[0] != weights.shape[0]:
        msg = (
            f"первая ось chi_samples ({chi_samples.shape[0]}) должна совпадать "
            f"с числом весов квадратуры ({weights.shape[0]})"
        )
        raise ValueError(msg)

    normalised = (chi_samples / params.reference_stress) ** params.exponent_r
    integral: FloatArray = np.tensordot(weights, normalised, axes=(0, 0))
    return integral / params.reference_time


def kr_exact_jump(
    damage: FloatLike, phi: FloatLike, n_blocks: FloatLike, params: DamageParams
) -> FloatArray:
    """Точное продвижение повреждаемости на ``n_blocks`` блоков при замороженной Φ.

    Разделяя переменные в ``ḋ = (χ̄/σ_ref)^r (1−d)^(−k) / t_ref``, получаем, что
    линейно по времени меняется не сама ``d``, а величина ``(1−d)^(k+1)``::

        (1−d_{n+M})^(k+1) = (1−d_n)^(k+1) − (k+1)·M·Φ

    Отсюда два следствия, на которых стоит весь движок:

    * обновление **линейно по M**, то есть прыжок произвольного размера не
      накапливает ошибки интегрирования — она возникает только из-за того,
      что за время прыжка меняется само Φ;
    * жёсткость при ``d → 1`` исчезает. Явная схема по ``d`` схлопывалась бы
      до одного блока ровно у разрушения, то есть там, где начинается
      интересное, и всё ускорение испарялось бы.

    Уход скобки в отрицательную область означает разрушение внутри прыжка;
    результат прижимается к потолку ``max_damage``, а не превращается в NaN.
    Момент разрушения при этом восстанавливается через :func:`blocks_to_damage`.
    """
    exponent = params.exponent_k + 1.0
    start = 1.0 - np.asarray(damage, dtype=np.float64)
    floor = (1.0 - params.max_damage) ** exponent

    remaining = start**exponent - exponent * np.asarray(n_blocks) * np.asarray(phi)
    integrity: FloatArray = np.maximum(remaining, floor) ** (1.0 / exponent)
    return 1.0 - integrity


def blocks_to_damage(
    damage_from: FloatLike, damage_to: FloatLike, phi: FloatLike, params: DamageParams
) -> FloatArray:
    """Число блоков, переводящее повреждаемость из ``damage_from`` в ``damage_to``.

    Обращение :func:`kr_exact_jump`. При ``damage_from=0`` и ``damage_to=max_damage``
    с Φ, посчитанной на неповреждённом поле напряжений, даёт **аналитический базовый
    прогноз ресурса** — тот порог, который суррогат обязан побить. Не побил —
    останавливаемся и чиним физику или данные, а не архитектуру сети.

    Φ = 0 (нет движущей силы) даёт бесконечность: разрушение не наступает никогда.
    """
    exponent = params.exponent_k + 1.0
    start = (1.0 - np.asarray(damage_from, dtype=np.float64)) ** exponent
    end = (1.0 - np.asarray(damage_to, dtype=np.float64)) ** exponent

    phi_array = np.asarray(phi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        blocks: FloatArray = (start - end) / (exponent * phi_array)
    return np.where(phi_array > 0.0, blocks, np.inf)
