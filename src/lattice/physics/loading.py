"""Программа нагружения: температурный профиль внутри блока.

Блок — период программы, а не отдельная физическая сущность. Повреждение копится
во времени по Качанову–Работнову, а движок прыгает по блокам; «цикл» здесь лишь
имя периода. Разделение этих двух смыслов держится тем, что время внутри блока
задаётся долей периода, а не номером шага.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lattice.core.spec import LoadProgram

FloatArray = npt.NDArray[np.float64]


def boundary_temperature(program: LoadProgram, phase: FloatArray) -> FloatArray:
    """Температура поверхности как функция фазы блока ``phase ∈ [0, 1]``.

    Профиль трапециевидный: подъём, выдержка при максимуме, спад. Доля выдержки
    задаётся ``hold_fraction``, оставшееся делится поровну между подъёмом и спадом.

    Внутренность тела следует за поверхностью с отставанием — конечная
    теплопроводность не даёт установиться мгновенно. Именно это отставание
    порождает градиенты, а с ними температурные напряжения, ради которых
    нестационарная задача вообще решается: при мгновенном прогреве тело
    расширялось бы однородно и напряжений не возникло бы вовсе.
    """
    fraction = np.clip(np.asarray(phase, dtype=np.float64), 0.0, 1.0)
    ramp = 0.5 * (1.0 - program.hold_fraction)
    span = program.temperature_hot - program.temperature_cold

    if ramp <= 0.0:
        # Выдержка занимает весь период: подъём мгновенный, тело всё время горячее.
        return np.full_like(fraction, program.temperature_hot)

    rising = program.temperature_cold + span * np.clip(fraction / ramp, 0.0, 1.0)
    falling = program.temperature_hot - span * np.clip(
        (fraction - ramp - program.hold_fraction) / ramp, 0.0, 1.0
    )
    return np.minimum(rising, falling)


def block_sample_times(program: LoadProgram, samples: int) -> tuple[FloatArray, FloatArray]:
    """Моменты внутри блока и веса трапеции для интегрирования Φ.

    Веса имеют размерность времени и в сумме дают период — таков контракт
    :func:`~lattice.physics.damage.damage_driving_integral`.

    Интегрируется величина, пропорциональная χ^r с r ≈ 5, поэтому подынтегральная
    функция много острее самой температуры, и число точек здесь — не формальность.
    Слишком редкая выборка систематически занижает Φ, а с ней завышает ресурс.
    """
    if samples < 2:
        msg = f"нужно минимум 2 точки на блок, получено {samples}"
        raise ValueError(msg)

    times = np.linspace(0.0, program.period, samples)
    step = program.period / (samples - 1)
    weights = np.full(samples, step)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return times, weights
