"""Программа нагружения: форма температурного профиля и квадратура по блоку."""

from __future__ import annotations

import numpy as np
import pytest

from lattice.core import LoadProgram
from lattice.physics.loading import block_sample_times, boundary_temperature


@pytest.fixture
def program() -> LoadProgram:
    return LoadProgram(
        period=200.0,
        temperature_cold=300.0,
        temperature_hot=600.0,
        hold_fraction=0.6,
    )


class TestTemperatureProfile:
    def test_starts_and_ends_cold(self, program: LoadProgram) -> None:
        edges = boundary_temperature(program, np.array([0.0, 1.0]))

        assert np.allclose(edges, program.temperature_cold)

    def test_reaches_the_hot_level_during_the_hold(self, program: LoadProgram) -> None:
        assert boundary_temperature(program, np.array([0.5])) == pytest.approx(
            program.temperature_hot
        )

    def test_hold_spans_the_declared_fraction(self, program: LoadProgram) -> None:
        phase = np.linspace(0.0, 1.0, 2001)
        at_peak = boundary_temperature(program, phase) > program.temperature_hot - 1e-9

        assert at_peak.mean() == pytest.approx(program.hold_fraction, abs=0.01)

    def test_rises_then_falls(self, program: LoadProgram) -> None:
        # Фаза считается от начала блока. Перепутанный знак или сдвиг дал бы
        # выдержку не на том участке, и повреждение копилось бы в холодной фазе.
        profile = boundary_temperature(program, np.linspace(0.0, 1.0, 101))
        peak = int(np.argmax(profile))

        assert np.all(np.diff(profile[: peak + 1]) >= -1e-12)
        assert np.all(np.diff(profile[peak:]) <= 1e-12)

    def test_stays_within_the_declared_range(self, program: LoadProgram) -> None:
        profile = boundary_temperature(program, np.linspace(-0.5, 1.5, 201))

        assert profile.min() >= program.temperature_cold - 1e-12
        assert profile.max() <= program.temperature_hot + 1e-12

    def test_full_hold_removes_the_ramps(self) -> None:
        # Вырожденный случай: выдержка занимает весь период, деления на ноль быть
        # не должно, тело всё время горячее.
        program = LoadProgram(hold_fraction=1.0, temperature_cold=300.0, temperature_hot=600.0)

        profile = boundary_temperature(program, np.linspace(0.0, 1.0, 11))

        assert np.allclose(profile, 600.0)


class TestQuadrature:
    def test_weights_sum_to_the_period(self, program: LoadProgram) -> None:
        # Контракт с damage_driving_integral: веса имеют размерность времени.
        _, weights = block_sample_times(program, 5)

        assert weights.sum() == pytest.approx(program.period)

    def test_samples_span_the_whole_block(self, program: LoadProgram) -> None:
        times, _ = block_sample_times(program, 4)

        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(program.period)

    def test_trapezoid_halves_the_end_weights(self, program: LoadProgram) -> None:
        _, weights = block_sample_times(program, 5)

        assert weights[0] == pytest.approx(0.5 * weights[1])
        assert weights[-1] == pytest.approx(0.5 * weights[-2])

    def test_single_point_is_rejected(self, program: LoadProgram) -> None:
        with pytest.raises(ValueError, match="минимум 2"):
            block_sample_times(program, 1)
