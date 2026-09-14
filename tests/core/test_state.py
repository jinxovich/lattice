"""Контракт C2: состояние неизменяемо, интеграторы возвращают новое, время не идёт назад."""

from __future__ import annotations

import numpy as np
import pytest

from lattice.core import CaseSpec
from lattice.core.grid import StructuredGrid
from lattice.core.state import State


@pytest.fixture
def grid() -> StructuredGrid:
    return StructuredGrid(nx=9, ny=9, width=0.1, height=0.1)


@pytest.fixture
def state(grid: StructuredGrid) -> State:
    return State.initial(grid, CaseSpec())


class TestConstruction:
    def test_initial_state_has_consistent_shapes(self, state: State, grid: StructuredGrid) -> None:
        assert state.temperature.shape == (grid.nx, grid.ny)
        assert state.displacement.shape == (grid.nx, grid.ny, 2)
        assert state.damage.shape == (grid.nx, grid.ny)

    def test_initial_state_starts_undamaged_at_rest(self, state: State) -> None:
        assert np.all(state.damage == 0.0)
        assert np.all(state.displacement == 0.0)
        assert state.time == 0.0
        assert state.blocks == 0.0

    def test_initial_temperature_is_the_cold_end_of_the_cycle(self, state: State) -> None:
        spec = CaseSpec()

        assert np.all(state.temperature == spec.load.temperature_cold)

    def test_spec_hash_is_recorded(self, state: State) -> None:
        # Без этого нельзя проследить результат до породившей его постановки.
        assert state.spec_hash == CaseSpec().content_hash()

    def test_grid_shape_reports_the_scalar_field_shape(
        self, state: State, grid: StructuredGrid
    ) -> None:
        assert state.grid_shape == (grid.nx, grid.ny)

    def test_mismatched_field_shapes_are_rejected(self, grid: StructuredGrid) -> None:
        with pytest.raises(ValueError, match="форм"):
            State(
                temperature=np.zeros((grid.nx, grid.ny)),
                displacement=np.zeros((grid.nx, grid.ny, 2)),
                damage=np.zeros((grid.nx + 1, grid.ny)),
                time=0.0,
                blocks=0.0,
                spec_hash="deadbeef",
            )

    def test_damage_outside_the_unit_interval_is_rejected(self, grid: StructuredGrid) -> None:
        damage = np.zeros((grid.nx, grid.ny))
        damage[0, 0] = 1.5

        with pytest.raises(ValueError, match="повреждённост"):
            State(
                temperature=np.zeros((grid.nx, grid.ny)),
                displacement=np.zeros((grid.nx, grid.ny, 2)),
                damage=damage,
                time=0.0,
                blocks=0.0,
                spec_hash="deadbeef",
            )


class TestImmutability:
    def test_fields_are_read_only(self, state: State) -> None:
        with pytest.raises(ValueError):
            state.damage[0, 0] = 0.5

    def test_attribute_rebinding_is_rejected(self, state: State) -> None:
        with pytest.raises(AttributeError):
            state.time = 1.0  # type: ignore[misc]

    def test_mutating_the_source_array_does_not_leak_in(self, grid: StructuredGrid) -> None:
        # Состояние копирует входные массивы. Иначе вызывающий код сохранял бы
        # живую ссылку и мог менять «неизменяемое» состояние задним числом —
        # ровно тот сорт ошибки, который не видно ни в одном стек-трейсе.
        source = np.zeros((grid.nx, grid.ny))
        built = State(
            temperature=np.zeros((grid.nx, grid.ny)),
            displacement=np.zeros((grid.nx, grid.ny, 2)),
            damage=source,
            time=0.0,
            blocks=0.0,
            spec_hash="deadbeef",
        )

        source[0, 0] = 0.9

        assert built.damage[0, 0] == 0.0


class TestAdvance:
    def test_advanced_returns_a_new_object(self, state: State) -> None:
        moved = state.advanced(time=10.0, blocks=1.0)

        assert moved is not state
        assert state.time == 0.0

    def test_advanced_keeps_unspecified_fields(self, state: State) -> None:
        moved = state.advanced(time=10.0, blocks=1.0)

        assert np.array_equal(moved.temperature, state.temperature)
        assert moved.spec_hash == state.spec_hash

    def test_advanced_replaces_the_damage_field(self, state: State, grid: StructuredGrid) -> None:
        new_damage = np.full((grid.nx, grid.ny), 0.25)

        moved = state.advanced(time=10.0, blocks=1.0, damage=new_damage)

        assert np.allclose(moved.damage, 0.25)
        assert np.all(state.damage == 0.0)

    def test_time_cannot_run_backwards(self, state: State) -> None:
        # Движок прыгает по блокам большими шагами; отрицательный шаг означает,
        # что контроллер шага сломан, и это должно падать сразу, а не копиться.
        moved = state.advanced(time=10.0, blocks=1.0)

        with pytest.raises(ValueError, match="назад"):
            moved.advanced(time=5.0, blocks=2.0)

    def test_blocks_cannot_run_backwards(self, state: State) -> None:
        moved = state.advanced(time=10.0, blocks=5.0)

        with pytest.raises(ValueError, match="назад"):
            moved.advanced(time=20.0, blocks=1.0)

    def test_damage_cannot_heal(self, state: State, grid: StructuredGrid) -> None:
        # Повреждаемость термодинамически необратима. Убывание означает ошибку
        # знака или перепутанные аргументы прыжка.
        damaged = state.advanced(
            time=10.0, blocks=1.0, damage=np.full((grid.nx, grid.ny), 0.3)
        )

        with pytest.raises(ValueError, match="убыва"):
            damaged.advanced(time=20.0, blocks=2.0, damage=np.full((grid.nx, grid.ny), 0.1))

    def test_staying_at_the_same_instant_is_allowed(self, state: State) -> None:
        # Staggered-итерация уточняет поля, не двигая время.
        refined = state.advanced(time=0.0, blocks=0.0)

        assert refined.time == 0.0
