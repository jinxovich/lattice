"""Итерационный решатель: совпадение с прямым, кеш предобуславливателя, тёплый старт.

Отдельно закреплён результат, который задаёт формулировку всего проекта: точное
начальное приближение почти не ускоряет решение при хорошем предобуславливателе.
Это утверждение здесь — исполняемое, чтобы оно не выродилось со временем
в удобное «суррогат ускоряет решатель в разы».
"""

from __future__ import annotations

import numpy as np
import pytest

from pdsopromat.core import MaterialParams
from pdsopromat.core.grid import StructuredGrid
from pdsopromat.solver import (
    FemSpace,
    StiffnessAssembler,
    assemble_traction,
    equilibrium_residual,
    solve_displacement,
)
from pdsopromat.solver.linear import WarmSolver

TOL = 1e-9
RESOLUTION = 33


@pytest.fixture
def space() -> FemSpace:
    return FemSpace.from_grid(
        StructuredGrid(nx=RESOLUTION, ny=RESOLUTION, width=1.0, height=1.0)
    )


@pytest.fixture
def material() -> MaterialParams:
    return MaterialParams(youngs_modulus=1.0, poisson_ratio=0.3)


@pytest.fixture
def constrained(space: FemSpace) -> np.ndarray:
    return np.concatenate(
        [
            space.vector.get_dofs(lambda x: x[0] < TOL).all("u^1"),
            space.vector.get_dofs(lambda x: x[1] < TOL).all("u^2"),
        ]
    )


def damage_bump(space: FemSpace, level: float) -> np.ndarray:
    """Локализованное ослабление в центре — как в реальном прокате."""
    xx, yy = space.grid.coordinates()
    return 1.0 - level * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02)


class TestAgreementWithDirectSolve:
    def test_matches_the_direct_solution(
        self, space: FemSpace, material: MaterialParams, constrained: np.ndarray
    ) -> None:
        assembler = StiffnessAssembler.build(space, material)
        stiffness = assembler.assemble(damage_bump(space, 0.3))
        load = assemble_traction(space, "right", (1.0e-3, 0.0))

        iterative, report = WarmSolver(space).solve(stiffness, load, constrained)
        direct = solve_displacement(stiffness, load, constrained)

        assert report.converged
        assert np.linalg.norm(iterative - direct) / np.linalg.norm(direct) < 1e-6

    def test_reported_residual_matches_an_independent_computation(
        self, space: FemSpace, material: MaterialParams, constrained: np.ndarray
    ) -> None:
        # Критерий остановки и метрика доверия обязаны быть одной величиной.
        # Если они разойдутся, сертификат будет утверждать не то, что проверялось.
        assembler = StiffnessAssembler.build(space, material)
        stiffness = assembler.assemble(damage_bump(space, 0.2))
        load = assemble_traction(space, "right", (1.0e-3, 0.0))

        solution, report = WarmSolver(space).solve(stiffness, load, constrained)
        independent = equilibrium_residual(stiffness, load, solution, constrained)

        assert report.residual == pytest.approx(independent, rel=0.5, abs=1e-10)

    def test_guess_does_not_change_the_answer(
        self, space: FemSpace, material: MaterialParams, constrained: np.ndarray
    ) -> None:
        # Начальное приближение влияет на число итераций, но не на ответ.
        # Именно это делает сертифицированный режим безопасным: подсунуть туда
        # можно что угодно, включая плохо обученную сеть.
        assembler = StiffnessAssembler.build(space, material)
        stiffness = assembler.assemble(damage_bump(space, 0.3))
        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        solver = WarmSolver(space)

        cold, _ = solver.solve(stiffness, load, constrained)
        rng = np.random.default_rng(0)
        nonsense = rng.normal(size=cold.shape) * np.linalg.norm(cold)
        warm, report = solver.solve(stiffness, load, constrained, guess=nonsense)

        assert report.converged
        assert np.linalg.norm(warm - cold) / np.linalg.norm(cold) < 1e-6


class TestPreconditionerCache:
    def test_first_call_builds_and_second_reuses(
        self, space: FemSpace, material: MaterialParams, constrained: np.ndarray
    ) -> None:
        assembler = StiffnessAssembler.build(space, material)
        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        solver = WarmSolver(space)

        _, first = solver.solve(assembler.assemble(damage_bump(space, 0.1)), load, constrained)
        _, second = solver.solve(assembler.assemble(damage_bump(space, 0.15)), load, constrained)

        assert first.preconditioner_rebuilt
        assert not second.preconditioner_rebuilt
        assert solver.rebuilds == 1

    def test_changed_constraints_force_a_rebuild(
        self, space: FemSpace, material: MaterialParams, constrained: np.ndarray
    ) -> None:
        # Иерархия относится к конкретной сжатой системе. Переиспользовать её
        # при другом наборе закреплений — значит предобуславливать не ту матрицу.
        assembler = StiffnessAssembler.build(space, material)
        stiffness = assembler.assemble(np.ones((RESOLUTION, RESOLUTION)))
        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        solver = WarmSolver(space)

        solver.solve(stiffness, load, constrained)
        wider = np.concatenate(
            [constrained, space.vector.get_dofs(lambda x: x[0] < TOL).all("u^2")]
        )
        _, report = solver.solve(stiffness, load, wider)

        assert report.preconditioner_rebuilt

    def test_severe_damage_triggers_a_rebuild(
        self, space: FemSpace, material: MaterialParams, constrained: np.ndarray
    ) -> None:
        # По мере локализации матрица портится, и старая иерархия перестаёт
        # окупаться. Порог по числу итераций ловит это без ручной настройки.
        assembler = StiffnessAssembler.build(space, material)
        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        solver = WarmSolver(space, rebuild_threshold=3)

        solver.solve(assembler.assemble(damage_bump(space, 0.1)), load, constrained)
        _, second = solver.solve(
            assembler.assemble(damage_bump(space, 0.95)), load, constrained
        )

        assert second.preconditioner_rebuilt


class TestWarmStartIsOnlyModestlyUseful:
    """Результат, определяющий, что именно проект вправе утверждать.

    Быстрая сходимость многосеточного метода означает, что улучшение старта
    окупается логарифмически: AMG сокращает невязку примерно вшестеро за итерацию,
    поэтому стократно лучший старт стоит log(100)/log(6) ≈ 2.6 итерации.
    """

    @pytest.mark.verification
    def test_accurate_guess_saves_only_a_couple_of_iterations(
        self, space: FemSpace, material: MaterialParams, constrained: np.ndarray
    ) -> None:
        assembler = StiffnessAssembler.build(space, material)
        stiffness = assembler.assemble(damage_bump(space, 0.3))
        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        solver = WarmSolver(space)

        exact, cold = solver.solve(stiffness, load, constrained)

        direction = np.random.default_rng(1).normal(size=exact.shape)
        direction /= np.linalg.norm(direction)
        one_percent = exact + 0.01 * np.linalg.norm(exact) * direction
        _, warm = solver.solve(stiffness, load, constrained, guess=one_percent)

        assert warm.iterations <= cold.iterations
        # Суть утверждения: приближение точностью 1% НЕ сокращает работу вдвое.
        # Если этот assert когда-нибудь упадёт, значит изменился предобуславливатель,
        # и заявление о вкладе суррогата нужно перемерять, а не подправлять порог.
        assert warm.iterations > 0.5 * cold.iterations, (
            f"холодный {cold.iterations}, тёплый {warm.iterations}: "
            "выигрыш оказался больше ожидаемого — перемерить, а не ослаблять тест"
        )

    @pytest.mark.verification
    def test_a_useless_guess_does_not_blow_up_the_iteration_count(
        self, space: FemSpace, material: MaterialParams, constrained: np.ndarray
    ) -> None:
        # Обратная сторона той же гарантии: плохое предсказание стоит нескольких
        # лишних итераций, а не расходимости. Сертифицированный режим остаётся
        # безопасным даже при суррогате, вышедшем из распределения.
        assembler = StiffnessAssembler.build(space, material)
        stiffness = assembler.assemble(damage_bump(space, 0.3))
        load = assemble_traction(space, "right", (1.0e-3, 0.0))
        solver = WarmSolver(space)

        exact, cold = solver.solve(stiffness, load, constrained)
        rubbish = 50.0 * np.linalg.norm(exact) * np.random.default_rng(2).normal(
            size=exact.shape
        )
        _, bad = solver.solve(stiffness, load, constrained, guess=rubbish)

        assert bad.converged
        assert bad.iterations < 4 * cold.iterations
