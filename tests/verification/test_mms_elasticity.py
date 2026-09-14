"""Метод искусственных решений для оператора упругости с деградацией и теплом.

Золотой стандарт верификации PDE-кода. Схема такая: выбирается гладкое поле
перемещений, из него символьно выводится объёмная сила, при которой это поле
является точным решением, затем измеряется, с каким порядком численное решение
к нему сходится. Q1-элементы обязаны давать O(h²) по норме L2.

Почему это сильнее любого сравнения картинок: тест ловит не только грубые ошибки,
но и потерю порядка. Перепутанный коэффициент в форме часто даёт визуально
правдоподобное поле, сходящееся как O(h) — и только измерение наклона это видит.

Правая часть выводится sympy, а не руками. Ошибка в ручной выкладке дала бы
зелёную верификацию неверного солвера, что хуже отсутствия проверки.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import sympy as sp
from skfem import Basis, ElementQuad1, ElementVector, Functional

from lattice.core import MaterialParams
from lattice.core.grid import StructuredGrid
from lattice.physics.elasticity import plane_stress_moduli
from lattice.solver import (
    FemSpace,
    assemble_body_force,
    assemble_stiffness,
    assemble_thermal_load,
    solve_displacement,
)
from lattice.solver.assembly import StiffnessAssembler

Scalar2D = Callable[[np.ndarray, np.ndarray], np.ndarray]
RESOLUTIONS = (9, 17, 33, 65)


@dataclass(frozen=True)
class ManufacturedCase:
    """Точное решение вместе с порождающей его правой частью."""

    ux: Scalar2D
    uy: Scalar2D
    multiplier: Scalar2D
    delta_t: Scalar2D
    fx: Scalar2D
    fy: Scalar2D


def manufacture(
    ux_expr: sp.Expr,
    uy_expr: sp.Expr,
    multiplier_expr: sp.Expr,
    delta_t_expr: sp.Expr,
    material: MaterialParams,
) -> ManufacturedCase:
    """Символьно строит f = −∇·σ для заданного поля перемещений.

    Деградация ``m(x, y)`` стоит **внутри** дивергенции: именно так она входит
    в слабую форму. Вынести её наружу — распространённая ошибка, которая при
    постоянном m незаметна и ломает ровно тот случай, ради которого всё делается.
    """
    x, y = sp.symbols("x y", real=True)
    lam, mu = plane_stress_moduli(material)
    alpha = material.thermal_expansion

    strain = sp.Matrix(
        [
            [sp.diff(ux_expr, x), (sp.diff(ux_expr, y) + sp.diff(uy_expr, x)) / 2],
            [(sp.diff(ux_expr, y) + sp.diff(uy_expr, x)) / 2, sp.diff(uy_expr, y)],
        ]
    )
    thermal = alpha * delta_t_expr * sp.eye(2)
    mechanical = strain - thermal
    trace = mechanical[0, 0] + mechanical[1, 1]

    stress = multiplier_expr * (lam * trace * sp.eye(2) + 2 * mu * mechanical)

    fx_expr = -(sp.diff(stress[0, 0], x) + sp.diff(stress[0, 1], y))
    fy_expr = -(sp.diff(stress[1, 0], x) + sp.diff(stress[1, 1], y))

    def compile_expr(expr: sp.Expr) -> Scalar2D:
        fn = sp.lambdify((x, y), expr, "numpy")

        def evaluate(px: np.ndarray, py: np.ndarray) -> np.ndarray:
            # lambdify схлопывает константное выражение в скаляр — восстанавливаем форму.
            return np.broadcast_to(np.asarray(fn(px, py), dtype=np.float64), px.shape)

        return evaluate

    return ManufacturedCase(
        ux=compile_expr(ux_expr),
        uy=compile_expr(uy_expr),
        multiplier=compile_expr(multiplier_expr),
        delta_t=compile_expr(delta_t_expr),
        fx=compile_expr(fx_expr),
        fy=compile_expr(fy_expr),
    )


def solve_manufactured(
    case: ManufacturedCase, n: int, material: MaterialParams, *, cached: bool = False
) -> float:
    """Решает задачу на решётке n×n и возвращает относительную ошибку L2.

    ``cached=True`` переключает сборку на :class:`StiffnessAssembler`, где множитель
    жёсткости постоянен на элементе, а не интерполирован по узлам. Это другая
    дискретизация, и вопрос, который здесь проверяется, — сохраняет ли она порядок.
    """
    space = FemSpace.from_grid(StructuredGrid(nx=n, ny=n, width=1.0, height=1.0))
    node_x, node_y = space.grid.coordinates()

    multiplier = case.multiplier(node_x, node_y)
    temperature = material.reference_temperature + case.delta_t(node_x, node_y)

    quadrature = np.asarray(space.vector.global_coordinates())
    body_force = np.array(
        [
            case.fx(quadrature[0], quadrature[1]),
            case.fy(quadrature[0], quadrature[1]),
        ]
    )

    stiffness = (
        StiffnessAssembler.build(space, material).assemble(multiplier)
        if cached
        else assemble_stiffness(space, material, multiplier)
    )
    load = assemble_body_force(space, body_force) + assemble_thermal_load(
        space, material, multiplier, temperature
    )

    exact = np.stack([case.ux(node_x, node_y), case.uy(node_x, node_y)], axis=-1).ravel()
    boundary = space.vector.get_dofs().all()
    dofs = solve_displacement(stiffness, load, boundary, exact[boundary])

    # Квадратура повышенного порядка: иначе измерялась бы ошибка интегрирования
    # самой ошибки, и наклон получился бы завышенным.
    fine = Basis(space.mesh, ElementVector(ElementQuad1()), intorder=6)

    @Functional
    def squared_error(w: object) -> np.ndarray:
        coords = w.x  # type: ignore[attr-defined]
        field = w["uh"]  # type: ignore[index]
        dx = field[0] - case.ux(coords[0], coords[1])
        dy = field[1] - case.uy(coords[0], coords[1])
        return dx**2 + dy**2

    @Functional
    def squared_reference(w: object) -> np.ndarray:
        coords = w.x  # type: ignore[attr-defined]
        return case.ux(coords[0], coords[1]) ** 2 + case.uy(coords[0], coords[1]) ** 2

    error = np.sqrt(squared_error.assemble(fine, uh=fine.interpolate(dofs)))
    reference = np.sqrt(squared_reference.assemble(fine))
    return float(error / reference)


def convergence_order(errors: list[float]) -> float:
    """Наклон log(ошибка) по log(1/n): при равномерном измельчении вдвое это порядок."""
    sizes = np.array([1.0 / (n - 1) for n in RESOLUTIONS])
    slope, _ = np.polyfit(np.log(sizes), np.log(errors), 1)
    return float(slope)


@pytest.fixture
def material() -> MaterialParams:
    return MaterialParams(
        youngs_modulus=1.0, poisson_ratio=0.3, thermal_expansion=1.0e-3
    )


@pytest.mark.verification
class TestManufacturedSolutions:
    def test_plain_elasticity_converges_at_second_order(
        self, material: MaterialParams
    ) -> None:
        x, y = sp.symbols("x y", real=True)
        case = manufacture(
            ux_expr=sp.sin(sp.pi * x) * sp.cos(sp.pi * y),
            uy_expr=sp.cos(sp.pi * x) * sp.sin(2 * sp.pi * y),
            multiplier_expr=sp.Integer(1),
            delta_t_expr=sp.Integer(0),
            material=material,
        )

        errors = [solve_manufactured(case, n, material) for n in RESOLUTIONS]

        assert convergence_order(errors) == pytest.approx(2.0, abs=0.15)
        assert errors[-1] < errors[0]

    def test_degraded_operator_keeps_second_order(self, material: MaterialParams) -> None:
        # Главный тест файла: проверяется оператор с переменным множителем жёсткости,
        # то есть ровно тот, через который проходит поле повреждённости. Постоянный m
        # этот путь не задействует.
        x, y = sp.symbols("x y", real=True)
        case = manufacture(
            ux_expr=sp.sin(sp.pi * x) * sp.sin(sp.pi * y),
            uy_expr=sp.sin(2 * sp.pi * x) * sp.sin(sp.pi * y),
            multiplier_expr=sp.Rational(1, 2) + sp.Rational(1, 4) * sp.cos(sp.pi * x * y),
            delta_t_expr=sp.Integer(0),
            material=material,
        )

        errors = [solve_manufactured(case, n, material) for n in RESOLUTIONS]

        assert convergence_order(errors) == pytest.approx(2.0, abs=0.2)

    def test_cached_assembly_keeps_second_order(self, material: MaterialParams) -> None:
        # Кешированная сборка берёт множитель жёсткости постоянным на элементе
        # вместо интерполяции по узлам. Дискретизация другая, и без этой проверки
        # ускорение в 40–74 раза могло бы оказаться куплено потерей порядка —
        # ровно тем, что на глаз не видно, а в ресурсе стоит множителя r.
        x, y = sp.symbols("x y", real=True)
        case = manufacture(
            ux_expr=sp.sin(sp.pi * x) * sp.sin(sp.pi * y),
            uy_expr=sp.sin(2 * sp.pi * x) * sp.sin(sp.pi * y),
            multiplier_expr=sp.Rational(1, 2) + sp.Rational(1, 4) * sp.cos(sp.pi * x * y),
            delta_t_expr=sp.Integer(0),
            material=material,
        )

        errors = [solve_manufactured(case, n, material, cached=True) for n in RESOLUTIONS]

        assert convergence_order(errors) == pytest.approx(2.0, abs=0.2)

    def test_thermal_coupling_keeps_second_order(self, material: MaterialParams) -> None:
        # Коэффициент 2(λ+μ)α в тепловой форме выведен вручную. Стеснённый нагрев
        # проверяет его в постоянном поле; здесь он проверяется в переменном,
        # где ошибка в коэффициенте испортила бы именно наклон.
        x, y = sp.symbols("x y", real=True)
        case = manufacture(
            ux_expr=sp.sin(sp.pi * x) * sp.sin(sp.pi * y),
            uy_expr=sp.sin(sp.pi * x) * sp.sin(2 * sp.pi * y),
            multiplier_expr=sp.Integer(1),
            delta_t_expr=100 * sp.cos(sp.pi * x) * sp.cos(sp.pi * y),
            material=material,
        )

        errors = [solve_manufactured(case, n, material) for n in RESOLUTIONS]

        assert convergence_order(errors) == pytest.approx(2.0, abs=0.2)

    def test_a_deliberately_wrong_operator_fails_the_order_check(
        self, material: MaterialParams
    ) -> None:
        # Контроль самого теста: если подсунуть правую часть от ДРУГОГО материала,
        # решение сойдётся не к тому полю и порядок развалится. Без этой проверки
        # нельзя утверждать, что тест вообще способен что-либо поймать.
        x, y = sp.symbols("x y", real=True)
        wrong = MaterialParams(youngs_modulus=1.0, poisson_ratio=0.45)
        case = manufacture(
            ux_expr=sp.sin(sp.pi * x) * sp.cos(sp.pi * y),
            uy_expr=sp.cos(sp.pi * x) * sp.sin(2 * sp.pi * y),
            multiplier_expr=sp.Integer(1),
            delta_t_expr=sp.Integer(0),
            material=wrong,
        )

        errors = [solve_manufactured(case, n, material) for n in RESOLUTIONS]

        assert convergence_order(errors) < 1.0
