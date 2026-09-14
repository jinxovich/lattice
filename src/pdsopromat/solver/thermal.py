"""Нестационарная теплопроводность с деградацией по полю повреждаемости.

Третий физический процесс связки. Уравнение::

    ρc ∂T/∂t = ∇·(k(d)∇T),    k(d) = k₀·(1−d̄)·маска

Деградация теплопроводности — не украшение. Без неё связка была бы односторонней
цепочкой «тепло → напряжения → повреждение» с единственной обратной связью
через жёсткость, тогда как карточка проекта говорит о процессах, взаимодействующих
одновременно. С ``k(d)`` повреждение влияет на температурное поле, температурное
поле — на напряжения, напряжения — обратно на повреждение. Связь становится
двусторонней по существу, а не по формулировке.

Отверстие получает ту же почти нулевую проводимость, что и жёсткость: тепло сквозь
пустоту не идёт. Отдельной обработки границы отверстия при этом не требуется —
погружённая формулировка даёт её естественным образом, как и в механике.

Шаг по времени — θ-схема. По умолчанию θ=1, неявный Эйлер: первого порядка,
но безусловно монотонный.

Кранк–Николсон (θ=0.5) второго порядка и доступен, но умолчанием быть не может.
Его множитель усиления (1−λΔt/2)/(1+λΔt/2) при λΔt ≫ 1 стремится к **−1**: жёсткие
сеточные моды не подавляются, а знакопеременно ползут. Измерено на выходе
к стационару при h=1/32 и Δt=0.05 — за 200 шагов остаётся рябь 1.5·10⁻⁴, тогда как
физическое время установления давно прошло. На мелкой сетке это неизбежно, потому
что λ_max растёт как α/h², каким бы разумным ни был шаг по времени.

При достаточно мелком Δt оба варианта совпадают с аналитикой слоя, так что выбор
здесь про устойчивость к крупному шагу, а не про точность. Немонотонность
недопустима именно в этой связке: температура порождает напряжения, а те входят
в закон повреждаемости в степени r ≈ 5, где любой нефизичный выброс усиливается.

Матрица левой части постоянна, пока не меняются шаг по времени и поле
повреждаемости. Внутри одного блока нагружения оба заморожены, поэтому разложение
строится один раз на блок и переиспользуется на всех его шагах.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import BilinearForm, condense
from skfem.helpers import dot, grad

from pdsopromat.core.spec import MaterialParams
from pdsopromat.solver.assembly import MultiplierCache
from pdsopromat.solver.space import FemSpace

FloatArray = npt.NDArray[np.float64]

CRANK_NICOLSON = 0.5
BACKWARD_EULER = 1.0


@BilinearForm
def conduction_form(u: Any, v: Any, w: Any) -> Any:
    """m·k₀·∇u·∇v — кондуктивный оператор с деградацией."""
    return w["multiplier"] * w["conductivity"] * dot(grad(u), grad(v))


@BilinearForm
def capacity_form(u: Any, v: Any, w: Any) -> Any:
    """ρ·c_p·u·v — теплоёмкость.

    Множителем не деградирует: разрушенный материал теряет способность передавать
    тепло, но не перестаёт его запасать. Обнулить теплоёмкость означало бы получить
    бесконечную скорость установления температуры в повреждённой зоне.
    """
    return w["capacity"] * u * v


@dataclass(frozen=True)
class ThermalOperator:
    """Предсобранные матрицы теплопроводности для одной сетки и материала."""

    space: FemSpace
    material: MaterialParams
    capacity: sp.csr_matrix
    conduction: MultiplierCache

    @classmethod
    def build(cls, space: FemSpace, material: MaterialParams) -> ThermalOperator:
        capacity: sp.csr_matrix = capacity_form.assemble(
            space.scalar, capacity=material.density * material.specific_heat
        )
        conduction = MultiplierCache.for_form(
            space,
            conduction_form,
            space.scalar,
            conductivity=material.conductivity,
        )
        return cls(
            space=space,
            material=material,
            capacity=capacity.tocsr(),
            conduction=conduction,
        )

    def stepper(
        self,
        multiplier: FloatArray,
        time_step: float,
        dirichlet_dofs: FloatArray,
        *,
        theta: float = BACKWARD_EULER,
    ) -> ThermalStepper:
        """Шагатель с замороженными полем повреждаемости и шагом по времени."""
        if time_step <= 0.0:
            msg = f"шаг по времени должен быть положительным, получено {time_step}"
            raise ValueError(msg)
        if not 0.0 <= theta <= 1.0:
            msg = f"θ должно лежать в [0, 1], получено {theta}"
            raise ValueError(msg)

        conduction = self.conduction.assemble(multiplier)
        return ThermalStepper(
            capacity=self.capacity,
            conduction=conduction,
            time_step=time_step,
            theta=theta,
            dirichlet_dofs=np.asarray(dirichlet_dofs),
            grid_shape=(self.space.grid.nx, self.space.grid.ny),
        )


@dataclass(frozen=True)
class ThermalStepper:
    """Один шаг θ-схемы с разложением, построенным заранее.

    Левая часть ``C/Δt + θ·K`` не зависит ни от температуры, ни от времени, поэтому
    внутри блока нагружения она собирается и раскладывается ровно один раз.
    """

    capacity: sp.csr_matrix
    conduction: sp.csr_matrix
    time_step: float
    theta: float
    dirichlet_dofs: FloatArray
    grid_shape: tuple[int, int]

    def advance(
        self, temperature: FloatArray, boundary_temperature: float
    ) -> FloatArray:
        """Продвигает поле температуры на один шаг.

        Граница держится при заданной температуре: программа нагружения задаёт
        нагрев и охлаждение поверхности, а внутренность отстаёт из-за конечной
        теплопроводности. Именно это отставание и порождает градиенты, а с ними
        температурные напряжения — ради них весь процесс и считается.
        """
        if temperature.shape != self.grid_shape:
            msg = f"ожидалось поле формы {self.grid_shape}, получено {temperature.shape}"
            raise ValueError(msg)

        scaled_capacity = self.capacity / self.time_step
        left = (scaled_capacity + self.theta * self.conduction).tocsr()
        right = (scaled_capacity - (1.0 - self.theta) * self.conduction) @ (
            temperature.reshape(-1)
        )

        prescribed = np.full(left.shape[0], boundary_temperature)
        reduced, load, full, free = condense(
            left, right, x=prescribed, D=self.dirichlet_dofs
        )

        solution = np.asarray(full, dtype=np.float64).copy()
        solution[free] = spla.spsolve(reduced.tocsc(), load)
        return solution.reshape(self.grid_shape)
