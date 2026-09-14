"""Нелокальная регуляризация движущей силы повреждаемости по Пирлингсу.

Локальное софтенинг-повреждение теряет эллиптичность: повреждение схлопывается
в полосу шириной в один элемент, и предсказанный ресурс становится функцией шага
сетки. Метрика, зависящая от дискретизации, обесценивает весь бенчмарк — сравнивать
суррогат с эталоном, который сам не сошёлся, бессмысленно.

Лечение — неявный градиентный скрининг::

    χ̄ − l_c²∇²χ̄ = χ,    ∇χ̄·n = 0

Ширина зоны локализации задаётся длиной ``l_c``, а не размером ячейки, и ресурс
перестаёт зависеть от сетки.

Побочное следствие, которое стоит назвать отдельно: отфильтрованное поле гладко
и ограничено по полосе на масштабе ``l_c``. То есть регуляризация здесь не только
физическая гигиена, но и **предпосылка обучаемости** — свёрточная сеть представляет
такое поле несопоставимо легче, чем одноэлементную сингулярность. Физическая
корректность и обучаемость здесь оказываются одним и тем же требованием.

Оператор строится один раз: матрица ``M + l_c²L`` постоянна, её разложение
переиспользуется на всём прогоне. Это принципиально для движка прыжков, где
скрининг вызывается на каждом шаге.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import BilinearForm
from skfem.helpers import dot, grad

from pdsopromat.core.grid import StructuredGrid
from pdsopromat.solver.space import FemSpace

FloatArray = npt.NDArray[np.float64]


@BilinearForm
def _mass_form(u: Any, v: Any, w: Any) -> Any:
    return u * v


@BilinearForm
def _laplacian_form(u: Any, v: Any, w: Any) -> Any:
    return dot(grad(u), grad(v))


@dataclass(frozen=True)
class ScreeningOperator:
    """Предсобранный фильтр ``χ ↦ χ̄``, работающий с узловыми полями.

    Вход и выход узловые — не в точках квадратуры — намеренно. МКЭ-путь получает
    напряжение в точках квадратуры и мог бы собирать правую часть ∫χv напрямую,
    что на полпорядка точнее. Но суррогатный путь знает поле только в узлах, и
    два разных скрининга означали бы, что пути расходятся необнаружимо: поля
    выглядят одинаково, а предсказанный ресурс отличается.

    Согласованность здесь важнее второго порядка малости, поэтому оба пути
    проецируют χ в узлы и вызывают один и тот же оператор.

    Применять умеет чистый scipy: ``skfem`` нужен только при сборке. Это позволяет
    движку передать оператор в суррогатный контур, не втаскивая туда МКЭ.
    """

    grid: StructuredGrid
    mass: sp.csr_matrix
    _solve: Any
    length: float

    @classmethod
    def build(cls, space: FemSpace, length: float) -> ScreeningOperator:
        if length <= 0.0:
            msg = f"длина скрининга должна быть положительной, получено {length}"
            raise ValueError(msg)

        mass = _mass_form.assemble(space.scalar)
        laplacian = _laplacian_form.assemble(space.scalar)
        screened = (mass + length**2 * laplacian).tocsc()

        return cls(
            grid=space.grid,
            mass=mass.tocsr(),
            _solve=spla.factorized(screened),
            length=length,
        )

    def apply(self, field: FloatArray) -> FloatArray:
        """Сглаживает узловое поле формы ``(nx, ny)``, сохраняя его форму."""
        expected = (self.grid.nx, self.grid.ny)
        if field.shape != expected:
            msg = f"ожидалось поле формы {expected}, получено {field.shape}"
            raise ValueError(msg)

        rhs = self.mass @ self.grid.as_dofs(field)
        return np.asarray(self._solve(rhs)).reshape(expected)

    def integrate(self, field: FloatArray) -> float:
        """∫field dx в конечноэлементном смысле.

        Нужен для проверки сохранения: скрининг перераспределяет величину
        в пространстве, но не создаёт и не уничтожает её. Условие Неймана
        на границе обращает вклад лапласиана в ноль, поэтому интеграл
        обязан сохраняться точно.
        """
        return float(np.sum(self.mass @ self.grid.as_dofs(field)))
