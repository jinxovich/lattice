"""Кешированная пересборка матриц, линейных по множителю.

Замер на этапе верификации показал, что сборка съедает больше времени, чем само
решение системы: на сетке 129×129 это 0.67 с против 1.4 с на решение. В прокате
до отказа механика вызывается сотни раз, и при таком раскладе генерация датасета
вылезала за бюджет недели 4 примерно втридцатеро.

Приём опирается на то, что слабая форма **линейна по множителю**. Если множитель
постоянен на элементе, то::

    A(m) = Σ_e m_e · A_e

где элементные матрицы ``A_e`` от ``m`` не зависят вовсе. Значит их достаточно
проинтегрировать один раз, а каждая последующая пересборка — это поэлементное
умножение и раскладка по структуре разреженной матрицы, то есть O(nnz) без
единого обращения к квадратуре. Измеренное ускорение — 40–74×.

Механизм общий для двух мест, где он нужен: жёсткость, деградирующая полем
повреждаемости, и теплопроводность, деградирующая тем же полем. Поэтому он
вынесен в :class:`MultiplierCache`, а конкретные операторы — тонкие обёртки.

Множитель берётся постоянным на элементе, а не интерполированным по узлам, как
в эталонной сборке :func:`~pdsopromat.solver.elasticity.assemble_stiffness`.
Это другая дискретизация; набор искусственных решений подтверждает, что второй
порядок сходимости она сохраняет.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from pdsopromat.core.spec import MaterialParams
from pdsopromat.physics.elasticity import plane_stress_moduli
from pdsopromat.solver.elasticity import stiffness_form
from pdsopromat.solver.space import FemSpace

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class MultiplierCache:
    """Пересобирает ``A(m)`` без интегрирования.

    Строится один раз на сетку, форму и материал, дальше переиспользуется на всём
    прогоне. Стоимость построения (порядка секунды) окупается уже на втором вызове.
    """

    space: FemSpace
    _unit_data: FloatArray
    _mapping: IntArray
    _pattern: sp.csr_matrix
    _connectivity: IntArray
    _entries_per_element: int

    @classmethod
    def for_form(
        cls, space: FemSpace, form: Any, basis: Any, **form_kwargs: Any
    ) -> Self:
        """Кеш для формы, принимающей множитель под именем ``multiplier``.

        ``basis`` — то пространство, на котором форма собирается: векторное для
        упругости, скалярное для теплопроводности.
        """
        unit_multiplier = space.scalar.interpolate(np.ones(space.scalar.N))
        coo: Any = form.coo_data(basis, multiplier=unit_multiplier, **form_kwargs)

        data = np.asarray(coo.data, dtype=np.float64)
        rows, cols = (np.asarray(index, dtype=np.int64) for index in coo.indices)

        size = basis.N
        pattern = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
        pattern.sort_indices()

        # Ключ «строка·N + столбец» монотонен по порядку хранения CSR, поэтому
        # позиция каждого вклада находится одним двоичным поиском без циклов.
        csr_keys = (
            np.repeat(np.arange(size, dtype=np.int64), np.diff(pattern.indptr)) * size
            + pattern.indices.astype(np.int64)
        )
        mapping = np.searchsorted(csr_keys, rows * size + cols)

        n_elements = space.mesh.t.shape[1]
        return cls(
            space=space,
            _unit_data=data,
            _mapping=mapping.astype(np.int64),
            _pattern=pattern,
            _connectivity=np.asarray(space.mesh.t, dtype=np.int64),
            _entries_per_element=data.size // n_elements,
        )

    def element_multiplier(self, multiplier: FloatArray) -> FloatArray:
        """Узловое поле множителя, усреднённое по углам каждого элемента."""
        nodal = self.space.grid.as_dofs(multiplier)
        averaged: FloatArray = nodal[self._connectivity].mean(axis=0)
        return averaged

    def assemble(self, multiplier: FloatArray) -> sp.csr_matrix:
        """A(m) для узлового поля множителя формы ``(nx, ny)``."""
        expected = (self.space.grid.nx, self.space.grid.ny)
        if multiplier.shape != expected:
            msg = f"ожидался множитель формы {expected}, получено {multiplier.shape}"
            raise ValueError(msg)

        # Раскладка coo_data — (локальная строка, локальный столбец, элемент)
        # в C-порядке, то есть индекс элемента меняется быстрее всех. Отсюда tile,
        # а не repeat: перепутанное здесь дало бы правдоподобную, но неверную матрицу.
        scaled = self._unit_data * np.tile(
            self.element_multiplier(multiplier), self._entries_per_element
        )

        matrix = self._pattern.copy()
        matrix.data = np.bincount(
            self._mapping, weights=scaled, minlength=self._pattern.data.size
        )
        return matrix


class StiffnessAssembler(MultiplierCache):
    """Кеш матрицы жёсткости плоской задачи упругости."""

    @classmethod
    def build(cls, space: FemSpace, material: MaterialParams) -> Self:
        lam, mu = plane_stress_moduli(material)
        return cls.for_form(space, stiffness_form, space.vector, lam=lam, mu=mu)
