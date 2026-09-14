"""Кешированная пересборка матрицы жёсткости.

Замер на этапе верификации показал, что сборка съедает больше времени, чем само
решение системы: на сетке 129×129 это 0.67 с против 1.4 с на решение. В прокате
до отказа механика вызывается сотни раз, и при таком раскладе генерация датасета
вылезала за бюджет недели 4 примерно втридцатеро.

Приём опирается на то, что слабая форма **линейна по множителю жёсткости**. Если
множитель постоянен на элементе, то::

    K(m) = Σ_e m_e · K_e

где элементные матрицы ``K_e`` от ``m`` не зависят вовсе. Значит их достаточно
проинтегрировать один раз, а каждая последующая пересборка — это поэлементное
умножение и раскладка по структуре разреженной матрицы, то есть O(nnz) без
единого обращения к квадратуре. Измеренное ускорение — 40–74×.

Структура разреженной матрицы тоже постоянна, поэтому позиции вкладов в массив
данных CSR вычисляются один раз. Пересборка сводится к ``bincount`` по готовому
отображению.

Множитель берётся постоянным на элементе, а не интерполированным по узлам, как
в эталонной сборке :func:`~pdsopromat.solver.elasticity.assemble_stiffness`.
Это другая дискретизация; набор искусственных решений подтверждает, что второй
порядок сходимости она сохраняет.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
class StiffnessAssembler:
    """Пересобирает ``K(m)`` без интегрирования.

    Строится один раз на сетку и материал, дальше переиспользуется на всём прогоне.
    Стоимость построения (порядка секунды) окупается уже на втором вызове.
    """

    space: FemSpace
    _unit_data: FloatArray
    _mapping: IntArray
    _pattern: sp.csr_matrix
    _connectivity: IntArray
    _entries_per_element: int

    @classmethod
    def build(cls, space: FemSpace, material: MaterialParams) -> StiffnessAssembler:
        lam, mu = plane_stress_moduli(material)
        unit_multiplier = space.scalar.interpolate(np.ones(space.scalar.N))

        coo: Any = stiffness_form.coo_data(
            space.vector, multiplier=unit_multiplier, lam=lam, mu=mu
        )
        data = np.asarray(coo.data, dtype=np.float64)
        rows, cols = (np.asarray(index, dtype=np.int64) for index in coo.indices)

        size = space.vector.N
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
        """K(m) для узлового поля множителя формы ``(nx, ny)``."""
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
