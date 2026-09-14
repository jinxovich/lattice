"""Конечноэлементное пространство над структурной решёткой.

Единственное место в проекте, где ``skfem`` встречается с ``StructuredGrid``.
Здесь же проверяется инвариант, ради которого выбрана структурная сетка: узлы
``MeshQuad.init_tensor`` идут ровно в том порядке, в каком лежит поле формы
``(nx, ny)``. Пока это так, перевод «поле ↔ вектор степеней свободы» есть
``reshape``, то есть представление без копирования и без интерполяции.

Проверка стоит микросекунды и делается один раз при построении пространства.
Молчаливое нарушение этого инварианта — смена версии skfem, другой конструктор
сетки — испортило бы весь датасет способом, который не виден ни на одной картинке:
поля выглядели бы правдоподобно, просто транспонированными.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skfem import Basis, ElementQuad1, ElementVector, FacetBasis, MeshQuad

from pdsopromat.core.grid import StructuredGrid

_ORDERING_TOLERANCE = 1e-12


@dataclass(frozen=True)
class FemSpace:
    """Сетка и базисы: векторный для перемещений, скалярный для полей."""

    grid: StructuredGrid
    mesh: MeshQuad
    vector: Basis
    scalar: Basis

    @classmethod
    def from_grid(cls, grid: StructuredGrid) -> FemSpace:
        mesh = MeshQuad.init_tensor(grid.x, grid.y)
        vector = Basis(mesh, ElementVector(ElementQuad1()))
        # with_element сохраняет точки квадратуры векторного базиса. Без этого
        # интерполяция множителя жёсткости попадала бы не в те точки, в которых
        # ведётся сборка, и ошибка проявилась бы только на неоднородных полях.
        scalar = vector.with_element(ElementQuad1())

        space = cls(grid=grid, mesh=mesh, vector=vector, scalar=scalar)
        space._assert_node_ordering()
        return space

    def _assert_node_ordering(self) -> None:
        expected_x, expected_y = self.grid.coordinates()
        actual_x = self.mesh.p[0].reshape(self.grid.nx, self.grid.ny)
        actual_y = self.mesh.p[1].reshape(self.grid.nx, self.grid.ny)

        if not (
            np.allclose(actual_x, expected_x, atol=_ORDERING_TOLERANCE)
            and np.allclose(actual_y, expected_y, atol=_ORDERING_TOLERANCE)
        ):
            msg = (
                "порядок узлов сетки разошёлся с раскладкой решётки: reshape перестал "
                "быть представлением. Датасет, собранный после такого расхождения, "
                "непригоден — поля окажутся транспонированными"
            )
            raise RuntimeError(msg)

    def facet_basis(self, edge: str) -> FacetBasis:
        """Базис на одной из кромок прямоугольника для приложения нагрузки."""
        predicates = {
            "left": lambda x: x[0] < _ORDERING_TOLERANCE,
            "right": lambda x: x[0] > self.grid.width - _ORDERING_TOLERANCE,
            "bottom": lambda x: x[1] < _ORDERING_TOLERANCE,
            "top": lambda x: x[1] > self.grid.height - _ORDERING_TOLERANCE,
        }
        if edge not in predicates:
            msg = f"неизвестная кромка {edge!r}, ожидалось одно из {sorted(predicates)}"
            raise ValueError(msg)

        return FacetBasis(
            self.mesh,
            ElementVector(ElementQuad1()),
            facets=self.mesh.facets_satisfying(predicates[edge]),
        )
