"""Структурная фоновая решётка и маска материала.

Модуль намеренно не импортирует ``skfem``: им пользуются и солвер, и суррогат,
а суррогатному слою МКЭ-зависимость запрещена по контракту. Здесь только
геометрия решётки и правила перехода «поле ↔ вектор степеней свободы».

Раскладка узлов согласована с ``MeshQuad.init_tensor`` из scikit-fem: узлы идут
y-быстрейшим порядком, поэтому вектор степеней свободы и поле формы ``(nx, ny)``
— это один и тот же буфер, а ``reshape`` возвращает представление, а не копию.
Именно ради этого геометрия задаётся маской, а не согласованной сеткой: шва,
на котором обычно теряются данные и сходится половина ошибок, просто нет.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lattice.core.spec import Geometry

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class StructuredGrid:
    """Равномерная решётка ``nx × ny`` узлов на прямоугольнике ``width × height``."""

    nx: int
    ny: int
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            msg = f"решётка должна иметь минимум 2 узла по стороне, получено {self.nx}×{self.ny}"
            raise ValueError(msg)

    @property
    def n_nodes(self) -> int:
        return self.nx * self.ny

    @property
    def cell_size(self) -> tuple[float, float]:
        return self.width / (self.nx - 1), self.height / (self.ny - 1)

    @property
    def x(self) -> FloatArray:
        return np.linspace(0.0, self.width, self.nx)

    @property
    def y(self) -> FloatArray:
        return np.linspace(0.0, self.height, self.ny)

    def coordinates(self) -> tuple[FloatArray, FloatArray]:
        """Координаты узлов в форме решётки ``(nx, ny)``."""
        return np.meshgrid(self.x, self.y, indexing="ij")

    def _check_size(self, array: FloatArray, expected: int) -> None:
        if array.size != expected:
            msg = (
                f"ожидалось {expected} степеней свободы, получено {array.size}: "
                "вектор не соответствует решётке"
            )
            raise ValueError(msg)

    def as_scalar_field(self, dofs: FloatArray) -> FloatArray:
        """Скалярный вектор степеней свободы как поле ``(nx, ny)`` — представление."""
        self._check_size(dofs, self.n_nodes)
        return dofs.reshape(self.nx, self.ny)

    def as_vector_field(self, dofs: FloatArray) -> FloatArray:
        """Векторный вектор степеней свободы как поле ``(nx, ny, 2)`` — представление.

        Компоненты в ``ElementVector`` чередуются (u_x, u_y на каждом узле), что
        совпадает с C-порядком последней оси.
        """
        self._check_size(dofs, 2 * self.n_nodes)
        return dofs.reshape(self.nx, self.ny, 2)

    def as_dofs(self, field: FloatArray) -> FloatArray:
        """Обратный переход. Для непрерывного поля возвращает представление."""
        return field.reshape(-1)


def material_mask(
    grid: StructuredGrid, geometry: Geometry, interface_fraction: float = 0.1
) -> FloatArray:
    """Индикатор материала: 1 — тело, 0 — отверстие, промежуточное — переходная полоса.

    Геометрия входит в расчёт через этот массив, а не через перестроение сетки.
    Следствие: радиус отверстия становится параметром DoE бесплатно, а вход CNN
    получает форму объекта как обычный канал.

    ``interface_fraction`` задаёт ширину перехода **в долях радиуса отверстия**,
    то есть в физических единицах, а не в ячейках. Различие принципиальное, и оно
    выяснилось на верификации: при ширине, привязанной к ячейке, полоса сжимается
    вместе с сеткой, маска стремится к ступеньке, и углы пикселей начинают давать
    искусственную концентрацию. Такая маска описывает геометрию, **зависящую от
    сетки**, — то есть непрерывной задачи, к которой могло бы сходиться численное
    решение, попросту не существует, и сеточная сходимость недостижима в принципе.

    С физической шириной задача определена корректно: решение сходится к пластине
    с отверстием заданного градиентного профиля. Это не резкое круглое отверстие,
    и разницу с решением Кирша следует называть ценой погружённой границы, а не
    ошибкой солвера.

    Нулевое значение даёт чистую ступеньку. Оставлено для набора верификации,
    который замеряет, во что эта ступенька обходится; для расчётов не годится.
    """
    xx, yy = grid.coordinates()
    centre_x, centre_y = 0.5 * geometry.width, 0.5 * geometry.height
    signed_distance = np.hypot(xx - centre_x, yy - centre_y) - geometry.hole_radius

    if interface_fraction <= 0.0:
        return (signed_distance >= 0.0).astype(np.float64)

    band = interface_fraction * geometry.hole_radius
    mask: FloatArray = np.clip(signed_distance / band + 0.5, 0.0, 1.0)
    return mask


def stiffness_multiplier(mask: FloatArray, floor: float) -> FloatArray:
    """Множитель жёсткости ersatz-материала: ``floor + (1−floor)·mask``.

    Пустота получает малую, но ненулевую жёсткость. Чистый ноль оставил бы узлы
    внутри отверстия без единого уравнения, матрица стала бы вырожденной, и солвер
    падал бы не в момент ошибки, а через несколько часов прогона DoE.
    """
    return floor + (1.0 - floor) * mask
