"""Контракт C2 — состояние связанной задачи в один момент времени.

Состояние неизменяемо, и это не стилистика. Движок разделения масштабов пробует
шаг, оценивает ошибку удвоением и при неудаче откатывается к предыдущему состоянию.
Если бы интегратор менял поля на месте, откат требовал бы ручного снимка в каждой
ветке — и первая же забытая ветка давала бы тихо испорченную траекторию.

Массивы копируются при создании и помечаются доступными только для чтения.
Копия стоит микросекунды против миллисекунд на решение системы, зато исключает
случай, когда вызывающий код сохранил ссылку на «неизменяемое» поле и правит его
задним числом.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from lattice.core.grid import StructuredGrid
from lattice.core.spec import CaseSpec

FloatArray = npt.NDArray[np.float64]


def _frozen_copy(array: npt.ArrayLike) -> FloatArray:
    copied: FloatArray = np.array(array, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


@dataclass(frozen=True)
class State:
    """Поля задачи плюс положение на оси времени.

    Хранятся обе временные координаты — физическое время и число пройденных блоков
    нагружения. Держать только одну нельзя: закон повреждаемости живёт во времени,
    а движок прыгает по блокам, и пересчёт одного в другое на каждом шаге — ровно
    то место, где «повреждение в секунду» и «повреждение за цикл» смешиваются
    и обнуляют метрику ресурса, выглядя при этом правдоподобно.
    """

    temperature: FloatArray
    displacement: FloatArray
    damage: FloatArray
    time: float
    blocks: float
    spec_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature", _frozen_copy(self.temperature))
        object.__setattr__(self, "displacement", _frozen_copy(self.displacement))
        object.__setattr__(self, "damage", _frozen_copy(self.damage))

        shape = self.temperature.shape
        if self.damage.shape != shape or self.displacement.shape != (*shape, 2):
            msg = (
                "несогласованность форм полей: "
                f"температура {self.temperature.shape}, повреждаемость {self.damage.shape}, "
                f"перемещения {self.displacement.shape}"
            )
            raise ValueError(msg)

        if self.damage.min() < 0.0 or self.damage.max() > 1.0:
            msg = (
                f"повреждённость вне [0, 1]: диапазон "
                f"[{self.damage.min():.3e}, {self.damage.max():.3e}]"
            )
            raise ValueError(msg)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return self.temperature.shape

    @classmethod
    def initial(cls, grid: StructuredGrid, spec: CaseSpec) -> State:
        """Неповреждённое тело в покое при холодном конце цикла."""
        shape = (grid.nx, grid.ny)
        return cls(
            temperature=np.full(shape, spec.load.temperature_cold),
            displacement=np.zeros((*shape, 2)),
            damage=np.zeros(shape),
            time=0.0,
            blocks=0.0,
            spec_hash=spec.content_hash(),
        )

    def advanced(
        self,
        *,
        time: float,
        blocks: float,
        temperature: FloatArray | None = None,
        displacement: FloatArray | None = None,
        damage: FloatArray | None = None,
    ) -> State:
        """Новое состояние с проверкой направления процесса.

        Откат времени или залечивание повреждаемости означают перепутанные
        аргументы или сломанный контроллер шага. Такое обязано падать в точке
        возникновения: дальше по траектории причина уже неотличима от
        «сеть плохо обучилась».

        Равенство допускается — staggered-итерация уточняет поля, не двигая время.
        """
        if time < self.time:
            msg = f"время не идёт назад: {time} < {self.time}"
            raise ValueError(msg)
        if blocks < self.blocks:
            msg = f"счётчик блоков не идёт назад: {blocks} < {self.blocks}"
            raise ValueError(msg)

        if damage is not None and np.any(np.asarray(damage) < self.damage - 1e-12):
            deficit = float((self.damage - np.asarray(damage)).max())
            msg = f"повреждённость убывает: максимальное залечивание {deficit:.3e}"
            raise ValueError(msg)

        return replace(
            self,
            temperature=self.temperature if temperature is None else temperature,
            displacement=self.displacement if displacement is None else displacement,
            damage=self.damage if damage is None else damage,
            time=time,
            blocks=blocks,
        )
