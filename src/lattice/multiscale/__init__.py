"""Разделение временных масштабов — ответ на барьер из карточки проекта.

Напряжения живут на масштабе секунд, накопление повреждаемости — на масштабе лет.
Считать каждый блок нагружения по отдельности невозможно, поэтому быстрый процесс
разрешается внутри одного блока, а медленный продвигается прыжками через тысячи
блоков сразу.

Слой намеренно не знает, кто именно решает механику. И эталонный МКЭ, и суррогат
реализуют один интерфейс, поэтому гибридный режим — композиция, а не отдельная
подсистема.
"""

from lattice.multiscale.block import BlockEvaluator, BlockResult
from lattice.multiscale.lifetime import (
    LifetimeEngine,
    LifetimeResult,
    LifetimeSample,
    LifetimeSettings,
)

__all__ = [
    "BlockEvaluator",
    "BlockResult",
    "LifetimeEngine",
    "LifetimeResult",
    "LifetimeSample",
    "LifetimeSettings",
]
