"""Контракт C1 — единственный источник истины по параметрам задачи.

Всё в СИ, без исключений. Ни один слой не передаёт параметры свободным ``dict``:
из этих моделей генерируется TS-клиент для фронта и хеш, которым помечается датасет.
Расхождение схемы обязано ломать типы на границе, а не всплывать через месяц как
необъяснимые 30% ошибки в инференсе.

Модели заморожены (``frozen=True``) — по контракту C2 состояние и спецификация
не мутируются, интеграторы возвращают новые объекты.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Final даёт mypy вывести Literal["1"], иначе поле схемы принимает любую строку
# и смысл версионирования теряется.
SCHEMA_VERSION: Final = "1"

PositiveFloat = Annotated[float, Field(gt=0.0)]
UnitFraction = Annotated[float, Field(ge=0.0, le=1.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MaterialParams(_Frozen):
    """Термоупругие свойства неповреждённого материала.

    Значения по умолчанию — порядок величин конструкционной стали. Они намеренно
    не «точные марочные»: проект верифицируется, а не валидируется, и подстановка
    паспортных чисел создала бы ложное впечатление, что результат относится
    к конкретному сплаву.
    """

    youngs_modulus: PositiveFloat = Field(default=2.0e11, description="E, Па")
    poisson_ratio: Annotated[float, Field(gt=-1.0, lt=0.5)] = Field(
        default=0.3, description="ν, безразм. Верхняя граница строгая: 0.5 — несжимаемость"
    )
    thermal_expansion: PositiveFloat = Field(default=1.2e-5, description="α, 1/К")
    conductivity: PositiveFloat = Field(default=45.0, description="k₀, Вт/(м·К)")
    density: PositiveFloat = Field(default=7850.0, description="ρ, кг/м³")
    specific_heat: PositiveFloat = Field(default=470.0, description="c_p, Дж/(кг·К)")
    reference_temperature: PositiveFloat = Field(
        default=293.15,
        description="T_ref, К — температура, при которой тепловая деформация нулевая",
    )


class DamageParams(_Frozen):
    """Качанов–Работнов с эквивалентным напряжением Хейхёрста и нелокальным скринингом.

    Закон в обезразмеренной форме::

        ḋ = (1/t_ref) · (χ̄/σ_ref)^r · (1−d)^(−k)

    Параметризация через (t_ref, σ_ref) вместо классической константы ``A``
    выбрана сознательно: у ``A`` единицы Па·с^(1/r), то есть они меняются вместе
    с показателем. Это ровно тот сорт неявной связи, на которой ошибки не видны
    ни в коде, ни в конфиге.
    """

    exponent_r: PositiveFloat = Field(
        default=5.0,
        description=(
            "r — показатель по напряжению. Он же коэффициент усиления ошибки: "
            "δN/N = r·δχ/χ, поэтому 2% ошибки в напряжениях дают 10% по ресурсу при r=5"
        ),
    )
    exponent_k: Annotated[float, Field(ge=0.0)] = Field(
        default=3.0, description="k — показатель по повреждённости в знаменателе"
    )
    reference_stress: PositiveFloat = Field(default=1.0e8, description="σ_ref, Па")
    reference_time: PositiveFloat = Field(default=1.0, description="t_ref, с")
    triaxiality_alpha: UnitFraction = Field(
        default=0.5,
        description=(
            "α по Хейхёрсту: вес максимального главного напряжения против Мизеса. "
            "α=0 — чистый Мизес (повреждает и в сжатии, физически неверно), "
            "α=1 — чистое растяжение первого главного"
        ),
    )
    nonlocal_length: PositiveFloat = Field(
        default=2.0e-3,
        description=(
            "l_c, м — длина скрининга Пирлингса. Не косметика: без регуляризации "
            "повреждение схлопывается в одноэлементную полосу и ресурс становится "
            "функцией шага сетки"
        ),
    )
    max_damage: Annotated[float, Field(gt=0.0, lt=1.0)] = Field(
        default=0.99,
        description="Потолок локального поля — при d→1 жёсткость вырождается и K необратима",
    )
    residual_stiffness: Annotated[float, Field(gt=0.0, lt=1.0)] = Field(
        default=1.0e-6,
        description="Нижняя полка множителя (1−d): удерживает K положительно определённой",
    )

    @model_validator(mode="after")
    def _check_stiffness_floor_consistent(self) -> DamageParams:
        if self.residual_stiffness >= 1.0 - self.max_damage:
            # Иначе полка срабатывает раньше потолка, и потолок ничего не ограничивает.
            msg = (
                f"residual_stiffness={self.residual_stiffness} должна быть меньше "
                f"1−max_damage={1.0 - self.max_damage}"
            )
            raise ValueError(msg)
        return self


class Geometry(_Frozen):
    """Прямоугольная пластина с центральным отверстием, плоское напряжённое состояние.

    Геометрия входит в расчёт маской материала на структурной фоновой сетке, а не
    согласованной сеткой. Это убирает шов «меш ↔ решётка» и делает радиус отверстия
    параметром DoE без перестроения сетки. Цена — ступенчатая граница у концентратора;
    она замеряется против решения Кирша в наборе верификации.
    """

    width: PositiveFloat = Field(default=0.1, description="м")
    height: PositiveFloat = Field(default=0.1, description="м")
    thickness: PositiveFloat = Field(default=1.0e-3, description="м")
    hole_radius: PositiveFloat = Field(default=0.01, description="м")

    @model_validator(mode="after")
    def _check_hole_fits(self) -> Geometry:
        half_min = 0.5 * min(self.width, self.height)
        if self.hole_radius >= 0.9 * half_min:
            msg = (
                f"hole_radius={self.hole_radius} не оставляет перемычки: "
                f"должен быть меньше 0.9·{half_min}"
            )
            raise ValueError(msg)
        return self


class LoadProgram(_Frozen):
    """Периодический термомеханический блок.

    Один «цикл» — это период программы, а не отдельная сущность. Различие принципиально:
    Качанов–Работнов есть закон по времени, повреждение копится в секундах, а прыжки
    делаются по блокам. Смешение этих двух смыслов обнуляет метрику ресурса, при этом
    выглядя совершенно правдоподобно.
    """

    period: PositiveFloat = Field(
        default=200.0,
        description=(
            "t_c, с — длительность блока. Значение по умолчанию выбрано не круглым "
            "числом, а по числу Фурье: температуропроводность стали α ≈ 1.2·10⁻⁵ м²/с, "
            "полутолщина пластины 0.05 м, время выравнивания L²/α ≈ 205 с. При заметно "
            "большем периоде тело прогревается равномерно, а равномерно расширяющееся "
            "тело напряжений не несёт — термомеханическая связь вырождается, и связка "
            "фактически становится двухпроцессной"
        ),
    )
    far_field_stress: PositiveFloat = Field(
        default=8.0e7, description="σ_∞, Па — амплитуда растяжения на дальней границе"
    )
    temperature_hot: PositiveFloat = Field(default=873.15, description="К")
    temperature_cold: PositiveFloat = Field(default=293.15, description="К")
    hold_fraction: UnitFraction = Field(
        default=0.6, description="Доля периода на выдержке при T_hot под нагрузкой"
    )

    @model_validator(mode="after")
    def _check_thermal_range(self) -> LoadProgram:
        if self.temperature_hot <= self.temperature_cold:
            msg = (
                f"temperature_hot={self.temperature_hot} должна превышать "
                f"temperature_cold={self.temperature_cold}"
            )
            raise ValueError(msg)
        return self


class MeshSpec(_Frozen):
    """Структурная фоновая сетка. Квадратная — вход CNN нативен без ресемплинга."""

    resolution: Annotated[int, Field(ge=16, le=1024)] = Field(
        default=128, description="Узлов по стороне"
    )
    interface_fraction: Annotated[float, Field(ge=0.0, le=0.5)] = Field(
        default=0.1,
        description=(
            "Ширина переходной полосы маски в долях радиуса отверстия — в физических "
            "единицах, не в ячейках. Привязка к ячейке делала бы геометрию зависящей "
            "от сетки, и сеточная сходимость стала бы недостижима в принципе"
        ),
    )


class CaseSpec(_Frozen):
    """Полная постановка. Хеш этого объекта участвует в ``dataset_id``."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    material: MaterialParams = Field(default_factory=MaterialParams)
    damage: DamageParams = Field(default_factory=DamageParams)
    geometry: Geometry = Field(default_factory=Geometry)
    load: LoadProgram = Field(default_factory=LoadProgram)
    mesh: MeshSpec = Field(default_factory=MeshSpec)

    def content_hash(self) -> str:
        """Канонический хеш содержимого.

        Устойчив к порядку ключей и к незначащему форматированию float, поэтому
        два одинаковых по смыслу конфига дают один ``dataset_id``. Любое число
        в отчёте прослеживается до кода и данных через эту цепочку.
        """
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
