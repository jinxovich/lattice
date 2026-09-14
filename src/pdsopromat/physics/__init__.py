"""Определяющие соотношения — единственный источник истины (контракт C4).

И эталонный МКЭ-путь, и суррогатный путь обязаны звать функции отсюда, а не
заводить собственные копии формул. Расхождение между ними невозможно обнаружить
по картинке, зато оно прекрасно маскируется под «сеть недоучилась».
"""

from pdsopromat.physics.damage import (
    blocks_to_damage,
    damage_driving_integral,
    damage_rate,
    hayhurst_equivalent_stress,
    kr_exact_jump,
    max_principal_plane_stress,
    von_mises_plane_stress,
)

__all__ = [
    "blocks_to_damage",
    "damage_driving_integral",
    "damage_rate",
    "hayhurst_equivalent_stress",
    "kr_exact_jump",
    "max_principal_plane_stress",
    "von_mises_plane_stress",
]
