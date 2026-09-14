"""Определяющие соотношения — единственный источник истины (контракт C4).

И эталонный МКЭ-путь, и суррогатный путь обязаны звать функции отсюда, а не
заводить собственные копии формул. Расхождение между ними невозможно обнаружить
по картинке, зато оно прекрасно маскируется под «сеть недоучилась».
"""

from lattice.physics.damage import (
    blocks_to_damage,
    damage_driving_integral,
    damage_rate,
    hayhurst_equivalent_stress,
    kr_exact_jump,
    max_principal_plane_stress,
    von_mises_plane_stress,
)
from lattice.physics.elasticity import (
    degradation_multiplier,
    plane_stress_moduli,
    stress_from_strain,
    thermal_strain,
)

__all__ = [
    "blocks_to_damage",
    "damage_driving_integral",
    "damage_rate",
    "degradation_multiplier",
    "hayhurst_equivalent_stress",
    "kr_exact_jump",
    "max_principal_plane_stress",
    "plane_stress_moduli",
    "stress_from_strain",
    "thermal_strain",
    "von_mises_plane_stress",
]
