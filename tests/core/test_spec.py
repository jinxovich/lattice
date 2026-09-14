"""Контракт C1: спецификация неизменяема, валидируется на границе и хешируется канонически."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pdsopromat.core import (
    CaseSpec,
    DamageParams,
    Geometry,
    LoadProgram,
    MaterialParams,
)


class TestImmutability:
    def test_spec_rejects_mutation(self) -> None:
        spec = CaseSpec()

        with pytest.raises(ValidationError):
            spec.material = MaterialParams()  # type: ignore[misc]

    def test_nested_model_rejects_mutation(self) -> None:
        spec = CaseSpec()

        with pytest.raises(ValidationError):
            spec.damage.exponent_r = 8.0  # type: ignore[misc]

    def test_unknown_field_is_rejected(self) -> None:
        # Опечатка в конфиге должна падать на границе, а не молча игнорироваться
        # и оставлять параметр на значении по умолчанию.
        with pytest.raises(ValidationError):
            MaterialParams(yongs_modulus=2.0e11)  # type: ignore[call-arg]


class TestPhysicalValidation:
    def test_incompressible_poisson_ratio_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MaterialParams(poisson_ratio=0.5)

    def test_negative_modulus_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MaterialParams(youngs_modulus=-1.0)

    def test_hole_larger_than_ligament_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="перемычки"):
            Geometry(width=0.1, height=0.1, hole_radius=0.049)

    def test_hole_leaving_ligament_is_accepted(self) -> None:
        assert Geometry(width=0.1, height=0.1, hole_radius=0.02).hole_radius == 0.02

    def test_inverted_thermal_cycle_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="temperature_hot"):
            LoadProgram(temperature_hot=300.0, temperature_cold=400.0)

    def test_stiffness_floor_above_damage_cap_is_rejected(self) -> None:
        # Полка сработала бы раньше потолка, и max_damage перестал бы что-либо значить.
        with pytest.raises(ValidationError, match="residual_stiffness"):
            DamageParams(max_damage=0.99, residual_stiffness=0.1)

    def test_triaxiality_alpha_outside_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DamageParams(triaxiality_alpha=1.5)


class TestContentHash:
    def test_equal_specs_hash_identically(self) -> None:
        # Два независимо собранных одинаковых конфига обязаны дать один dataset_id,
        # иначе кеш датасета промахивается на каждом запуске.
        assert CaseSpec().content_hash() == CaseSpec().content_hash()

    def test_hash_is_insensitive_to_field_order(self) -> None:
        a = CaseSpec(material=MaterialParams(), damage=DamageParams())
        b = CaseSpec(damage=DamageParams(), material=MaterialParams())

        assert a.content_hash() == b.content_hash()

    def test_hash_tracks_a_changed_parameter(self) -> None:
        baseline = CaseSpec()
        perturbed = CaseSpec(damage=DamageParams(exponent_r=5.001))

        assert baseline.content_hash() != perturbed.content_hash()

    def test_hash_is_short_and_hex(self) -> None:
        digest = CaseSpec().content_hash()

        assert len(digest) == 16
        assert all(c in "0123456789abcdef" for c in digest)

    def test_schema_version_participates_in_hash(self) -> None:
        # Смена версии схемы обязана инвалидировать датасет.
        digest = CaseSpec().content_hash()
        assert CaseSpec().schema_version == "1"
        assert digest == CaseSpec(schema_version="1").content_hash()
