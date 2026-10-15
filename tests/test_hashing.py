from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import BaseModel

from nerlex.hashing import canonical_json, sha256_hex


def test_canonical_hash_is_key_order_independent() -> None:
    left = {"b": 2, "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": 2}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_hex(left) == sha256_hex(right)


def test_canonical_json_preserves_unicode_characters() -> None:
    value = {"message": "Türkçe ğüşiöç İI"}

    encoded = canonical_json(value)

    assert encoded == '{"message":"Türkçe ğüşiöç İI"}'
    assert "ğüşiöç" in encoded
    assert "\\u" not in encoded


def test_unicode_hash_is_deterministic() -> None:
    left = {"message": "Türkçe ğüşiöç İI"}
    right = {"message": "Türkçe ğüşiöç İI"}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_hex(left) == sha256_hex(right)
    assert sha256_hex(left) == "bd4feaa85eaf5513d976956708e78ae135f83bf503902cc36327306dabe836a5"


def test_canonical_json_serializes_date_and_aware_datetime() -> None:
    value = {
        "day": date(2024, 1, 2),
        "utc": datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        "offset": datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=3))),
    }

    assert canonical_json(value) == (
        '{"day":"2024-01-02",'
        '"offset":"2024-01-02T03:04:05+03:00",'
        '"utc":"2024-01-02T03:04:05+00:00"}'
    )


def test_canonical_json_serializes_uuid() -> None:
    value = UUID("12345678-1234-5678-1234-567812345678")

    assert canonical_json(value) == '"12345678-1234-5678-1234-567812345678"'
    assert sha256_hex(value) == "be6bf1ea57078ff17fa690d9c62d24fb803c8aac2368bb9088431187034a2cfe"


def test_extension_types_hash_is_deterministic() -> None:
    def build() -> dict[str, object]:
        return {
            "at": datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
            "day": date(2024, 1, 2),
            "id": UUID("12345678-1234-5678-1234-567812345678"),
            "text": "Türkçe",
        }

    first = build()
    second = build()
    assert first is not second
    assert canonical_json(first) == canonical_json(second)
    assert sha256_hex(first) == "5a6d8d2035c5a9bfcdcaef5309d2adf102c98bbccc7da0b643badea00a709f9a"


@pytest.mark.parametrize(
    "value",
    [object(), {1, 2}, b"bytes"],
    ids=["object", "set", "bytes"],
)
def test_canonical_json_rejects_unsupported_types(value: object) -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        canonical_json(value)


def test_canonical_json_rejects_nested_unsupported_values() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        canonical_json({"supported": 1, "unsupported": {1, 2}})


def test_sha256_hex_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        sha256_hex({"unsupported": object()})


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinity", "-infinity"],
)
def test_canonical_json_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite float"):
        canonical_json({"score": value})


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinity", "-infinity"],
)
def test_sha256_hex_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite float"):
        sha256_hex({"score": value})


def test_canonical_json_rejects_nested_non_finite_values() -> None:
    with pytest.raises(ValueError, match="Non-finite float"):
        canonical_json({"outer": [1.0, {"score": float("inf")}]})

    with pytest.raises(ValueError, match="Non-finite float"):
        canonical_json((1.5, float("-inf")))


def test_canonical_json_rejects_non_finite_values_in_model_data() -> None:
    class ScoreModel(BaseModel):
        score: float

    with pytest.raises(ValueError, match="Non-finite float"):
        canonical_json(ScoreModel(score=float("nan")))

    with pytest.raises(ValueError, match="Non-finite float"):
        sha256_hex(ScoreModel(score=float("-inf")))


def test_canonical_json_preserves_finite_floats() -> None:
    value = {"score": 0.5, "counts": [1, 2.25, -3.5], "nested": {"tiny": 1e-09}}

    assert canonical_json(value) == (
        '{"counts":[1,2.25,-3.5],"nested":{"tiny":1e-09},"score":0.5}'
    )
