from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


def _default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"Non-finite float value {value!r} is not valid canonical JSON"
        )
    if isinstance(value, BaseModel):
        _reject_non_finite(value.model_dump(mode="json"))
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(key)
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def canonical_json(value: Any) -> str:
    _reject_non_finite(value)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_default,
        allow_nan=False,
    )


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
