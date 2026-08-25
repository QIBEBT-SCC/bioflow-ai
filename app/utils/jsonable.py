import json
import math
import uuid
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SecretStr

_SENSITIVE_KEYS = {
    'access_key',
    'api_key',
    'authorization',
    'cookie',
    'password',
    'refresh_token',
    'secret',
    'secret_key',
    'token',
}


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower().replace('-', '_')
    return normalized in _SENSITIVE_KEYS or normalized.endswith(('_api_key', '_password', '_secret', '_token'))


def to_jsonable(value: Any) -> Any:
    """Recursively convert an event value into safe JSON primitives."""
    if isinstance(value, SecretStr):
        return '[REDACTED]'
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode='json'))
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): '[REDACTED]' if _is_sensitive_key(key) else to_jsonable(item) for key, item in value.items()}
    if isinstance(value, AbstractSet):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Decimal, Path, uuid.UUID)):
        return str(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, bytes):
        return {'encoding': 'bytes', 'size': len(value)}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def to_json_object(value: object) -> dict[str, Any]:
    """Convert a value to a JSON object suitable for a PostgreSQL JSON column."""
    converted = to_jsonable(value)
    if isinstance(converted, dict):
        return converted
    return {'value': converted}


def dumps_jsonable(value: Any, *, indent: int | None = None, sort_keys: bool = False) -> str:
    """Serialize a value after converting it to safe JSON primitives."""
    return json.dumps(to_jsonable(value), ensure_ascii=False, indent=indent, sort_keys=sort_keys)
