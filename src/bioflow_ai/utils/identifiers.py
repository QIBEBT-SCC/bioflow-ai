import uuid
from typing import TypeAlias

UUIDLike: TypeAlias = uuid.UUID | str


def as_uuid(value: UUIDLike) -> uuid.UUID:
    """Normalize a UUID object or string into a UUID object."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
