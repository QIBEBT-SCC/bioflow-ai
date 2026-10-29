"""Bidirectional UID/index mappings and alphabetic identifiers."""

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self

from pydantic import BaseModel, Field


class HasUid(Protocol):
    uid: uuid.UUID


class BidirectionalMapper(BaseModel):
    uid_to_idx: dict[str, int] = Field(default_factory=dict)
    idx_to_uid: dict[int, str] = Field(default_factory=dict)

    def __init__(
        self,
        datas: Sequence[HasUid] | None = None,
        *,
        uid_to_idx: dict[str, int] | None = None,
        idx_to_uid: dict[int, str] | None = None,
    ) -> None:
        super().__init__(
            uid_to_idx=uid_to_idx or {},
            idx_to_uid=idx_to_uid or {},
        )

        if not datas:
            return

        self.uid_to_idx = {str(getattr(data, 'uid', '')): idx for idx, data in enumerate(datas)}
        self.idx_to_uid = {idx: str(getattr(data, 'uid', '')) for idx, data in enumerate(datas)}

    @classmethod
    def from_dicts(cls, uid_to_idx: dict[str, int], idx_to_uid: dict[int, str]) -> Self:
        """Return the mapping object from existing data."""
        return cls(uid_to_idx=uid_to_idx, idx_to_uid=idx_to_uid)

    def get_index_by_uid(self, uid: str) -> int:
        """Get the index of a UID."""
        return self.uid_to_idx.get(uid, -1)

    def get_uid_by_index(self, index: int) -> str:
        """Get the UID of an index."""
        return self.idx_to_uid.get(index, '')


MapperState = BidirectionalMapper | Mapping[str, Any] | None


def restore_mapper(value: MapperState) -> BidirectionalMapper:
    """Restore a mapper serialized as a plain checkpoint dictionary."""
    if value is None:
        return BidirectionalMapper()
    if isinstance(value, BidirectionalMapper):
        return value
    return BidirectionalMapper.model_validate(dict(value))


def merge_mappers(existing: MapperState, new: MapperState) -> BidirectionalMapper:
    """Merge new UID mappings into the existing mapper.

    New UIDs get indices continuing from the current max. Duplicate UIDs keep their existing index.
    """
    if existing is None:
        return restore_mapper(new)

    existing_mapper = restore_mapper(existing)
    new_mapper = restore_mapper(new)

    uid_to_idx = dict(existing_mapper.uid_to_idx)
    idx_to_uid = dict(existing_mapper.idx_to_uid)

    next_index = max(idx_to_uid.keys()) + 1 if idx_to_uid else 0

    for uid in new_mapper.uid_to_idx:
        if uid not in uid_to_idx:
            uid_to_idx[uid] = next_index
            idx_to_uid[next_index] = uid
            next_index += 1

    return BidirectionalMapper.from_dicts(uid_to_idx, idx_to_uid)


def generate_char_index(n: int) -> str:
    """Convert a positive integer to (A, B, ..., Z, AA, AB, ...) format."""
    if n < 1:
        err_msg = 'Input must be a positive integer greater than or equal to 1'
        raise ValueError(err_msg)

    result = []
    while n > 0:
        remainder = (n - 1) % 26
        result.append(chr(65 + remainder))
        n = (n - 1) // 26
    return ''.join(reversed(result))
