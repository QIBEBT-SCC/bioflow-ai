import uuid
from typing import Optional, Protocol, Self, runtime_checkable

from langchain_core.documents import Document
from pydantic import BaseModel, Field


@runtime_checkable
class HasUid(Protocol):
    uid: uuid.UUID


class BidirectionalMapper(BaseModel):
    uid_to_idx: dict[str, int] = Field(default_factory=dict)
    idx_to_uid: dict[int, str] = Field(default_factory=dict)

    def __init__(
        self,
        datas: list[HasUid | Document] | None = None,
        *,
        use_metadata: bool = False,
        metadata_field: Optional[str] = None,
    ):
        super().__init__()

        if not datas:
            return

        if use_metadata:
            if metadata_field is None:
                err_msg = 'When use_metadata=True is set, the metadata_field parameter must be specified.'
                raise ValueError(err_msg)
            self.uid_to_idx = {data.metadata[metadata_field]: idx for idx, data in enumerate(datas)}
            self.idx_to_uid = {idx: data.metadata[metadata_field] for idx, data in enumerate(datas)}
        else:
            self.uid_to_idx = {str(data.uid): idx for idx, data in enumerate(datas)}
            self.idx_to_uid = {idx: str(data.uid) for idx, data in enumerate(datas)}

    @classmethod
    def from_dicts(cls, uid_to_idx: dict[str, int], idx_to_uid: dict[int, str]) -> Self:
        """Return the mapping object from existing data."""
        instance = cls()
        instance.uid_to_idx = uid_to_idx
        instance.idx_to_uid = idx_to_uid
        return instance

    def get_index_by_uid(self, uid: str) -> int:
        """Get the index of a UID."""
        return self.uid_to_idx.get(uid)

    def get_uid_by_index(self, index: int) -> str:
        """Get the UID of an index."""
        return self.idx_to_uid.get(index)


def merge_mappers(existing: BidirectionalMapper, new: BidirectionalMapper) -> BidirectionalMapper:
    """LangGraph reducer: merges new search results into the existing mapper.

    New UIDs get indices continuing from the current max. Duplicate UIDs keep their existing index.
    """
    if existing is None:
        return new

    uid_to_idx = dict(existing.uid_to_idx)
    idx_to_uid = dict(existing.idx_to_uid)

    next_index = max(idx_to_uid.keys()) + 1 if idx_to_uid else 0

    for uid in new.uid_to_idx:
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
