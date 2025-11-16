from typing import Optional

from pydantic import BaseModel, Field


class BidirectionalMapper(BaseModel):
    uid_to_idx: dict[str, int] = Field(default_factory=dict)
    idx_to_uid: dict[int, str] = Field(default_factory=dict)

    def __init__(self, datas: list[BaseModel] = None, use_metadata: bool = False, metadata_field: Optional[str] = None):
        super().__init__()

        assert datas is not None

        if use_metadata:
            assert metadata_field is not None
            self.uid_to_idx = {data.metadata[metadata_field]: idx for idx, data in enumerate(datas)}
            self.idx_to_uid = {idx: data.metadata[metadata_field] for idx, data in enumerate(datas)}
        else:
            self.uid_to_idx = {str(data.uid): idx for idx, data in enumerate(datas)}
            self.idx_to_uid = {idx: str(data.uid) for idx, data in enumerate(datas)}

    def get_index_by_uid(self, uid: str) -> int:
        return self.uid_to_idx.get(uid)

    def get_uid_by_index(self, index: int) -> str:
        return self.idx_to_uid.get(index)

    def has_uid(self, uid: str) -> bool:
        return uid in self.uid_to_idx

    def has_index(self, index: int) -> bool:
        return index in self.idx_to_uid


def generate_char_index(n: int) -> str:
    """将一个正整数转换为（A, B, ..., Z, AA, AB, ...）形式的索引"""
    assert n >= 1, '输入必须是一个大于等于1的正整数'

    result = []
    while n > 0:
        remainder = (n - 1) % 26
        result.append(chr(65 + remainder))
        n = (n - 1) // 26
    return ''.join(reversed(result))
