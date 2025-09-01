from pydantic import BaseModel, Field


class BidirectionalMapper(BaseModel):
    uid_to_idx: dict[str, int] = Field(default_factory=dict)
    idx_to_uid: dict[int, str] = Field(default_factory=dict)

    def __init__(self, datas: list[BaseModel] = None):
        super().__init__()
        if datas is not None:
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
