import json
from enum import Enum
from typing import Union
import uuid

from pydantic import BaseModel


class EventType(Enum):
    LOADING = 'loading'
    GENERATING = 'generating'
    TOOL_CALL = 'tool_call'
    ERROR = 'error'
    SUCCESS = 'success'
    INTERRUPT = 'interrupt'


class UUIDEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 UUID 对象"""
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


class SSEMessage(BaseModel):
    event: EventType
    data: Union[str, list[dict], dict]

    def to_sse(self) -> str:
        """转换为 SSE 格式的消息"""
        return f'event: {self.event.value}\ndata: {json.dumps(self.data, ensure_ascii=False, cls=UUIDEncoder)}\n\n'
