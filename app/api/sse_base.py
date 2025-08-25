import json
from enum import Enum
from typing import Union

from pydantic import BaseModel


class EventType(Enum):
    LOADING = 'loading'
    GENERATING = 'generating'
    TOOL_CALL = 'tool_call'
    ERROR = 'error'
    SUCCESS = 'success'


class SSEMessage(BaseModel):
    event: EventType
    data: Union[str, list[dict], dict]

    def to_sse(self) -> str:
        """转换为 SSE 格式的消息"""
        return f'event: {self.event.value}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n'
