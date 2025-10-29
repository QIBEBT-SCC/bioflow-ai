from enum import Enum
from typing import Any, Literal, Optional, Union

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


class UIState(str, Enum):
    STREAMING = 'streaming'
    DONE = 'done'


class TextUIPart(BaseModel):
    type: Literal['text'] = 'text'
    text: str
    state: Optional[UIState] = None
    reasoning: Optional[str] = None


class FileUIPart(BaseModel):
    type: Literal['file'] = 'file'
    mediaType: str
    filename: Optional[str] = None
    url: str
    size: Optional[int] = None


class ToolCallUIPart(BaseModel):
    type: Literal['tool-call'] = 'tool-call'
    toolCallId: str
    toolName: str
    args: dict[str, Any]
    result: Optional[Any] = None
    state: Optional[UIState] = None


class ReasoningUIPart(BaseModel):
    type: Literal['reasoning'] = 'reasoning'
    text: str
    state: Optional[UIState] = None


class SourceUIPart(BaseModel):
    type: Literal['source'] = 'source'
    url: str
    title: Optional[str] = None
    description: Optional[str] = None


class CustomUIPart(BaseModel):
    type: str  # 自定义类型，不限制为特定值
    data: dict[str, Any]


# UI part联合类型
UIPart = Union[TextUIPart, FileUIPart, ToolCallUIPart, ReasoningUIPart, SourceUIPart, CustomUIPart]


class UIMessage(BaseModel):
    id: str
    role: Literal['system', 'user', 'assistant', 'tool']
    metadata: Optional[dict[str, Any]] = None
    parts: list[UIPart] = Field(default_factory=list)
    createdAt: Optional[str] = None
    modelId: Optional[str] = None


def convert_to_langchain_message(ui_message: dict[str, Any]) -> HumanMessage:
    """将UI消息转换为LangChain HumanMessage格式

    Args:
        ui_message: UI消息字典

    Returns:
        HumanMessage: 转换后的LangChain消息

    Raises:
        AssertionError: 如果消息角色不是'user'
    """
    ui_message = UIMessage.model_validate(ui_message)
    assert ui_message.role == 'user', f'Expected user role, got {ui_message.role}'

    content_parts = []

    for part in ui_message.parts:
        if isinstance(part, TextUIPart):
            content_parts.append({'type': 'text', 'text': part.text})
        elif isinstance(part, FileUIPart):
            content_parts.append(
                {
                    'type': 'image_url' if part.mediaType.startswith('image/') else 'file_url',
                    'image_url' if part.mediaType.startswith('image/') else 'file_url': {
                        'url': part.url,
                        'media_type': part.mediaType,
                        'filename': part.filename,
                        'size': part.size,
                    },
                }
            )

    return HumanMessage(content=content_parts)


def truncate_string(text, max_length=50):
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text
