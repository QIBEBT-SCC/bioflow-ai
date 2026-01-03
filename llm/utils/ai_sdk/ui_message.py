"""
Vercel AI SDK UIMessage 数据类及与 LangChain Message 的互转功能

基于 Vercel AI SDK v6 文档: https://v6.ai-sdk.dev/docs/reference/ai-sdk-core/ui-message
"""

from typing import Any, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

# UIMessagePart 类型定义
# ============================================================================


class TextUIPart(BaseModel):
    """文本消息部分"""

    type: Literal['text'] = 'text'
    text: str
    state: Optional[Literal['streaming', 'done']] = 'done'


class ReasoningUIPart(BaseModel):
    """推理/思考消息部分"""

    type: Literal['reasoning'] = 'reasoning'
    text: str
    state: Optional[Literal['streaming', 'done']] = 'done'
    providerMetadata: Optional[dict[str, Any]] = None


class FileUIPart(BaseModel):
    """文件附件部分

    ## 图片类型
    {
        "type": "file",
        "mediaType": "image/png",
        "filename": "example.png",
        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    }

    ## 文本文件类型
    {
        "type": "file",
        "mediaType": "text/plain",
        "filename": "example.txt",
        "url": "data:text/plain;base64,aHR0cHM6Ly9ub3ZvLWludGVs..."
    }

    ## pdf文件类型
    {
        "type": "file",
        "mediaType": "application/pdf",
        "filename": "example.pdf",
        "url": "data:application/pdf;base64,JVBERi0xLjQKJcfs..."
    }
    """

    type: Literal['file'] = 'file'
    mediaType: str
    filename: Optional[str] = None
    url: str


class SourceUrlUIPart(BaseModel):
    """源URL部分"""

    type: Literal['source-url'] = 'source-url'
    sourceId: str
    url: str
    title: Optional[str] = None
    providerMetadata: Optional[dict[str, Any]] = None


class SourceDocumentUIPart(BaseModel):
    """源文档部分"""

    type: Literal['source-document'] = 'source-document'
    sourceId: str
    mediaType: str
    title: str
    filename: Optional[str] = None
    providerMetadata: Optional[dict[str, Any]] = None


class ToolCallUIPart(BaseModel):
    """工具调用部分"""

    type: str  # 格式为 'tool-{toolName}'
    toolCallId: str
    toolName: str
    input: Optional[dict[str, Any]] = None
    output: Optional[dict[str, Any]] = None
    state: Optional[Literal['pending', 'streaming', 'done', 'failed']] = None


class DataUIPart(BaseModel):
    """自定义数据部分"""

    type: str  # 格式为 'data-{dataType}'
    id: Optional[str] = None
    data: dict[str, Any]


class StepStartUIPart(BaseModel):
    """步骤开始标记"""

    type: Literal['step-start'] = 'step-start'


class StepFinishUIPart(BaseModel):
    """步骤完成标记"""

    type: Literal['step-finish'] = 'step-finish'


UIMessagePart = Union[
    TextUIPart,
    ReasoningUIPart,
    FileUIPart,
    SourceUrlUIPart,
    SourceDocumentUIPart,
    ToolCallUIPart,
    DataUIPart,
    StepStartUIPart,
    StepFinishUIPart,
]


# ============================================================================
# UIMessage 主类定义
# ============================================================================


class UIMessage(BaseModel):
    """
    Vercel AI SDK UIMessage 数据类

    表示应用程序状态的完整消息，包括消息历史、元数据、数据部分和上下文信息
    用于 UI 渲染和客户端功能

    Attributes:
        id: 消息的唯一标识符
        role: 消息发送者的角色 ('system', 'user', 'assistant')
        metadata: 可选的额外元数据
        parts: 消息部分数组，每个部分代表消息内容的一个片段
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal['system', 'user', 'assistant']
    metadata: Optional[dict[str, Any]] = None
    parts: list[UIMessagePart] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
