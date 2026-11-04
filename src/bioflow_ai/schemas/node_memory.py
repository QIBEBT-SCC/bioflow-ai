import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

MemoryDescription = Annotated[
    str,
    Field(
        min_length=1,
        max_length=120,
        pattern=r'^[^\r\n]+$',
        description='One or two short sentences describing when this memory applies; at most 120 characters',
    ),
]
MemoryContent = Annotated[
    str,
    Field(
        min_length=1,
        max_length=8000,
        description='Reusable Markdown lesson with its applicability, evidence, fix, and actual validation limits',
    ),
]


class ToolMemoryFields(BaseModel):
    """Editable content of a shared tool memory."""

    model_config = ConfigDict(str_strip_whitespace=True)

    description: MemoryDescription
    content: MemoryContent


class NodeMemoryDraft(ToolMemoryFields):
    """One proposed memory targeting a tool node on the active workflow panel."""

    node_id: str = Field(min_length=1, description='Tool node ID on the current panel, not a search result ID')


class ToolMemoryCreate(ToolMemoryFields):
    """Create a memory for an existing tool."""

    tool_uid: uuid.UUID


class ToolMemoryImport(BaseModel):
    """Persisted memory fields restored from the preset dataset."""

    uid: uuid.UUID
    tool_uid: uuid.UUID
    description: str = Field(max_length=120)
    content: str
    created_by: int
    created_at: datetime


class ToolMemoryUpdate(BaseModel):
    """Edit a memory's description or body without changing its tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    description: MemoryDescription | None = None
    content: MemoryContent | None = None

    @field_validator('description', 'content')
    @classmethod
    def reject_null(cls, value: str | None) -> str:
        """Allow omitted fields, but reject explicit null values."""
        if value is None:
            msg = 'Omit the field to leave it unchanged.'
            raise ValueError(msg)
        return value


class ToolMemorySummary(BaseModel):
    """Compact memory reference advertised when a tool node is added or inspected."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    description: str


class ToolMemoryListItem(ToolMemorySummary):
    """A memory summary that identifies its associated tool and creator."""

    tool_uid: uuid.UUID
    tool_name: str
    creator_username: str | None
    created_at: datetime


class ToolMemoryPublic(ToolMemoryListItem):
    """Full memory content returned by a detail read or a write."""

    content: str
