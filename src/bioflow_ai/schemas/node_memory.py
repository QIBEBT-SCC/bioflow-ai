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
        description=(
            'Single-line summary (<=120 chars) of the configuration topic or usage scenario covered so '
            'future workflow-building agents can judge relevance'
        ),
    ),
]
MemoryContent = Annotated[
    str,
    Field(
        min_length=1,
        max_length=8000,
        description=(
            'Reusable Markdown usage guidance for future workflows: explain the underlying tool mechanism '
            'or constraint first, then give concrete configuration/usage recommendations and key trade-offs; '
            'avoid single-run incident logs or unverified test claims'
        ),
    ),
]


class ToolMemoryFields(BaseModel):
    """Editable content of a shared tool memory."""

    model_config = ConfigDict(str_strip_whitespace=True)

    description: MemoryDescription
    content: MemoryContent


class NodeMemoryDraft(ToolMemoryFields):
    """One proposed memory targeting a Tool node in the selected workflow run."""

    node_id: str = Field(min_length=1, description='Local Tool node ID from workflow_reader in the selected run')
    subgraph_path: list[str] | None = Field(
        default=None,
        description='Nested ordinary subgraph node IDs when the target Tool node is inside a subgraph',
    )
    child_run_uid: str | None = Field(
        default=None,
        description='ForEach child run UID from workflow_reader(action="children") when targeting a batch child run',
    )


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
