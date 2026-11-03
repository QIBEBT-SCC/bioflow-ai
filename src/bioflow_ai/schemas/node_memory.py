import uuid

from pydantic import BaseModel, ConfigDict, Field


class NodeMemoryDraft(BaseModel):
    """One proposed memory targeting a tool node on the active workflow panel."""

    model_config = ConfigDict(str_strip_whitespace=True)

    node_id: str = Field(min_length=1, description='Tool node ID on the current panel, not a search result ID')
    description: str = Field(
        min_length=1,
        max_length=120,
        pattern=r'^[^\r\n]+$',
        description='One or two short sentences describing when this memory applies; at most 120 characters',
    )
    content: str = Field(
        min_length=1,
        max_length=8000,
        description='Reusable Markdown lesson with its applicability, evidence, fix, and actual validation limits',
    )


class ToolMemorySummary(BaseModel):
    """Compact memory reference advertised when a tool node is added or inspected."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    description: str


class ToolMemoryPublic(ToolMemorySummary):
    """Full memory content returned only when the agent explicitly reads it."""

    tool_uid: uuid.UUID
    content: str
