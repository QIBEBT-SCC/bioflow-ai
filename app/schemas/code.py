import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CodeNodeType(StrEnum):
    """Workflow CODE node types that can be persisted for reuse."""

    BASH = 'code_bash'
    PYTHON = 'code_python'


class CodeData(BaseModel):
    """Shared persisted properties of a reusable CODE definition."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    node_type: CodeNodeType
    code: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)


class CodeCreate(CodeData):
    """Properties accepted when creating a reusable CODE definition."""


class CodeUpdate(BaseModel):
    """Editable reusable CODE definition properties."""

    name: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    node_type: CodeNodeType | None = None
    code: str | None = Field(default=None, min_length=1)
    dependencies: list[str] | None = None


class SimpleCodePublic(BaseModel):
    """A compact reusable CODE definition for list and search results."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str
    description: str
    node_type: CodeNodeType


class CodePublic(CodeData):
    """A complete reusable CODE definition."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
