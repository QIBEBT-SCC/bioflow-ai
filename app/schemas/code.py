import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

BASH_DEPENDENCIES_ERROR = 'Bash CODE definitions cannot declare Python dependencies'


class CodeNodeType(StrEnum):
    """Workflow CODE node types that can be persisted for reuse."""

    BASH = 'code_bash'
    PYTHON = 'code_python'


class CodeData(BaseModel):
    """Shared persisted properties of a reusable CODE definition."""

    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1)
    node_type: CodeNodeType
    code: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_dependencies(self) -> 'CodeData':
        """Allow runtime dependencies only for Python CODE definitions."""
        if self.node_type is CodeNodeType.BASH and self.dependencies:
            raise ValueError(BASH_DEPENDENCIES_ERROR)
        return self


class CodeCreate(CodeData):
    """Properties accepted when creating a reusable CODE definition."""


class CodeMetadataRequest(BaseModel):
    """Source code submitted for AI-generated catalog metadata."""

    model_config = ConfigDict(extra='forbid')

    node_type: CodeNodeType
    code: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_dependencies(self) -> 'CodeMetadataRequest':
        """Allow runtime dependencies only for Python CODE definitions."""
        if self.node_type is CodeNodeType.BASH and self.dependencies:
            raise ValueError(BASH_DEPENDENCIES_ERROR)
        return self


class CodeMetadataPublic(BaseModel):
    """AI-generated human-facing catalog metadata."""

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1)


class CodeUpdate(BaseModel):
    """Editable reusable CODE definition properties."""

    model_config = ConfigDict(extra='forbid')

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, min_length=1)
    code: str | None = Field(default=None, min_length=1)
    dependencies: list[str] | None = None

    @model_validator(mode='after')
    def reject_explicit_nulls(self) -> 'CodeUpdate':
        """Treat omitted fields as unchanged and reject explicit null values."""
        null_fields = sorted(field for field in self.model_fields_set if getattr(self, field) is None)
        if null_fields:
            message = f'CODE update fields cannot be null: {", ".join(null_fields)}'
            raise ValueError(message)
        return self


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
