"""Workflow documents and the single validation result contract."""

import uuid
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from bioflow_ai.schemas.tools import WorkflowToolDefinition

ExecutionScopeName = Literal['sample_level', 'project_level']
SubgraphPurpose = Literal['ordinary', 'foreach']


class SourceLocation(BaseModel):
    path: str
    node_id: str | None = None
    edge_id: str | None = None
    subgraph_path: list[str] = Field(default_factory=list)


class Diagnostic(BaseModel):
    code: str
    message: str
    path: str = ''
    node_id: str | None = None
    edge_id: str | None = None
    handle: str | None = None
    subgraph_path: list[str] = Field(default_factory=list)


class ValidationContext(BaseModel):
    """Domain facts, independent of the operation invoking validation."""

    execution_scope: ExecutionScopeName = 'sample_level'
    subgraph_purpose: SubgraphPurpose = 'ordinary'
    mark_conflicts: dict[str, str] = Field(default_factory=dict)
    dynamic_marks: set[str] = Field(default_factory=set)


class WorkflowDocument(BaseModel):
    """A graph plus resolved definitions and facts needed to understand it."""

    model_config = ConfigDict(frozen=True)

    graph: Any
    context: ValidationContext = Field(default_factory=ValidationContext)
    catalog: Mapping[str, WorkflowToolDefinition] | None = None


class ValidationResult(BaseModel):
    """Every reported issue blocks validation; there are no validation modes."""

    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @computed_field
    @property
    def valid(self) -> bool:
        """Pass only when every rule passes."""
        return not self.diagnostics

    @property
    def message(self) -> str:
        """Render the same diagnostics for exceptions and agent messages."""
        return '\n'.join(f'{item.code} at {item.path or "/"}: {item.message}' for item in self.diagnostics)


class WorkflowValidationRequest(BaseModel):
    """The HTTP adapter submits exactly the same document as other callers."""

    model_config = ConfigDict(extra='forbid')

    graph: dict[str, Any]
    execution_scope: ExecutionScopeName = 'sample_level'
    project_id: int | None = None
    workflow_uid: uuid.UUID | None = None
