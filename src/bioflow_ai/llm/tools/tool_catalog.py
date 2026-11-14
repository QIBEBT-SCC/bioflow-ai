"""Catalog search shared by the workflow builder and the tool generator."""

import uuid

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

import bioflow_ai.crud.tools.tool as tool_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.schemas.tools import ToolNodePublic

SEARCH_RESULT_LIMIT = 15


class SearchToolSchema(BaseModel):
    """Search tool nodes by their complete command, like chaining `rg -i` over every node's command line."""

    patterns: list[str] = Field(
        min_length=1,
        description=(
            'Case-insensitive regular expressions matched against the complete command of each tool node; '
            'a node is returned only when every pattern matches. Use one pattern per command fragment so that '
            'option order does not matter, e.g. `["samtools sort"]`, `["gatk", "HaplotypeCaller"]` or '
            '`["emapper", "-m cache"]`. Start with the program and subcommand, then add mode-defining options '
            'to narrow down the results.'
        ),
    )

    @field_validator('patterns')
    @classmethod
    def drop_blank_patterns(cls, patterns: list[str]) -> list[str]:
        """Remove blank patterns and require at least one meaningful pattern."""
        patterns = [pattern.strip() for pattern in patterns if pattern.strip()]
        if not patterns:
            message = 'At least one non-blank pattern is required'
            raise ValueError(message)
        return patterns


class InvalidToolDefinitionError(RuntimeError):
    """A catalog result cannot be represented as a Tool node."""


def search_catalog_nodes(patterns: list[str]) -> tuple[list[ToolNodePublic], list[uuid.UUID]]:
    """Load the displayed page and every matching tool UID."""
    with Session(engine) as session:
        tools, matched_uids = tool_crud.search_by_command(session, patterns, SEARCH_RESULT_LIMIT)
        nodes = []
        for item in tools:
            try:
                nodes.append(ToolNodePublic.model_validate(item))
            except Exception as error:
                message = f'Invalid tool definition "{item.name}" ({item.uid}): {error}'
                raise InvalidToolDefinitionError(message) from error
    return nodes, matched_uids
