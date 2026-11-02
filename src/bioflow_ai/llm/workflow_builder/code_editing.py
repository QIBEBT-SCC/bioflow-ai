"""Read workflow-local CODE source and delegate updates to the shared coding agent."""

import ast
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.schemas.workflow_build import BuildEdge, BuildNode
from bioflow_ai.services.code.coding import build_code_inputs, message_text, run_coding_turn


class CodeEditingError(ValueError):
    """A requested panel CODE-node read or update is invalid."""


class CodeNodeUpdate(BaseModel):
    """A complete coding turn that may replace the panel's node list."""

    model_config = ConfigDict(frozen=True)

    nodes: list[BuildNode]
    changed: bool
    summary: str


def read_code_node(nodes: Sequence[BuildNode], node_id: str) -> dict[str, Any]:
    """Return complete workflow-local CODE data for one panel node."""
    node = next((candidate for candidate in nodes if candidate.id == node_id), None)
    if node is None:
        message = f'Node "{node_id}" does not exist on the current panel.'
        raise CodeEditingError(message)
    if node.type not in {CodeNodeType.BASH.value, CodeNodeType.PYTHON.value, CodeNodeType.R.value}:
        message = f'Node "{node_id}" is not a Bash, Python, or R CODE node.'
        raise CodeEditingError(message)

    code = node.data.get('code')
    if not isinstance(code, str) or not code.strip():
        message = f'CODE node "{node_id}" has no readable source code.'
        raise CodeEditingError(message)
    dependencies = node.data.get('dependencies', [])
    return {
        'id': node.id,
        'name': node.name,
        'type': node.type,
        'description': node.data.get('description', ''),
        'dependencies': dependencies if isinstance(dependencies, list) else [],
        'code': code,
    }


async def update_code_node(
    nodes: Sequence[BuildNode],
    node_id: str,
    request: str,
    *,
    edges: Sequence[BuildEdge],
    user_id: int,
    language: str | None = None,
) -> CodeNodeUpdate:
    """Edit existing source and Python dependencies without changing graph identity."""
    baseline = read_code_node(nodes, node_id)
    node_type = CodeNodeType(baseline['type'])
    if not request.strip():
        message = 'A code update request is required.'
        raise CodeEditingError(message)
    try:
        inputs = build_code_inputs([edge for edge in edges if edge.target_node == node_id], list(nodes), list(edges))
    except (KeyError, ValueError) as error:
        raise CodeEditingError(str(error)) from error

    generated = await run_coding_turn(
        request,
        node_type=node_type,
        user_id=user_id,
        source=baseline['code'],
        dependencies=baseline['dependencies'] if node_type is CodeNodeType.PYTHON else [],
        language=language,
        input_contract=inputs,
    )
    summary = next((message_text(message) for message in reversed(generated.messages) if message.type == 'ai'), '')
    if generated.proposal is None:
        return CodeNodeUpdate(
            nodes=list(nodes),
            changed=False,
            summary=summary or 'The coding agent returned no code changes.',
        )

    source = generated.proposal.source
    if not source.strip():
        message = 'Updated code must not be empty.'
        raise CodeEditingError(message)
    if '\x00' in source:
        message = 'Updated code must not contain NUL bytes.'
        raise CodeEditingError(message)
    if node_type is CodeNodeType.PYTHON:
        try:
            ast.parse(source)
        except SyntaxError as error:
            location = f'line {error.lineno}' if error.lineno is not None else 'an unknown line'
            message = f'Updated Python code has invalid syntax at {location}: {error.msg}.'
            raise CodeEditingError(message) from error

    updated_nodes = []
    changed = False
    for candidate in nodes:
        if candidate.id != node_id:
            updated_nodes.append(candidate.model_copy(deep=True))
            continue
        data = {**candidate.data, 'code': source}
        if node_type is CodeNodeType.PYTHON:
            data['dependencies'] = generated.proposal.dependencies
        changed = data != candidate.data
        updated_nodes.append(candidate.model_copy(update={'data': data}, deep=True))
    return CodeNodeUpdate(nodes=updated_nodes, changed=changed, summary=summary or 'Updated the CODE node.')
