"""Read and safely replace workflow-local CODE node source."""

import ast
import hashlib
from collections.abc import Sequence
from typing import Any

from app.schemas.code import CodeNodeType
from llm.states import BuildNode


class CodeEditingError(ValueError):
    """A requested panel CODE-node read or update is invalid."""


def code_sha256(code: str) -> str:
    """Return a stable digest for optimistic CODE-node updates."""
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


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
        'code_sha256': code_sha256(code),
        'code': code,
    }


def update_code_node(
    nodes: Sequence[BuildNode],
    node_id: str,
    code: str,
    expected_code_sha256: str,
) -> tuple[list[BuildNode], str]:
    """Replace panel CODE source while preserving node identity and other data."""
    node = next((candidate for candidate in nodes if candidate.id == node_id), None)
    if node is None:
        message = f'Node "{node_id}" does not exist on the current panel.'
        raise CodeEditingError(message)
    if node.type not in {CodeNodeType.BASH.value, CodeNodeType.PYTHON.value, CodeNodeType.R.value}:
        message = f'Node "{node_id}" is not a Bash, Python, or R CODE node.'
        raise CodeEditingError(message)

    current_code = node.data.get('code')
    if not isinstance(current_code, str) or not current_code.strip():
        message = f'CODE node "{node_id}" has no editable source code.'
        raise CodeEditingError(message)
    if code_sha256(current_code) != expected_code_sha256:
        message = f'CODE node "{node_id}" changed after it was read. Call read_code again before updating it.'
        raise CodeEditingError(message)
    if not code.strip():
        message = 'Replacement code must not be empty.'
        raise CodeEditingError(message)
    if '\x00' in code:
        message = 'Replacement code must not contain NUL bytes.'
        raise CodeEditingError(message)

    normalized_code = f'{code.rstrip()}\n'
    if node.type == CodeNodeType.PYTHON.value:
        try:
            ast.parse(normalized_code)
        except SyntaxError as error:
            location = f'line {error.lineno}' if error.lineno is not None else 'an unknown line'
            message = f'Replacement Python code has invalid syntax at {location}: {error.msg}.'
            raise CodeEditingError(message) from error
    if normalized_code == f'{current_code.rstrip()}\n':
        message = 'Replacement code is unchanged.'
        raise CodeEditingError(message)

    updated_nodes = []
    for candidate in nodes:
        if candidate.id != node_id:
            updated_nodes.append(candidate.model_copy(deep=True))
            continue
        updated_nodes.append(
            candidate.model_copy(
                update={'data': {**candidate.data, 'code': normalized_code}},
                deep=True,
            )
        )
    return updated_nodes, code_sha256(normalized_code)
