"""Read and patch workflow-local CODE node source."""

import ast
from collections.abc import Sequence
from typing import Any

from bioflow_ai.llm.utils.patch import MAX_PATCH_CHARS, PatchError, derive_new_contents, parse_patch_chunks
from bioflow_ai.schemas.code import CodeNodeType
from bioflow_ai.schemas.workflow_build import BuildNode


class CodeEditingError(ValueError):
    """A requested panel CODE-node read or update is invalid."""


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


def update_code_node(
    nodes: Sequence[BuildNode],
    node_id: str,
    patch: str,
) -> list[BuildNode]:
    """Apply context hunks to panel CODE source while preserving other node data."""
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
    if len(patch) > MAX_PATCH_CHARS:
        message = 'Patch exceeds the maximum size.'
        raise CodeEditingError(message)

    try:
        chunks = parse_patch_chunks(patch.replace('\r\n', '\n').strip('\n').split('\n'), node_id)
        patched_code = derive_new_contents(current_code, chunks, node_id)
    except PatchError as error:
        raise CodeEditingError(str(error)) from error
    if not patched_code.strip():
        message = 'Patched code must not be empty.'
        raise CodeEditingError(message)
    if '\x00' in patched_code:
        message = 'Patched code must not contain NUL bytes.'
        raise CodeEditingError(message)
    if node.type == CodeNodeType.PYTHON.value:
        try:
            ast.parse(patched_code)
        except SyntaxError as error:
            location = f'line {error.lineno}' if error.lineno is not None else 'an unknown line'
            message = f'Patched Python code has invalid syntax at {location}: {error.msg}.'
            raise CodeEditingError(message) from error
    if patched_code == current_code:
        message = 'Patch does not change the code.'
        raise CodeEditingError(message)

    updated_nodes = []
    for candidate in nodes:
        if candidate.id != node_id:
            updated_nodes.append(candidate.model_copy(deep=True))
            continue
        updated_nodes.append(
            candidate.model_copy(
                update={'data': {**candidate.data, 'code': patched_code}},
                deep=True,
            )
        )
    return updated_nodes
