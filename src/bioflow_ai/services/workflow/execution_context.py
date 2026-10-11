"""Execution context shared by plain node executors inside one worker task."""

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import takewhile
from pathlib import Path

from sqlalchemy.orm import Session

from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.crud import project_file_mapping as project_file_mapping_crud
from bioflow_ai.crud.resource import sample as sample_crud
from bioflow_ai.crud.tools import tool as tool_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.schemas.tools import ToolTaskInfo
from bioflow_ai.schemas.workflow_run import NodeRunStatus, NodeRunUpdate
from bioflow_ai.services.workflow.runtime import WorkflowRuntimeInvariantError

_current_node: ContextVar[uuid.UUID] = ContextVar('workflow_current_node')


@contextmanager
def node_execution_context(node_uid: uuid.UUID) -> Generator[None]:
    """Mark ``node_uid`` as the node whose executor runs inside this block."""
    token = _current_node.set(node_uid)
    try:
        yield
    finally:
        _current_node.reset(token)


def current_node_uid() -> uuid.UUID:
    """Return the executing node; raises ``LookupError`` outside an execution context."""
    return _current_node.get()


def get_tool_data(tool_uid: str) -> ToolTaskInfo:
    """Load the immutable tool execution configuration for a tool node."""
    with Session(engine) as session:
        tool = tool_crud.get_with_image(session, tool_uid)
        if not tool:
            message = f'Tool {tool_uid} does not exist'
            raise RuntimeError(message)
        if tool.image is None:
            message = f'Tool {tool_uid} has no image'
            raise RuntimeError(message)
        return ToolTaskInfo(
            uid=tool.uid,
            name=tool.name,
            file_mounts=tool.file_mounts,
            command_template=tool.command_template,
            dynamic_params=tool.dynamic_params,
            immutable_static_params=tool.immutable_static_params,
            modifiable_static_params=tool.modifiable_static_params,
            image=tool.image,
        )


def update_node(data: NodeRunUpdate) -> None:
    """Persist executor metadata on the currently executing node while it is running."""
    values = data.model_dump(exclude_none=True)
    if not values:
        return
    with Session(engine) as session, session.begin():
        node_run_crud.update_if_status(session, [current_node_uid()], [NodeRunStatus.RUNNING], values)


def get_node_output(source_uid: str, output_key: str) -> tuple[str, str | None]:
    """Resolve one predecessor output referenced by the executing node's frozen inputs.

    The compiler builds these references from the run's edges, and a node only becomes
    READY after every predecessor has succeeded, so the source output is already recorded.
    """
    with Session(engine) as session:
        source = node_run_crud.get(session, uuid.UUID(source_uid))
        if source is None or source.tool_output is None:
            message = f'Dependency {source_uid} has no recorded output'
            raise WorkflowRuntimeInvariantError(message)
        if output_key == 'workdir' and source.base_dir:
            return str(Path(source.base_dir).absolute()), source.tool_output.sample_name
        values = source.tool_output.all_files()
        if output_key not in values:
            message = f'Dependency {source_uid} has no output named {output_key!r}'
            raise WorkflowRuntimeInvariantError(message)
        return str(values[output_key]), source.tool_output.sample_name


def resolve_path(
    path_or_reference: str,
    session: Session,
    project_id: int | None,
    sample_uid: str | None,
) -> tuple[Path, str | None]:
    """Resolve a direct, project, or sample path reference."""
    if not path_or_reference.startswith(('proj:', 'sample:')):
        return Path(path_or_reference), None
    if path_or_reference.startswith('proj:'):
        return _resolve_project_path(path_or_reference, session, project_id)
    return _resolve_sample_path(path_or_reference, session, sample_uid)


def get_prefix(strings: list[str | None]) -> str | None:
    """Return the common biological-sample name prefix."""
    valid_strings = [value for value in strings if value is not None]
    if not valid_strings:
        return None
    pairs = zip(*valid_strings, strict=False)
    common = takewhile(lambda characters: all(character == characters[0] for character in characters), pairs)
    return ''.join(characters[0] for characters in common).rstrip('.-_rR')


def _resolve_project_path(reference: str, session: Session, project_id: int | None) -> tuple[Path, str | None]:
    if project_id is None:
        message = f'Cannot resolve {reference}: temporary workflows must use a direct path.'
        raise ValueError(message)
    keyword = reference.split(':', 1)[1]
    mapping = project_file_mapping_crud.get_by_keyword(session, project_id, keyword)
    if not mapping:
        message = f'Project file mapping not found: {keyword}'
        raise FileNotFoundError(message)
    return Path(mapping.file_path).absolute(), keyword


def _resolve_sample_path(reference: str, session: Session, sample_uid: str | None) -> tuple[Path, str | None]:
    if sample_uid is None:
        message = f'Cannot resolve {reference}: project workflows must use a project mapping.'
        raise ValueError(message)
    tag = reference.split(':', 1)[1]
    sample = sample_crud.get_with_files(session, sample_uid)
    if not sample:
        message = f'Sample not found: {sample_uid}'
        raise KeyError(message)
    matching_file = next((file for file in sample.sample_files if file.tag == tag), None)
    if not matching_file:
        message = f'Sample file with tag {tag!r} not found'
        raise KeyError(message)
    return Path(matching_file.file_path).absolute(), tag
