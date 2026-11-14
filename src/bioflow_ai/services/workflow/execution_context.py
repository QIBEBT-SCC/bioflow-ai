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
from bioflow_ai.schemas.passthrough import ToolWorkspace
from bioflow_ai.schemas.tools import ToolTaskInfo
from bioflow_ai.schemas.workflow_run import NodeRunStatus, NodeRunUpdate
from bioflow_ai.services.workflow.output_validation import output_path_exists
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
            tool_type=tool.tool_type,
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
        value = values[output_key]
        if source.node_type == 'tool' and not output_path_exists(value):
            message = f'Dependency {source.name!r} ({source_uid}) output {output_key!r} does not exist: {value}'
            raise FileNotFoundError(message)
        return str(value), source.tool_output.sample_name


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


def _dynamic_mark_root(session: Session, file_path: str, producer_run_uid: uuid.UUID | None) -> str:
    from bioflow_ai.models.workflow import WorkflowRun

    if producer_run_uid is not None and (producer := session.get(WorkflowRun, producer_run_uid)) and producer.base_dir:
        return str(Path(producer.base_dir).absolute())
    resolved = Path(file_path).absolute()
    return str(resolved if resolved.is_dir() else resolved.parent)


def _dynamic_resource_origin(
    session: Session,
    node: object,
    current: object,
    source_key: str,
) -> tuple[bool, str | None]:
    node_type = getattr(node, 'node_type', '')
    if node_type not in {'resource_file', 'resource_sequence'}:
        return False, None
    kwargs = (getattr(node, 'prepared_data', None) or {}).get('kwargs', {})
    if node_type == 'resource_file':
        ref = str(kwargs.get('file', '')).strip()
    else:
        key = source_key or 'r1'
        default = f'sample:{key}' if key in {'r1', 'r2'} else ''
        ref = str(kwargs.get(key, default)).strip()

    run = getattr(current, 'run', None)
    project_id = kwargs.get('project_id') or (run.project_id if run is not None else None)
    sample_uid = kwargs.get('sample_uid') or (str(run.sample_uid) if run is not None and run.sample_uid else None)

    if ref.startswith('proj:') and project_id is not None:
        mapping = project_file_mapping_crud.get_by_keyword(session, int(project_id), ref.removeprefix('proj:'))
        if mapping is not None and mapping.is_dynamic:
            return True, _dynamic_mark_root(session, mapping.file_path, mapping.producer_run_uid)
        return False, None

    if ref.startswith('sample:') and sample_uid is not None:
        tag = ref.removeprefix('sample:')
        sample = sample_crud.get_with_files(session, str(sample_uid))
        if sample is not None:
            matching = next((item for item in sample.sample_files if item.tag == tag), None)
            if matching is not None and matching.is_dynamic:
                return True, _dynamic_mark_root(session, matching.file_path, matching.producer_run_uid)
    return False, None


def tool_input_provenance(
    source_uid: str,
    *,
    writable: bool = False,
    source_key: str = '',
) -> tuple[ToolWorkspace | None, str, str | None]:
    """Follow path adapters to the nearest producing Tool or resource node.

    Returns:
        ``(workspace, source_node_uid, dynamic_root)`` where ``workspace`` and ``source_node_uid``
        identify an upstream Tool workspace (or ``(None, '')``), and ``dynamic_root`` identifies a
        trusted prior-run directory when the input resolves a dynamic workflow mark.
    """
    from bioflow_ai.models.workflow import NodeRun
    from bioflow_ai.services.workflow.validation.passthrough import PATH_INPUTS, validate_passthrough_origin

    with Session(engine) as session:
        current = session.get(NodeRun, current_node_uid())
        node = session.get(NodeRun, uuid.UUID(source_uid))
        current_key = source_key
        dynamic_root: str | None = None
        seen = set()
        while node is not None:
            base_input = PATH_INPUTS.get(node.node_type)
            is_dynamic = False
            if base_input is None:
                is_dynamic, dynamic_root = _dynamic_resource_origin(session, node, current, current_key)
            validate_passthrough_origin(
                node,
                current,
                seen,
                writable=writable and base_input is None,
                is_dynamic_mark=is_dynamic,
            )
            seen.add(node.uid)
            if base_input is None:
                break
            upstream = (node.prepared_data or {}).get('kwargs', {}).get('inputs', {}).get(base_input)
            if not upstream:
                message = 'Path adapter has no original source'
                raise ValueError(message)
            current_key = str(upstream.get('source_k', ''))
            node = session.get(NodeRun, uuid.UUID(upstream['source_uid']))
        if node is None:
            message = 'Pass-through source is unavailable'
            raise ValueError(message)
        workspace = node.tool_output.workspace if node.tool_output else None
        return (workspace, str(node.uid), dynamic_root) if workspace else (None, '', dynamic_root)


def current_run_provenance() -> tuple[str, str]:
    """Supply trusted identity for newly published workspace artifacts.

    Returns:
        The current workflow run uid and its absolute managed directory.
    """
    from bioflow_ai.models.workflow import NodeRun, WorkflowRun

    with Session(engine) as session:
        node = session.get(NodeRun, current_node_uid())
        if node is None:
            message = 'Node run is unavailable'
            raise ValueError(message)
        run = session.get(WorkflowRun, node.run_uid)
        if run is None:
            message = 'Workflow run is unavailable'
            raise ValueError(message)
        return str(run.uid), str(Path(run.base_dir).absolute())
