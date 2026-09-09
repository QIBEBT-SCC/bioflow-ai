"""Execution context shared by plain node executors inside one worker task."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import takewhile
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.session import engine
from app.crud import project_file_mapping as project_file_mapping_crud
from app.crud.resource import sample as sample_crud
from app.models.tools import ToolDefine
from app.schemas.tools import ToolTaskInfo
from app.schemas.workflow_run import NodeRunUpdate

_output_resolver: ContextVar[Callable[[str, str], tuple[str, str | None]] | None] = ContextVar(
    'workflow_output_resolver',
    default=None,
)
_node_updater: ContextVar[Callable[[str, NodeRunUpdate], bool] | None] = ContextVar(
    'workflow_node_updater',
    default=None,
)
_MISSING_CONTEXT = 'Node executor is not running inside a workflow execution context'


@contextmanager
def node_execution_context(
    *,
    output_resolver: Callable[[str, str], tuple[str, str | None]],
    node_updater: Callable[[str, NodeRunUpdate], bool],
) -> Generator[None]:
    """Install dependency and persistence hooks for one local node execution."""
    resolver_token = _output_resolver.set(output_resolver)
    updater_token = _node_updater.set(node_updater)
    try:
        yield
    finally:
        _node_updater.reset(updater_token)
        _output_resolver.reset(resolver_token)


def get_tool_data(tool_uid: str) -> ToolTaskInfo:
    """Load the immutable tool execution configuration for a tool node."""
    with Session(engine) as session:
        statement = select(ToolDefine).where(ToolDefine.uid == tool_uid).options(joinedload(ToolDefine.image))
        tool = session.scalar(statement)
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


def update_node(uid: str, data: NodeRunUpdate) -> bool:
    """Persist metadata for the currently executing node."""
    updater = _node_updater.get()
    if updater is None:
        raise RuntimeError(_MISSING_CONTEXT)
    return updater(uid, data)


def get_node_output(uid: str, key: str) -> tuple[str, str | None]:
    """Resolve one declared predecessor output without polling."""
    resolver = _output_resolver.get()
    if resolver is None:
        raise RuntimeError(_MISSING_CONTEXT)
    return resolver(uid, key)


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
