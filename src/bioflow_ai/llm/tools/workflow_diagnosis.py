"""Read-only workflow execution evidence and optional diagnosis report output."""

import math
import re
import shlex
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session, load_only

import bioflow_ai.crud.project_file_mapping as project_file_mapping_crud
import bioflow_ai.crud.resource.sample as sample_crud
from bioflow_ai.infrastructure.containers import ContainerMount, run_command
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import WorkflowAssistantContext
from bioflow_ai.models.workflow import NodeRun, WorkflowRun
from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.schemas.tool_config import ImageConfig
from bioflow_ai.schemas.workflow_run import NodeRunStatus, RunStatus
from bioflow_ai.services.workflow.node_details import describe_node
from bioflow_ai.services.workflow.subgraphs import SubgraphError, definition_id, enter_subgraphs
from bioflow_ai.utils.formatters import format_json

EVIDENCE_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-seq:1.0.0')
EVIDENCE_MOUNT_PATH = '/app/data/input'
EVIDENCE_WORKDIR = '/app/data'
MAX_COMMAND_CHARS = 8_000
COMMAND_TIMEOUT_SECONDS = 60
COMMAND_KILL_GRACE_SECONDS = 5
MAX_CAPTURE_BYTES = 64 * 1024
MAX_RETURN_CHARS = 40_000
RETURN_HEAD_CHARS = 30_000
MAX_CHILD_PAGE = 100
_CAPTURE_FILE_BLOCKS = math.ceil(MAX_CAPTURE_BYTES / 512)
_WRAPPER_METADATA = re.compile(
    r'^__BIOFLOW_DIAG_STATUS__=(?P<status>\d+)\n'
    r'^__BIOFLOW_DIAG_CAPTURE_BYTES__=(?P<bytes>\d+)\n'
    r'^__BIOFLOW_DIAG_CAPTURE_TRUNCATED__=(?P<truncated>[01])\n?',
    re.MULTILINE,
)


# ============ Workflow Reader =============
def _parents_and_children(workflow: WorkflowDefinition) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    for edge in workflow.edges:
        parents[edge.target].add(edge.source)
        children[edge.source].add(edge.target)
    return parents, children


def _upstream_scope(node_ids: list[str], parents: dict[str, set[str]], depth: int) -> list[str]:
    """List targets first, then their ancestors by shortest upstream distance."""
    ordered = list(dict.fromkeys(node_ids))
    seen = set(ordered)
    frontier = ordered
    for _ in range(depth):
        upstream = sorted({parent for node_id in frontier for parent in parents.get(node_id, ())} - seen)
        if not upstream:
            break
        ordered.extend(upstream)
        seen.update(upstream)
        frontier = upstream
    return ordered


def _descendant_run(session: Session, root: WorkflowRun, uid: uuid.UUID) -> WorkflowRun | None:
    """Only expose runs linked by ForEach parent nodes of the root run."""
    child = session.get(WorkflowRun, uid)
    current = child
    for _ in range(33):
        if current is None or current.owner_id != root.owner_id or current.project_id != root.project_id:
            return None
        if current.uid == root.uid:
            return child
        if current.parent_node_uid is None:
            return None
        parent_node = session.get(NodeRun, current.parent_node_uid)
        if parent_node is None:
            return None
        current = session.get(WorkflowRun, parent_node.run_uid)
    return None


def _selected_run(session: Session, context: WorkflowAssistantContext, child_run_uid: str | None) -> WorkflowRun:
    root = _current_run(session, context)
    if context.source_generation is not None and root.generation != context.source_generation:
        code = 'generation_changed'
        raise _evidence_error(code, 'The source run generation changed; start a new request.')
    if not child_run_uid:
        return root
    try:
        child = _descendant_run(session, root, uuid.UUID(child_run_uid))
    except ValueError as error:
        code = 'run_not_found'
        raise _evidence_error(code, 'Invalid child run UID.') from error
    if child is None:
        code = 'access_denied'
        raise _evidence_error(code, 'Child run does not belong to this source run generation.')
    return child


def _workflow_payload(
    context: WorkflowAssistantContext,
    action: str,
    include_node_id: str | None = None,
    subgraph_path: list[str] | None = None,
    child_run_uid: str | None = None,
    offset: int = 0,
    limit: int = 20,
    *,
    failed_only: bool = True,
    upstream_depth: int = 0,
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= MAX_CHILD_PAGE:
        return {'error': 'offset must be nonnegative and limit must be between 1 and 100.'}
    if upstream_depth < 0:
        return {'error': 'upstream_depth must be nonnegative.'}
    if upstream_depth and action != 'overview':
        return {'error': 'upstream_depth is only supported for action=overview.'}
    if upstream_depth and not include_node_id and not failed_only:
        return {'error': 'upstream_depth requires node_id or failed_only=true.'}
    if action in {'node', 'children'} and not include_node_id:
        return {'error': f'node_id is required for action={action}.'}
    try:
        with Session(engine) as session:
            run = _selected_run(session, context, child_run_uid)
            path = tuple(subgraph_path or ())
            try:
                graph, _ = enter_subgraphs(run.workflow, list(path))
            except SubgraphError as error:
                return {'error': str(error)}
            graph_nodes = {node.id: node for node in graph.nodes}
            if include_node_id is not None and include_node_id not in graph_nodes:
                return {'error': f'Node not found: {include_node_id}'}
            target_runtime_id = definition_id((*path, include_node_id)) if include_node_id is not None else None
            if action == 'children':
                control = session.scalar(
                    select(NodeRun)
                    .where(
                        NodeRun.run_uid == run.uid,
                        NodeRun.definition_node_id == target_runtime_id,
                    )
                    .options(load_only(NodeRun.node_type))
                )
                if control is None or control.node_type != 'foreach':
                    return {'error': 'No ForEach control task exists for this node.'}
                filters = [WorkflowRun.parent_node_uid == control.uid]
                if failed_only:
                    filters.append(WorkflowRun.status == RunStatus.FAILED)
                children = list(
                    session.scalars(
                        select(WorkflowRun)
                        .where(*filters)
                        .options(
                            load_only(
                                WorkflowRun.item_index,
                                WorkflowRun.item_name,
                                WorkflowRun.generation,
                                WorkflowRun.status,
                                WorkflowRun.base_dir,
                            )
                        )
                        .order_by(WorkflowRun.item_index)
                        .offset(offset)
                        .limit(limit + 1)
                    )
                )
                return {
                    'parent_run_uid': str(run.uid),
                    'parent_generation': run.generation,
                    'subgraph_path': list(path),
                    'node_id': include_node_id,
                    **({'next_offset': offset + limit} if len(children) > limit else {}),
                    'children': [
                        {
                            'run_uid': str(item.uid),
                            **({'item_index': item.item_index} if item.item_index is not None else {}),
                            **({'item_name': item.item_name} if item.item_name is not None else {}),
                            'generation': item.generation,
                            'status': item.status.value,
                            'base_dir': item.base_dir,
                        }
                        for item in children[:limit]
                    ],
                }
            node_runs = list(
                session.scalars(
                    select(NodeRun)
                    .where(NodeRun.run_uid == run.uid)
                    .options(
                        load_only(
                            NodeRun.definition_node_id,
                            NodeRun.node_type,
                            NodeRun.name,
                            NodeRun.status,
                        )
                    )
                )
            )
            by_id = {node.definition_node_id: node for node in node_runs}
            parents, _ = _parents_and_children(graph)
            node_statuses = {}
            for node in graph.nodes:
                runtime_id = definition_id((*path, node.id))
                task = by_id.get(runtime_id)
                if task is None and node.type == 'subgraph':
                    prefix = definition_id((*path, node.id, ''))
                    internal = [item for key, item in by_id.items() if key.startswith(prefix)]
                    task_status = (
                        'failed'
                        if any(item.status == NodeRunStatus.FAILED for item in internal)
                        else (
                            'blocked'
                            if any(item.status == NodeRunStatus.BLOCKED for item in internal)
                            else 'running'
                            if any(item.status == NodeRunStatus.RUNNING for item in internal)
                            else 'succeeded'
                            if internal and all(item.status == NodeRunStatus.SUCCEEDED for item in internal)
                            else 'pending'
                        )
                    )
                else:
                    task_status = task.status.value if task else 'presentation_only'
                node_statuses[node.id] = task_status

            failed_node_ids = [node_id for node_id, status in node_statuses.items() if status == 'failed']
            targets = (
                [include_node_id]
                if include_node_id is not None
                else failed_node_ids
                if failed_only
                else list(graph_nodes)
            )
            scoped_ids = [
                node_id for node_id in _upstream_scope(targets, parents, upstream_depth) if node_id in graph_nodes
            ]
            shown_ids = scoped_ids[offset : offset + limit] if action == 'overview' else scoped_ids
            if action == 'node':
                detail = session.scalar(
                    select(NodeRun)
                    .where(
                        NodeRun.run_uid == run.uid,
                        NodeRun.definition_node_id == target_runtime_id,
                    )
                    .execution_options(populate_existing=True)
                )
                if detail is not None:
                    by_id[detail.definition_node_id] = detail

            nodes = []
            for node_id in shown_ids:
                node = graph_nodes[node_id]
                runtime_id = definition_id((*path, node.id))
                task = by_id.get(runtime_id)
                payload: dict[str, Any] = {
                    'node_id': node.id,
                    'name': node.data.get('name') or (task.name if task else node.id),
                    'type': node.type,
                    'status': node_statuses[node.id],
                }
                if node.type == 'subgraph':
                    payload['child_path'] = [*path, node.id]
                    payload['batch'] = task is not None and task.node_type == 'foreach'
                if action == 'node':
                    payload.update(describe_node(session, graph, node, path, by_id))
                nodes.append(payload)
            result = {
                'run': {
                    'workflow_uid': str(run.workflow_uid) if run.workflow_uid else None,
                    'status': run.status.value,
                    'base_dir': run.base_dir,
                    **({'item_index': run.item_index} if run.item_index is not None else {}),
                    **({'item_name': run.item_name} if run.item_name is not None else {}),
                },
                'subgraph_path': list(path),
                'nodes': nodes,
            }
            if action == 'overview':
                shown_set = set(shown_ids)
                result['edges'] = [
                    edge.model_dump(include={'source', 'sourceHandle', 'target', 'targetHandle'})
                    for edge in graph.edges
                    if edge.source in shown_set and edge.target in shown_set
                ]
                if offset + limit < len(scoped_ids):
                    result['next_offset'] = offset + limit
            return result
    except EvidenceSourceError as error:
        return {'error': {'code': error.code, 'message': str(error)}}


@tool('workflow_reader', parse_docstring=True)
def workflow_reader(
    action: Literal['overview', 'node', 'children'],
    runtime: ToolRuntime[WorkflowAssistantContext],
    node_id: str | None = None,
    subgraph_path: list[str] | None = None,
    child_run_uid: str | None = None,
    offset: int = 0,
    limit: int = 20,
    *,
    failed_only: bool = True,
    upstream_depth: int = 0,
) -> str:
    """Read compact failure summaries, a bounded upstream graph, or one node's full evidence.

    Start diagnosis with ``overview`` to list only failed nodes. For
    a named node, use ``node`` directly. Expand relevant context with
    ``overview(node_id=..., upstream_depth=1)``; increase the depth only when needed.
    Successful ancestors are included in the scope. Overview never returns full node
    configuration or scripts; use ``node`` for port descriptions, connected sources,
    the assembled command template, recorded output results and CODE source.

    Args:
        action: ``overview`` returns paginated compact node summaries and edges within
            the returned page; ``node`` returns configuration, commands and input/output
            handles with their sources or recorded results, without internal task metadata;
            ``children`` lists ForEach child runs (failed by default).
        node_id: Local node ID, required for ``node`` and ``children``. For ``overview``,
            focus on this node and its ancestors instead of filtering by failure status.
        subgraph_path: Nested subgraph node IDs to enter in the selected run.
        child_run_uid: ForEach child run UID returned by ``children``.
        offset: First scoped node or child item to list. Follow ``next_offset`` when
            present; its absence means all matching results have been returned.
        limit: Maximum scoped nodes or child items to list (1 to 100, default 20).
        failed_only: Default true. Without ``node_id``, overview starts from failed
            nodes; children lists failed runs. Set false explicitly to list all nodes
            or child runs. Ignored by ``node`` and by an overview with ``node_id``.
        upstream_depth: Overview only. Number of upstream edge hops (default 0).
            One hop includes all direct parents, two includes their parents as well.
            Start from ``node_id`` or all failed nodes; targets come first, then ancestors
            ordered by shortest distance. All scopes support pagination, including
            scopes inside ordinary subgraphs or ForEach child runs.
    """
    if action == 'node' and not node_id:
        return format_json({'error': 'node_id is required for action=node.'})
    return format_json(
        _workflow_payload(
            runtime.context,
            action,
            node_id,
            subgraph_path,
            child_run_uid,
            offset,
            limit,
            failed_only=failed_only,
            upstream_depth=upstream_depth,
        ),
        max_chars=None,
    )


# ============ Run Command =============


class EvidenceSourceError(ValueError):
    """A recoverable error while resolving an evidence source."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _evidence_error(code: str, message: str) -> EvidenceSourceError:
    return EvidenceSourceError(code, message)


class ResolvedEvidence(BaseModel):
    """A validated host resource that may be mounted into the evidence container."""

    model_config = ConfigDict(frozen=True)

    source: str
    path: Path
    kind: Literal['file', 'directory']


def _error(code: str, message: str) -> str:
    return format_json({'ok': False, 'error': {'code': code, 'message': message}}, max_chars=None)


def _validated_path(raw_path: str | None, source: str) -> tuple[Path, Literal['file', 'directory']]:
    if not raw_path:
        msg = 'missing_path'
        raise _evidence_error(msg, f'Evidence source `{source}` has no filesystem path.')
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        msg = 'path_not_found'
        raise _evidence_error(msg, f'Evidence source `{source}` no longer exists.') from error
    if path.is_file():
        return path, 'file'
    if path.is_dir():
        return path, 'directory'
    msg = 'unsupported_path'
    raise _evidence_error(msg, f'Evidence source `{source}` is not a regular file or directory.')


def _current_run(session: Session, context: WorkflowAssistantContext) -> Any:
    try:
        run_uid = uuid.UUID(context.source_run_uid)
    except ValueError as error:
        msg = 'run_not_found'
        raise _evidence_error(msg, 'The workflow execution no longer exists.') from error
    run = session.get(WorkflowRun, run_uid)
    if run is None:
        msg = 'run_not_found'
        raise _evidence_error(msg, 'The workflow execution no longer exists.')
    if run.owner_id != context.user_id:
        msg = 'access_denied'
        raise _evidence_error(msg, 'The workflow execution does not belong to the current user.')
    if run.project_id != context.project_id:
        msg = 'project_mismatch'
        raise _evidence_error(msg, 'The workflow execution does not belong to the current project.')
    return run


def resolve_evidence_source(source: str, context: WorkflowAssistantContext) -> ResolvedEvidence:
    """Resolve one logical evidence reference without accepting arbitrary host paths."""
    with Session(engine) as session:
        run = _selected_run(session, context, None)

        if source == 'run:current':
            path, kind = _validated_path(run.base_dir, source)
            if kind != 'directory':
                msg = 'invalid_run_path'
                raise _evidence_error(msg, 'The current run path is not a directory.')
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('run:child:'):
            child = _selected_run(session, context, source.removeprefix('run:child:'))
            path, kind = _validated_path(child.base_dir, source)
            if kind != 'directory':
                code = 'invalid_run_path'
                raise _evidence_error(code, 'The child run path is not a directory.')
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('sample:'):
            tag = source.removeprefix('sample:')
            if not tag:
                msg = 'invalid_source'
                raise _evidence_error(msg, 'A sample source must include a tag.')
            if run.sample_uid is None:
                msg = 'sample_unavailable'
                raise _evidence_error(
                    msg,
                    'The current workflow execution is project-level and has no bound sample.',
                )
            sample = sample_crud.get_with_files(session, run.sample_uid)
            if sample is None:
                msg = 'sample_not_found'
                raise _evidence_error(msg, 'The sample bound to the current run no longer exists.')
            if sample.owner_id != context.user_id or sample.project_id != context.project_id:
                msg = 'access_denied'
                raise _evidence_error(msg, 'The current run sample is outside the allowed project scope.')
            sample_file = next((item for item in sample.sample_files if item.tag == tag), None)
            if sample_file is None:
                msg = 'reference_not_found'
                raise _evidence_error(msg, f'Sample reference `{source}` was not found.')
            path, kind = _validated_path(sample_file.file_path, source)
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('proj:'):
            keyword = source.removeprefix('proj:')
            if not keyword:
                msg = 'invalid_source'
                raise _evidence_error(msg, 'A project source must include a keyword.')
            mapping = project_file_mapping_crud.get_by_keyword(session, context.project_id, keyword)
            if mapping is None:
                msg = 'reference_not_found'
                raise _evidence_error(msg, f'Project reference `{source}` was not found.')
            if mapping.project_id != context.project_id:
                msg = 'project_mismatch'
                raise _evidence_error(msg, 'The project reference is outside the allowed project scope.')
            path, kind = _validated_path(mapping.file_path, source)
            return ResolvedEvidence(source=source, path=path, kind=kind)

    msg = 'invalid_source'
    raise _evidence_error(
        msg,
        'source must be `run:current`, `run:child:<uid>`, `sample:<tag>`, or `proj:<keyword>`.',
    )


def build_bounded_command(command: str) -> str:
    """Wrap an inspection command with time and captured-output limits inside the container."""
    stripped = command.strip()
    if not stripped:
        msg = 'empty_command'
        raise _evidence_error(msg, 'The evidence inspection command cannot be empty.')
    if len(stripped) > MAX_COMMAND_CHARS:
        msg = 'command_too_long'
        raise _evidence_error(
            msg,
            f'The evidence inspection command exceeds {MAX_COMMAND_CHARS} characters.',
        )

    quoted_command = shlex.quote(stripped)
    return (
        'output_file=$(mktemp)\n'
        '(\n'
        f'  ulimit -f {_CAPTURE_FILE_BLOCKS}\n'
        f'  timeout -k {COMMAND_KILL_GRACE_SECONDS}s {COMMAND_TIMEOUT_SECONDS}s /bin/sh -c {quoted_command}\n'
        ') >"$output_file" 2>&1\n'
        'command_status=$?\n'
        'capture_bytes=$(wc -c <"$output_file" | tr -d " ")\n'
        f'if [ "$capture_bytes" -ge {MAX_CAPTURE_BYTES} ]; then capture_truncated=1; else capture_truncated=0; fi\n'
        'printf "__BIOFLOW_DIAG_STATUS__=%s\\n" "$command_status"\n'
        'printf "__BIOFLOW_DIAG_CAPTURE_BYTES__=%s\\n" "$capture_bytes"\n'
        'printf "__BIOFLOW_DIAG_CAPTURE_TRUNCATED__=%s\\n" "$capture_truncated"\n'
        f'head -c {RETURN_HEAD_CHARS} "$output_file"\n'
        f'if [ "$capture_bytes" -gt {RETURN_HEAD_CHARS} ]; then\n'
        '  printf "\\n... [container output tail] ...\\n"\n'
        f'  tail -c {MAX_RETURN_CHARS - RETURN_HEAD_CHARS} "$output_file"\n'
        'fi\n'
        'rm -f "$output_file"\n'
        'exit "$command_status"'
    )


def _parse_command_output(output: str, exit_code: int) -> tuple[str, int, int | None, bool]:
    match = _WRAPPER_METADATA.match(output)
    capture_bytes: int | None = None
    capture_truncated = False
    if match is not None:
        exit_code = int(match.group('status'))
        capture_bytes = int(match.group('bytes'))
        capture_truncated = match.group('truncated') == '1'
        output = output[match.end() :]

    output = output.strip()
    python_truncated = len(output) > MAX_RETURN_CHARS
    if python_truncated:
        tail_chars = MAX_RETURN_CHARS - RETURN_HEAD_CHARS
        omitted = len(output) - MAX_RETURN_CHARS
        output = f'{output[:RETURN_HEAD_CHARS]}\n... [truncated {omitted} characters] ...\n{output[-tail_chars:]}'
    return output, exit_code, capture_bytes, capture_truncated or python_truncated


@tool('run_command', parse_docstring=True)
def run_evidence_command(
    source: str,
    command: str,
    runtime: ToolRuntime[WorkflowAssistantContext],
) -> str:
    """Run a bounded shell inspection against one read-only workflow evidence source.

    The source is mounted at ``/app/data/input``. It may be a file or a directory, so
    inspect it with ``file /app/data/input`` when the type is not already known.

    Args:
        source: ``run:current``, ``run:child:<uid>``, ``sample:<tag>``, or ``proj:<keyword>``.
        command: Shell inspection command using ``/app/data/input`` as the evidence path.
    """
    try:
        evidence = resolve_evidence_source(source, runtime.context)
        bounded_command = build_bounded_command(command)
    except EvidenceSourceError as error:
        return _error(error.code, str(error))

    mount = ContainerMount(source=str(evidence.path), target=EVIDENCE_MOUNT_PATH, read_only=True)
    try:
        exit_code, raw_output = run_command(
            EVIDENCE_IMAGE,
            [mount],
            bounded_command,
            workdir=EVIDENCE_WORKDIR,
            ensure_ascii=False,
        )
    except Exception as error:  # Container failures are recoverable evidence gaps for the agent.
        return _error('container_error', f'{type(error).__name__}: {error}')

    output, exit_code, capture_bytes, truncated = _parse_command_output(raw_output, exit_code)
    return format_json(
        {
            'ok': exit_code == 0,
            'source': evidence.source,
            'mount_path': EVIDENCE_MOUNT_PATH,
            'source_type': evidence.kind,
            'exit_code': exit_code,
            'timed_out': exit_code in {124, 137},
            'truncated': truncated,
            'captured_bytes': capture_bytes,
            'output': output,
        },
        max_chars=None,
    )


@tool('write_diagnosis', parse_docstring=True)
def write_diagnosis(content: str, runtime: ToolRuntime[WorkflowAssistantContext]) -> Command:
    """Write the completed workflow diagnosis to ``DIAGNOSIS.md``.

    Use this tool when the user requests diagnosis, troubleshooting, or an investigation
    into a workflow failure, following the diagnosis skill. Do not call it for ordinary
    questions, status or result explanations, or follow-up questions about an existing report.

    Args:
        content: Complete Markdown diagnosis report in the language required by the system prompt.
    """
    report = content.strip()
    if not report:
        message = 'The diagnosis report cannot be empty.'
        raise ValueError(message)

    project_root = Path(runtime.context.workdir).expanduser().resolve()
    agent_run_uid = uuid.UUID(runtime.context.agent_run_uid)
    output_dir = project_root / 'fixes' / str(agent_run_uid)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'DIAGNOSIS.md'
    temporary_path = output_dir / f'.DIAGNOSIS.{uuid.uuid4().hex}.tmp'
    metadata = (
        '---\n'
        'schema_version: 1\n'
        f'diagnosis_run_uid: {agent_run_uid}\n'
        f'source_run_uid: {runtime.context.source_run_uid}\n'
        f'workflow_uid: {runtime.context.workflow_uid or "null"}\n'
        f'project_id: {runtime.context.project_id}\n'
        '---\n\n'
    )
    try:
        temporary_path.write_text(f'{metadata}{report}\n', encoding='utf-8')
        temporary_path.replace(report_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    relative_path = report_path.relative_to(project_root).as_posix()
    return Command(
        update={
            'diagnosis_path': relative_path,
            'source_run_uid': runtime.context.source_run_uid,
            'workflow_uid': runtime.context.workflow_uid,
            'messages': [
                ToolMessage(
                    content=f'The diagnosis report has been saved to {relative_path}.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
