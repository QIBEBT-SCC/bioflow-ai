"""Read-only workflow execution evidence and optional diagnosis report output."""

import math
import re
import shlex
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import bioflow_ai.crud.project_file_mapping as project_file_mapping_crud
import bioflow_ai.crud.resource.sample as sample_crud
from bioflow_ai.crud.tools import tool as tool_crud
from bioflow_ai.infrastructure.containers import ContainerMount, run_command
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.context import WorkflowAssistantContext
from bioflow_ai.models.workflow import NodeRun, WorkflowRun
from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.schemas.tool_config import ImageConfig
from bioflow_ai.schemas.workflow_run import NodeRunStatus, RunStatus
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


def _topology_levels(workflow: WorkflowDefinition) -> dict[str, int]:
    """Calculate presentation-only diagnostic levels without mirroring runtime scheduling."""
    node_ids = [node.id for node in workflow.nodes]
    indegree = dict.fromkeys(node_ids, 0)
    children: dict[str, list[str]] = defaultdict(list)
    levels = dict.fromkeys(node_ids, 0)
    for edge in workflow.edges:
        if edge.source not in indegree or edge.target not in indegree:
            continue
        children[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    while queue:
        node_id = queue.popleft()
        for child in children[node_id]:
            levels[child] = max(levels[child], levels[node_id] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return levels


def execution_labels(
    workflow: WorkflowDefinition,
    statuses: dict[str, NodeRunStatus],
    positions: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Describe each node from durable runtime state, not Celery backend state."""
    levels = _topology_levels(workflow)
    result: dict[str, dict[str, Any]] = {}
    for node in workflow.nodes:
        node_id = node.id
        status = statuses.get(node_id)
        if status is None:
            reason = 'presentation_only'
        elif status == NodeRunStatus.FAILED:
            reason = 'direct_failed'
        elif status == NodeRunStatus.SUCCEEDED:
            reason = 'completed'
        elif status == NodeRunStatus.RUNNING:
            reason = 'running'
        elif status == NodeRunStatus.BLOCKED:
            reason = 'blocked_by_dependency'
        elif status == NodeRunStatus.QUEUED:
            reason = 'queued'
        elif status == NodeRunStatus.READY:
            reason = 'ready'
        else:
            reason = 'waiting_for_dependencies'
        result[node_id] = {
            'dispatch_order': positions.get(node_id),
            'topology_level': levels.get(node_id),
            'execution_reason': reason,
        }
    return result


def _compact_node_data(data: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in data.items():
        if key in {'code', 'content', 'script'} and isinstance(value, str):
            compact[key] = {'characters': len(value), 'preview': value[:500]}
        else:
            compact[key] = value
    return compact


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
    failed_only: bool = False,
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= MAX_CHILD_PAGE:
        return {'error': 'offset must be nonnegative and limit must be between 1 and 100.'}
    try:
        with Session(engine) as session:
            run = _selected_run(session, context, child_run_uid)
            path = tuple(subgraph_path or ())
            try:
                graph, _ = enter_subgraphs(run.workflow, list(path))
            except SubgraphError as error:
                return {'error': str(error)}
            node_runs = list(session.scalars(select(NodeRun).where(NodeRun.run_uid == run.uid)))
            by_id = {node.definition_node_id: node for node in node_runs}
            if action == 'children':
                if not include_node_id:
                    return {'error': 'node_id is required for action=children.'}
                control = by_id.get(definition_id((*path, include_node_id)))
                if control is None or control.node_type != 'foreach':
                    return {'error': 'No ForEach control task exists for this node.'}
                filters = [WorkflowRun.parent_node_uid == control.uid]
                if failed_only:
                    filters.append(WorkflowRun.status == RunStatus.FAILED)
                children = list(
                    session.scalars(
                        select(WorkflowRun).where(*filters).order_by(WorkflowRun.item_index).offset(offset).limit(limit)
                    )
                )
                return {
                    'parent_run_uid': str(run.uid),
                    'parent_generation': run.generation,
                    'subgraph_path': list(path),
                    'node_id': include_node_id,
                    'total': session.scalar(select(func.count()).select_from(WorkflowRun).where(*filters)) or 0,
                    'offset': offset,
                    'children': [
                        {
                            'run_uid': str(item.uid),
                            'item_index': item.item_index,
                            'item_name': item.item_name,
                            'generation': item.generation,
                            'status': item.status.value,
                            'base_dir': item.base_dir,
                        }
                        for item in children
                    ],
                }
            statuses = {node.definition_node_id: node.status for node in node_runs}
            positions = {node.definition_node_id: node.position_index for node in node_runs}
            parents, descendants = _parents_and_children(graph)
            local_statuses = {
                node.id: statuses[definition_id((*path, node.id))]
                for node in graph.nodes
                if definition_id((*path, node.id)) in statuses
            }
            local_positions = {
                node.id: positions[definition_id((*path, node.id))]
                for node in graph.nodes
                if definition_id((*path, node.id)) in positions
            }
            schedule = execution_labels(graph, local_statuses, local_positions)
            nodes = []
            for node in graph.nodes:
                if include_node_id and node.id != include_node_id:
                    continue
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
                payload: dict[str, Any] = {
                    'node_id': node.id,
                    'template_node_id': node.id if not node.id.startswith('@foreach/input/') else None,
                    'runtime_node_id': runtime_id,
                    'subgraph_path': list(path),
                    'type': node.type,
                    'status': task_status,
                    'upstream_node_ids': sorted(parents.get(node.id, ())),
                    'downstream_node_ids': sorted(descendants.get(node.id, ())),
                    'schedule': schedule.get(node.id),
                }
                if node.type == 'subgraph' and payload['schedule'] is not None:
                    payload['schedule']['execution_reason'] = task_status
                if node.type == 'subgraph':
                    payload['child_path'] = [*path, node.id]
                    payload['source_uid'] = node.data.get('source_uid')
                    child_interface = WorkflowDefinition.model_validate(node.data['workflow']).interface
                    payload['input_ports'] = [port.id for port in child_interface.inputs] if child_interface else []
                    payload['output_ports'] = [port.id for port in child_interface.outputs] if child_interface else []
                    payload['batch'] = task is not None and task.node_type == 'foreach'
                if include_node_id:
                    payload['data'] = (
                        {key: value for key, value in node.data.items() if key != 'workflow'}
                        if node.type == 'subgraph'
                        else node.data
                    )
                    payload['task'] = (
                        {
                            'node_run_uid': str(task.uid),
                            'status': task.status.value,
                            'base_dir': task.base_dir,
                            'commands': task.commands,
                            'error_message': task.error_message,
                            'tool_output': task.tool_output.model_dump(mode='json') if task.tool_output else None,
                        }
                        if task
                        else None
                    )
                    if node.type == 'tool':
                        tool_uid = task.tool_uid if task is not None else node.data.get('tool_uid')
                        tool_info = tool_crud.get_summary(session, tool_uid) if tool_uid else None
                        payload['tool'] = {
                            'uid': str(tool_uid) if tool_uid else None,
                            'name': getattr(tool_info, 'name', None),
                            'description': getattr(tool_info, 'description', None),
                        }
                nodes.append(payload)
            node_total = len(nodes)
            failed_node_ids = [node['node_id'] for node in nodes if node['status'] == 'failed']
            if action == 'overview':
                listed_nodes = [node for node in nodes if node['status'] == 'failed'] if failed_only else nodes
                listed_total = len(listed_nodes)
                nodes = listed_nodes[offset : offset + limit]
                shown_ids = {node['node_id'] for node in nodes}
                visible_edges = [
                    edge.model_dump(mode='json')
                    for edge in graph.edges
                    if edge.source in shown_ids or edge.target in shown_ids
                ]
            else:
                listed_total = node_total
                visible_edges = None
            return {
                'run': {
                    'uid': str(run.uid),
                    'generation': run.generation,
                    'name': run.name,
                    'workflow_uid': str(run.workflow_uid) if run.workflow_uid else None,
                    'status': run.status.value,
                    'base_dir': run.base_dir,
                    'item_index': run.item_index,
                    'item_name': run.item_name,
                },
                'subgraph_path': list(path),
                'nodes': nodes,
                'node_total': node_total,
                'listed_total': listed_total,
                'failed_node_ids': failed_node_ids[:MAX_CHILD_PAGE],
                'failed_node_count': len(failed_node_ids),
                'offset': offset if action == 'overview' else None,
                'next_offset': offset + limit if action == 'overview' and offset + limit < listed_total else None,
                'edges': visible_edges,
                'edge_total': len(graph.edges),
                **({'error': f'Node not found: {include_node_id}'} if include_node_id and not nodes else {}),
            }
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
    failed_only: bool = False,
) -> str:
    """Read the run's immutable workflow snapshot and exact task execution mapping.

    Args:
        action: ``overview`` returns all nodes, edges, task states and scheduler order;
            ``node`` returns complete data and evidence for one local node ID;
            ``children`` lists the ForEach child runs belonging to a node.
        node_id: Node ID local to ``subgraph_path``; required for ``node`` and ``children``.
        subgraph_path: Nested subgraph node IDs to enter in the selected run.
        child_run_uid: ForEach child run UID returned by ``children``.
        offset: First node or child item to list.
        limit: Maximum nodes or child items to list (1 to 100).
        failed_only: Only list failed nodes or ForEach child runs.
    """
    if action == 'node' and not node_id:
        return format_json({'error': 'node_id is required for action=node.'})
    runtime.stream_writer({'info': f'Reading workflow execution: {action}'})
    return format_json(
        _workflow_payload(
            runtime.context, action, node_id, subgraph_path, child_run_uid, offset, limit, failed_only=failed_only
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

    runtime.stream_writer({'info': f'Inspecting read-only evidence: {source}'})
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
