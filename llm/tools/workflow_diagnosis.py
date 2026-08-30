"""Evidence readers and report output for diagnosing one workflow execution."""

import math
import re
import shlex
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import rustworkx as rx
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlalchemy.orm import Session

import app.crud.project_file_mapping as project_file_mapping_crud
import app.crud.resource.sample as sample_crud
import app.crud.run_instance as run_instance_crud
import app.crud.workflow as workflow_crud
from app.core.session import engine
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.tool_config import ImageConfig
from app.schemas.workflow_config import Status
from app.utils.container import ContainerMount, run_command
from llm.context import WorkflowDiagnosisContext
from llm.utils.formatters import format_json, format_text

EVIDENCE_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-seq:1.0.0')
EVIDENCE_MOUNT_PATH = '/app/data/input'
EVIDENCE_WORKDIR = '/app/data'
MAX_COMMAND_CHARS = 8_000
COMMAND_TIMEOUT_SECONDS = 60
COMMAND_KILL_GRACE_SECONDS = 5
MAX_CAPTURE_BYTES = 64 * 1024
MAX_RETURN_CHARS = 40_000
RETURN_HEAD_CHARS = 30_000
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


def _has_failed_ancestor(node_id: str, parents: dict[str, set[str]], failed: set[str]) -> bool:
    queue = deque(parents.get(node_id, ()))
    visited: set[str] = set()
    while queue:
        parent = queue.popleft()
        if parent in failed:
            return True
        if parent in visited:
            continue
        visited.add(parent)
        queue.extend(parents.get(parent, ()))
    return False


def _scheduler_groups(workflow: WorkflowDefinition) -> list[list[str]]:
    """Mirror the runtime scheduler's rustworkx layer ordering without task imports."""
    graph = rx.PyDiGraph()
    node_indices: dict[str, int] = {}
    for edge in workflow.edges:
        if edge.source not in node_indices:
            node_indices[edge.source] = graph.add_node(edge.source)
        if edge.target not in node_indices:
            node_indices[edge.target] = graph.add_node(edge.target)
        graph.add_edge(node_indices[edge.source], node_indices[edge.target], None)
    first_layer = [
        index for index in graph.node_indices() if graph.in_degree(index) == 0 and graph.out_degree(index) > 0
    ]
    return [[str(node_id) for node_id in layer] for layer in rx.layers(graph, first_layer)]


def execution_labels(workflow: WorkflowDefinition, statuses: dict[str, Status]) -> dict[str, dict[str, Any]]:
    """Describe scheduler order and why each executable node did or did not run."""
    groups = _scheduler_groups(workflow)
    topology_order = [node_id for group in groups for node_id in group]
    order = [node_id for node_id in topology_order if node_id in statuses]
    levels = {node_id: level for level, group in enumerate(groups) for node_id in group}
    positions = {node_id: index for index, node_id in enumerate(order)}
    failed = {node_id for node_id, status in statuses.items() if status == Status.ERROR}
    first_failed_position = min((positions[node_id] for node_id in failed if node_id in positions), default=None)
    parents, _ = _parents_and_children(workflow)
    result: dict[str, dict[str, Any]] = {}
    for node_id in topology_order:
        status = statuses.get(node_id)
        position = positions.get(node_id)
        if status is None:
            reason = 'no_task_instance'
        elif status == Status.ERROR:
            reason = 'direct_failed'
        elif status == Status.SUCCESS:
            reason = 'completed'
        elif status == Status.RUNNING:
            reason = 'running'
        elif _has_failed_ancestor(node_id, parents, failed):
            reason = 'blocked_by_dependency'
        elif first_failed_position is not None and position is not None and position > first_failed_position:
            reason = 'not_run_due_chain_abort'
        else:
            reason = 'waiting_or_not_scheduled'
        result[node_id] = {
            'dispatch_order': position + 1 if position is not None else None,
            'topology_level': levels[node_id],
            'execution_reason': reason,
        }
    return result


def _celery_failure(task_uid: str) -> dict[str, Any] | None:
    try:
        from script.worker import celery_app

        result = celery_app.AsyncResult(task_uid)
        if not result.failed():
            return {'state': result.state}
        return {
            'state': result.state,
            'error': format_text(str(result.result), max_chars=8_000),
            'traceback': format_text(str(result.traceback or ''), max_chars=20_000),
        }
    except Exception as error:
        return {'unavailable': f'{type(error).__name__}: {error}'}


def _compact_node_data(data: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in data.items():
        if key in {'code', 'content', 'script'} and isinstance(value, str):
            compact[key] = {'characters': len(value), 'preview': value[:500]}
        else:
            compact[key] = value
    return compact


def _template_node_ids(workflow: WorkflowDefinition, workflow_uid: uuid.UUID | None) -> dict[str, str]:
    if workflow_uid is None:
        return {}
    with Session(engine) as session:
        template = workflow_crud.get(session, workflow_uid)
        if template is None:
            return {}
        template_nodes = template.workflow.nodes
    if len(template_nodes) != len(workflow.nodes):
        return {}
    if any(current.type != original.type for current, original in zip(workflow.nodes, template_nodes, strict=True)):
        return {}
    return {current.id: original.id for current, original in zip(workflow.nodes, template_nodes, strict=True)}


def _workflow_payload(source_run_uid: str, include_node_id: str | None = None) -> dict[str, Any]:
    with Session(engine) as session:
        run = run_instance_crud.get(session, source_run_uid)
        if run is None:
            return {'error': 'Workflow execution not found.'}
        workflow = run.workflow
        tasks = list(run.tasks)
        run_payload = {
            'uid': str(run.uid),
            'name': run.name,
            'workflow_uid': str(run.workflow_uid) if run.workflow_uid else None,
            'project_id': run.project_id,
            'status': run.status.name,
            'base_dir': run.base_dir,
            'create_time': run.create_time,
            'start_time': run.start_time,
            'end_time': run.end_time,
        }
        task_payloads = {
            str(task.uid): {
                'task_uid': str(task.uid),
                'name': task.name,
                'tool_uid': str(task.tool_uid) if task.tool_uid else None,
                'status': task.status.name,
                'base_dir': task.base_dir,
                'commands': task.commands,
                'tool_output': task.tool_output.model_dump(mode='json') if task.tool_output else None,
                'create_time': task.create_time,
                'start_time': task.start_time,
                'end_time': task.end_time,
            }
            for task in tasks
        }
        statuses = {str(task.uid): task.status for task in tasks}
        workflow_uid = run.workflow_uid

    scheduler_error: str | None = None
    try:
        schedule = execution_labels(workflow, statuses)
    except Exception as error:
        schedule = {}
        scheduler_error = f'{type(error).__name__}: {error}'
    template_ids = _template_node_ids(workflow, workflow_uid)
    parents, children = _parents_and_children(workflow)
    requested_runtime_id = include_node_id
    if include_node_id and include_node_id not in {node.id for node in workflow.nodes}:
        requested_runtime_id = next(
            (runtime_id for runtime_id, template_id in template_ids.items() if template_id == include_node_id),
            include_node_id,
        )

    nodes: list[dict[str, Any]] = []
    for node in workflow.nodes:
        if requested_runtime_id and node.id != requested_runtime_id:
            continue
        task = task_payloads.get(node.id)
        if task and task['status'] == Status.ERROR.name:
            task['celery'] = _celery_failure(node.id)
        nodes.append(
            {
                'runtime_node_id': node.id,
                'template_node_id': template_ids.get(node.id),
                'template_mapping': 'position_and_type' if node.id in template_ids else None,
                'type': node.type,
                'data': node.data if requested_runtime_id else _compact_node_data(node.data),
                'task': task,
                'schedule': schedule.get(node.id),
                'upstream_runtime_node_ids': sorted(parents.get(node.id, ())),
                'downstream_runtime_node_ids': sorted(children.get(node.id, ())),
            }
        )
    mapped = {node.id for node in workflow.nodes}
    return {
        'run': run_payload,
        'scheduler_error': scheduler_error,
        'nodes': nodes,
        'edges': [edge.model_dump(mode='json') for edge in workflow.edges] if not requested_runtime_id else None,
        'unmapped_tasks': [payload for uid, payload in task_payloads.items() if uid not in mapped],
        **({'error': f'Node not found: {include_node_id}'} if include_node_id and not nodes else {}),
    }


@tool('workflow_reader', parse_docstring=True)
def workflow_reader(
    action: Literal['overview', 'node'],
    runtime: ToolRuntime[WorkflowDiagnosisContext],
    node_id: str | None = None,
) -> str:
    """Read the run's immutable workflow snapshot and exact task execution mapping.

    Args:
        action: ``overview`` returns all nodes, edges, task states and scheduler order;
            ``node`` returns complete data and evidence for one runtime or template node id.
        node_id: Runtime node id or inferred template node id, required for ``node``.
    """
    if action == 'node' and not node_id:
        return format_json({'error': 'node_id is required for action=node.'})
    runtime.stream_writer({'info': f'Reading workflow execution: {action}'})
    return format_json(_workflow_payload(runtime.context.source_run_uid, node_id if action == 'node' else None))


# ============ Run Command =============


class EvidenceSourceError(ValueError):
    """A recoverable error while resolving an evidence source."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _evidence_error(code: str, message: str) -> EvidenceSourceError:
    return EvidenceSourceError(code, message)


@dataclass(frozen=True)
class ResolvedEvidence:
    """A validated host resource that may be mounted into the evidence container."""

    source: str
    path: Path
    kind: Literal['file', 'directory']


def _error(code: str, message: str) -> str:
    return format_json({'ok': False, 'error': {'code': code, 'message': message}}, max_chars=None)


def _validated_path(raw_path: str | None, source: str) -> tuple[Path, Literal['file', 'directory']]:
    if not raw_path:
        raise _evidence_error('missing_path', f'Evidence source `{source}` has no filesystem path.')
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise _evidence_error('path_not_found', f'Evidence source `{source}` no longer exists.') from error
    if path.is_file():
        return path, 'file'
    if path.is_dir():
        return path, 'directory'
    raise _evidence_error('unsupported_path', f'Evidence source `{source}` is not a regular file or directory.')


def _current_run(session: Session, context: WorkflowDiagnosisContext) -> Any:
    run = run_instance_crud.get(session, context.source_run_uid)
    if run is None:
        raise _evidence_error('run_not_found', 'The workflow execution no longer exists.')
    if run.owner_id != context.user_id:
        raise _evidence_error('access_denied', 'The workflow execution does not belong to the current user.')
    if run.project_id != context.project_id:
        raise _evidence_error('project_mismatch', 'The workflow execution does not belong to the current project.')
    return run


def resolve_evidence_source(source: str, context: WorkflowDiagnosisContext) -> ResolvedEvidence:
    """Resolve one logical evidence reference without accepting arbitrary host paths."""
    with Session(engine) as session:
        run = _current_run(session, context)

        if source == 'run:current':
            path, kind = _validated_path(run.base_dir, source)
            if kind != 'directory':
                raise _evidence_error('invalid_run_path', 'The current run path is not a directory.')
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('sample:'):
            tag = source.removeprefix('sample:')
            if not tag:
                raise _evidence_error('invalid_source', 'A sample source must include a tag.')
            if run.sample_uid is None:
                raise _evidence_error(
                    'sample_unavailable',
                    'The current workflow execution is project-level and has no bound sample.',
                )
            sample = sample_crud.get_with_files(session, run.sample_uid)
            if sample is None:
                raise _evidence_error('sample_not_found', 'The sample bound to the current run no longer exists.')
            if sample.owner_id != context.user_id or sample.project_id != context.project_id:
                raise _evidence_error('access_denied', 'The current run sample is outside the allowed project scope.')
            sample_file = next((item for item in sample.sample_files if item.tag == tag), None)
            if sample_file is None:
                raise _evidence_error('reference_not_found', f'Sample reference `{source}` was not found.')
            path, kind = _validated_path(sample_file.file_path, source)
            return ResolvedEvidence(source=source, path=path, kind=kind)

        if source.startswith('proj:'):
            keyword = source.removeprefix('proj:')
            if not keyword:
                raise _evidence_error('invalid_source', 'A project source must include a keyword.')
            mapping = project_file_mapping_crud.get_by_keyword(session, context.project_id, keyword)
            if mapping is None:
                raise _evidence_error('reference_not_found', f'Project reference `{source}` was not found.')
            if mapping.project_id != context.project_id:
                raise _evidence_error('project_mismatch', 'The project reference is outside the allowed project scope.')
            path, kind = _validated_path(mapping.file_path, source)
            return ResolvedEvidence(source=source, path=path, kind=kind)

    raise _evidence_error(
        'invalid_source',
        'source must be exactly `run:current`, `sample:<tag>`, or `proj:<keyword>`.',
    )


def build_bounded_command(command: str) -> str:
    """Wrap an inspection command with time and captured-output limits inside the container."""
    stripped = command.strip()
    if not stripped:
        raise _evidence_error('empty_command', 'The evidence inspection command cannot be empty.')
    if len(stripped) > MAX_COMMAND_CHARS:
        raise _evidence_error(
            'command_too_long',
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
    runtime: ToolRuntime[WorkflowDiagnosisContext],
) -> str:
    """Run a bounded shell inspection against one read-only workflow evidence source.

    The source is mounted at ``/app/data/input``. It may be a file or a directory, so
    inspect it with ``file /app/data/input`` when the type is not already known.

    Args:
        source: Exactly ``run:current``, ``sample:<tag>``, or ``proj:<keyword>``.
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
def write_diagnosis(content: str, runtime: ToolRuntime[WorkflowDiagnosisContext]) -> Command:
    """Write the completed workflow diagnosis to ``DIAGNOSIS.md``.

    Args:
        content: Complete Markdown diagnosis report in the language required by the system prompt, including the
            machine-readable summary.
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
            'diagnosis_written': True,
            'source_run_uid': runtime.context.source_run_uid,
            'workflow_uid': runtime.context.workflow_uid,
            'updated': True,
            'messages': [
                ToolMessage(
                    content=f'The diagnosis report has been saved to {relative_path}.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
