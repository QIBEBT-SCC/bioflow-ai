"""Evidence readers and report output for diagnosing one workflow execution."""

import re
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

import rustworkx as rx
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlalchemy.orm import Session

import app.crud.run_instance as run_instance_crud
import app.crud.workflow as workflow_crud
from app.core.session import engine
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.workflow_config import Status
from llm.context import WorkflowDiagnosisContext
from llm.utils.formatters import DEFAULT_FORMAT_MAX_CHARS, format_json, format_text

MAX_SEARCHED_BYTES = 10 * 1024 * 1024
MAX_SEARCH_FILE_BYTES = 2 * 1024 * 1024
MAX_SEARCH_MATCHES = 100


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


def resolve_run_path(run_root: str, relative_path: str = '.') -> Path:
    """Resolve a user-supplied relative path without escaping the run directory."""
    root = Path(run_root).expanduser().resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        message = f'Path resolves outside the workflow run directory: {relative_path}'
        raise ValueError(message) from error
    return candidate


def _read_text(path: Path, mode: Literal['head', 'tail', 'all'], max_chars: int) -> str:
    max_chars = max(1_000, min(max_chars, DEFAULT_FORMAT_MAX_CHARS))
    with path.open('rb') as file:
        if mode == 'tail':
            file.seek(0, 2)
            size = file.tell()
            file.seek(max(0, size - max_chars * 4))
        raw = file.read(max_chars * 4 + 1)
    if b'\x00' in raw[:8_192]:
        return f'[binary file: {path.name}, {path.stat().st_size} bytes]'
    text = raw.decode('utf-8', errors='replace')
    return text[-max_chars:] if mode == 'tail' else text[:max_chars]


def _tree(root: Path, max_entries: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob('*')):
        if len(entries) >= max_entries:
            entries.append({'path': '...', 'note': 'listing truncated'})
            break
        if path.is_symlink():
            kind = 'symlink'
        elif path.is_dir():
            kind = 'directory'
        elif path.is_file():
            kind = 'file'
        else:
            continue
        item: dict[str, Any] = {'path': str(path.relative_to(root)), 'type': kind}
        if kind == 'file':
            item['size'] = path.stat().st_size
        entries.append(item)
    return entries


def _search_files(root: Path, query: str, glob: str, max_matches: int) -> dict[str, Any]:
    pattern = re.compile(query, re.IGNORECASE)
    matches: list[dict[str, Any]] = []
    searched_bytes = 0
    skipped_files = 0
    for path in sorted(root.glob(glob)):
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        if size > MAX_SEARCH_FILE_BYTES or searched_bytes + size > MAX_SEARCHED_BYTES:
            skipped_files += 1
            continue
        raw = path.read_bytes()
        searched_bytes += len(raw)
        if b'\x00' in raw[:8_192]:
            skipped_files += 1
            continue
        for line_number, line in enumerate(raw.decode('utf-8', errors='replace').splitlines(), start=1):
            if pattern.search(line):
                matches.append(
                    {
                        'path': str(path.relative_to(root)),
                        'line': line_number,
                        'text': line[:1_000],
                    }
                )
                if len(matches) >= max_matches:
                    return {
                        'matches': matches,
                        'searched_bytes': searched_bytes,
                        'skipped_files': skipped_files,
                        'truncated': True,
                    }
    return {
        'matches': matches,
        'searched_bytes': searched_bytes,
        'skipped_files': skipped_files,
        'truncated': False,
    }


@tool('run_file_reader', parse_docstring=True)
def run_file_reader(
    action: Literal['tree', 'read', 'search'],
    runtime: ToolRuntime[WorkflowDiagnosisContext],
    path: str = '.',
    query: str | None = None,
    glob: str = '**/*',
    mode: Literal['head', 'tail', 'all'] = 'tail',
    max_chars: int = 20_000,
) -> str:
    """Inspect files produced by the current workflow run without modifying them.

    Args:
        action: ``tree`` lists files, ``read`` reads one text file, and ``search``
            finds a regular expression across text files.
        path: A path relative to the run directory. Use ``.`` for the run root.
        query: Regular expression required by ``search``.
        glob: Relative glob used by ``search``. Defaults to every descendant.
        mode: For ``read``, return the beginning, end, or all bounded content.
        max_chars: Maximum characters returned by ``read``.
    """
    run_root = runtime.context.run_base_dir
    if not run_root:
        return format_json({'error': 'This workflow execution has no run directory.'})
    root = resolve_run_path(run_root)
    target = resolve_run_path(run_root, path)
    if not root.exists():
        return format_json({'error': 'The workflow run directory no longer exists.', 'run_root': str(root)})
    runtime.stream_writer({'info': f'Inspecting run files: {action} {path}'})
    if action == 'tree':
        if not target.is_dir():
            return format_json({'error': 'The requested tree path is not a directory.', 'path': path})
        return format_json({'root': str(root), 'path': path, 'entries': _tree(target, 500)})
    if action == 'read':
        if not target.is_file():
            return format_json({'error': 'The requested path is not a file.', 'path': path})
        content = f'# {path}\n{_read_text(target, mode, max_chars)}'
        return format_text(content)
    if not query:
        return format_json({'error': 'query is required for search.'})
    if not target.is_dir():
        return format_json({'error': 'The requested search path is not a directory.', 'path': path})
    try:
        return format_json(_search_files(target, query, glob, min(max(max_chars // 200, 10), MAX_SEARCH_MATCHES)))
    except re.error as error:
        return format_json({'error': f'Invalid regular expression: {error}'})


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
