"""Creation, rerun and read models for durable workflow runs."""

import io
import shutil
import uuid
import zipfile
from collections import Counter
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from bioflow_ai.core.config import get_settings
from bioflow_ai.crud import node_run as node_run_crud
from bioflow_ai.crud import project_file_mapping as project_file_mapping_crud
from bioflow_ai.crud import user as user_crud
from bioflow_ai.crud import workflow_run as run_crud
from bioflow_ai.crud.resource import sample as sample_crud
from bioflow_ai.crud.tools import tool as tool_crud
from bioflow_ai.models.resource import Sample
from bioflow_ai.models.workflow import NodeRun, Workflow, WorkflowRun
from bioflow_ai.schemas.project_workflow import WorkflowRunResult
from bioflow_ai.schemas.react_flow import WorkflowDefinition
from bioflow_ai.schemas.user import CurrentUser
from bioflow_ai.schemas.workflow_run import (
    NodeRunData,
    NodeRunPublic,
    NodeRunRecord,
    NodeRunStatistics,
    NodeRunStatus,
    PaginatedNodeRuns,
    PaginatedWorkflowRuns,
    RunStatus,
    WorkflowRunPublic,
    WorkflowRunRecord,
    WorkflowRunStatistics,
)
from bioflow_ai.services.workflow.batch_graph import lower_batch_subgraphs
from bioflow_ai.services.workflow.compiler import compile_run
from bioflow_ai.services.workflow.subgraphs import flatten_workflow, inject_run_data
from bioflow_ai.utils.file.paths import to_safe_path_segment
from bioflow_ai.utils.system import random_uid, utc_now

STORAGE_PATH = get_settings().persistent.storage.STORAGE_PATH
_NO_RUN = 'No run exists for this workflow'


class WorkflowRunConflictError(RuntimeError):
    """A run cannot be restarted until its active generation settles."""


class WorkflowRunNotFoundError(RuntimeError):
    """No current generation exists for a requested workflow package."""


def _project_output_root(project_id: int, workflow: Workflow, sample: Sample | None) -> Path:
    segment = f'{to_safe_path_segment(workflow.name)}-{workflow.uid}'
    root = Path(STORAGE_PATH).absolute() / 'proj' / str(project_id) / 'runs' / segment
    if sample is None:
        return root / 'project'
    sample_segment = f'{to_safe_path_segment(sample.sample_name)}-{sample.uid}'
    return root / sample_segment


def _clear_generation(session: Session, run: WorkflowRun) -> None:
    node_run_crud.delete_for_run(session, run.uid)
    session.flush()


def create_editor_run(
    session: Session,
    workflow: WorkflowDefinition,
    template_name: str,
    current_user: CurrentUser,
) -> WorkflowRun:
    """Create a fresh ad-hoc run; graph IDs remain unchanged."""
    run_uid = random_uid()
    output_root = (
        Path(STORAGE_PATH).absolute()
        / f'{current_user.username}({current_user.id})'
        / 'temporarily'
        / to_safe_path_segment(template_name)
        / str(run_uid)
    )
    run = WorkflowRun(
        uid=run_uid,
        generation=1,
        name=f'temporarily-{template_name}',
        owner_id=current_user.id,
        base_dir=str(output_root / 'generation-1'),
        workflow=workflow,
    )
    run_crud.insert(session, run)
    session.flush()
    compile_run(session, run, sample_name=None)
    session.commit()
    return run


def _create_or_rerun_project_run(
    session: Session,
    *,
    project_id: int,
    project_name: str,
    workflow: Workflow,
    current_user: CurrentUser,
    sample: Sample | None,
    run_name_prefix: str | None,
) -> tuple[WorkflowRun, str | None]:
    sample_uid = sample.uid if sample else None
    sample_name = sample.sample_name if sample else None
    existing = run_crud.find_project_run(session, project_id, workflow.uid, sample_uid)
    output_root = _project_output_root(project_id, workflow, sample)
    old_base_dir: str | None = None

    if existing is not None:
        if not existing.settled:
            label = sample_name or project_name
            message = f'{label} already has an unsettled workflow run'
            raise WorkflowRunConflictError(message)
        old_base_dir = existing.base_dir
        _clear_generation(session, existing)
        existing.generation += 1
        existing.base_dir = str(output_root / f'generation-{existing.generation}')
        existing.workflow = workflow.workflow.model_copy(deep=True)
        existing.status = RunStatus.PENDING
        existing.settled = False
        existing.start_time = None
        existing.end_time = None
        run = existing
    else:
        run_uid = random_uid()
        label = sample_name or project_name
        run_name = f'{run_name_prefix}-{label}' if run_name_prefix else f'{project_name}-{workflow.name}-{label}'
        run = WorkflowRun(
            uid=run_uid,
            generation=1,
            name=run_name,
            owner_id=current_user.id,
            project_id=project_id,
            workflow_uid=workflow.uid,
            sample_uid=sample_uid,
            base_dir=str(output_root / 'generation-1'),
            workflow=workflow.workflow.model_copy(deep=True),
        )
        run_crud.insert(session, run)
        session.flush()

    if sample is None:
        project_file_mapping_crud.delete_dynamic_for_workflow(
            session,
            project_id=project_id,
            workflow_uid=workflow.uid,
        )
    else:
        sample_crud.delete_dynamic_files_for_workflow(
            session,
            sample_uid=sample.uid,
            workflow_uid=workflow.uid,
        )
    compile_run(session, run, sample_name=sample_name)
    return run, old_base_dir


def create_project_runs(
    session: Session,
    *,
    project_id: int,
    project_name: str,
    workflow: Workflow,
    samples: Sequence[Sample | None],
    current_user: CurrentUser,
    run_name_prefix: str | None,
) -> WorkflowRunResult:
    """Atomically create or rerun all requested sample/project generations."""
    old_output_dirs: list[str] = []
    workflow_runs: list[WorkflowRun] = []
    for sample in samples:
        run, old_base_dir = _create_or_rerun_project_run(
            session,
            project_id=project_id,
            project_name=project_name,
            workflow=workflow,
            current_user=current_user,
            sample=sample,
            run_name_prefix=run_name_prefix,
        )
        workflow_runs.append(run)
        if old_base_dir:
            old_output_dirs.append(old_base_dir)
    session.commit()

    for old_output_dir in old_output_dirs:
        old_path = Path(old_output_dir)
        if old_path.exists():
            shutil.rmtree(old_path)
    return WorkflowRunResult(run_uids=[run.uid for run in workflow_runs], count=len(workflow_runs))


def file_still_written(nodes: Sequence[NodeRun]) -> bool:
    """Whether a file under these nodes' directories may still change or belongs to an unfinished node.

    Successive pass-through Tools share one workspace. There only a running node blocks previews, so a failed
    node's retained scene and its predecessors' results stay inspectable.
    """
    writers = Counter(node.base_dir for node in nodes)
    return any(
        node.status == NodeRunStatus.RUNNING if writers[node.base_dir] > 1 else node.status != NodeRunStatus.SUCCEEDED
        for node in nodes
    )


def _to_node_record(node: NodeRun, run: WorkflowRun, owner_username: str) -> NodeRunRecord:
    return NodeRunRecord(
        uid=node.uid,
        run_uid=node.run_uid,
        run_name=run.name,
        generation=run.generation,
        definition_node_id=node.definition_node_id,
        node_type=node.node_type,
        name=node.name,
        status=node.status,
        owner_id=run.owner_id,
        owner_username=owner_username,
        project_id=run.project_id,
        workflow_uid=run.workflow_uid,
        sample_uid=run.sample_uid,
        parent_node_uid=run.parent_node_uid,
        item_name=run.item_name,
        create_time=node.create_time,
        start_time=node.start_time,
        end_time=node.end_time,
    )


def to_node_public(session: Session, node: NodeRun) -> NodeRunPublic:
    """Build task-center details for one current-generation node execution."""
    run = run_crud.get(session, node.run_uid)
    if run is None:
        raise ValueError

    owner = user_crud.get_by_id(session, run.owner_id)
    if owner is None:
        raise ValueError
    tool = tool_crud.get_summary(session, node.tool_uid) if node.tool_uid is not None else None
    record = _to_node_record(node, run, owner.username)
    return NodeRunPublic(
        **record.model_dump(),
        track_progress=node.track_progress,
        tool_uid=node.tool_uid,
        tool_name=tool.name if tool is not None else None,
        tool_description=tool.description if tool is not None else None,
        base_dir=node.base_dir,
        commands=node.commands,
        error_message=node.error_message,
        tool_output=node.tool_output,
        system=node.system,
        hostname=node.hostname,
        worker_id=node.worker_id,
        queued_at=node.queued_at,
        heartbeat_at=node.heartbeat_at,
    )


def list_node_runs(
    session: Session,
    *,
    owner_id: int | None,
    project_id: int | None = None,
    status: NodeRunStatus | None = None,
    offset: int = 0,
    limit: int = 20,
) -> PaginatedNodeRuns:
    """Return current-generation node executions for the task center."""
    rows, total = node_run_crud.get_page(
        session, owner_id=owner_id, project_id=project_id, status=status, offset=offset, limit=limit
    )
    return PaginatedNodeRuns(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[_to_node_record(node, run, owner_username) for node, run, owner_username in rows],
    )


def recent_node_runs(
    session: Session,
    *,
    owner_id: int | None,
    hours: int,
    limit: int = 1000,
) -> list[NodeRunRecord]:
    """Return recent current-generation node executions for the task timeline."""
    rows = node_run_crud.list_recent(session, owner_id=owner_id, since=utc_now() - timedelta(hours=hours), limit=limit)
    return [_to_node_record(node, run, username) for node, run, username in rows]


def list_node_runs_for_run(session: Session, run: WorkflowRun) -> list[NodeRunRecord]:
    """Return current-generation node executions in workflow definition order."""
    owner = user_crud.get_by_id(session, run.owner_id)
    if owner is None:
        raise ValueError
    nodes = node_run_crud.list_for_runs(session, [run.uid])
    return [_to_node_record(node, run, owner.username) for node in nodes]


def _node_statistics(nodes: list[NodeRun]) -> NodeRunStatistics:
    counts = dict.fromkeys(NodeRunStatus, 0)
    for node in nodes:
        counts[node.status] += 1
    return NodeRunStatistics(
        total=len(nodes),
        pending=counts[NodeRunStatus.PENDING],
        ready=counts[NodeRunStatus.READY],
        queued=counts[NodeRunStatus.QUEUED],
        running=counts[NodeRunStatus.RUNNING],
        succeeded=counts[NodeRunStatus.SUCCEEDED],
        failed=counts[NodeRunStatus.FAILED],
        blocked=counts[NodeRunStatus.BLOCKED],
    )


def build_preview_data(run: WorkflowRun, nodes: list[NodeRun]) -> dict[str, dict[str, str | int]]:
    """Project current-generation file outputs onto presentation nodes without reading file contents."""
    node_by_definition = {node.definition_node_id: node for node in nodes}
    preview_by_definition: dict[str, dict[str, str | int]] = {}
    flat_graph = flatten_workflow(lower_batch_subgraphs(run.workflow))
    flat_nodes = {node.id: node for node in flat_graph.nodes}
    for edge in flat_graph.edges:
        target = flat_nodes.get(edge.target)
        if target is None or target.type != 'image_preview':
            continue
        source = node_by_definition.get(edge.source)
        if source is None or source.status not in {
            NodeRunStatus.SUCCEEDED,
            NodeRunStatus.FAILED,
            NodeRunStatus.BLOCKED,
        }:
            continue
        preview: dict[str, str | int] = {'run_uid': str(run.uid), 'generation': run.generation}
        if source.status != NodeRunStatus.SUCCEEDED:
            preview['error'] = 'upstream_failed'
        else:
            output_key = edge.sourceHandle.removeprefix(f'{edge.source}-out-')
            value = source.tool_output.result.get(output_key) if source.tool_output else None
            if value is None:
                preview['error'] = 'missing_path'
            else:
                try:
                    path = Path(value)
                    base = Path(run.base_dir).resolve()
                    resolved = (path if path.is_absolute() else base / path).resolve()
                    if not resolved.is_relative_to(base):
                        preview['error'] = 'outside_run'
                    elif resolved.suffix.lower() not in {
                        '.png',
                        '.jpg',
                        '.jpeg',
                        '.webp',
                        '.gif',
                        '.svg',
                        '.avif',
                        '.bmp',
                    }:
                        preview['error'] = 'unsupported_format'
                    else:
                        preview['path'] = resolved.relative_to(base).as_posix()
                except (OSError, RuntimeError):
                    preview['error'] = 'outside_run'
        preview_by_definition[edge.target] = preview
    return preview_by_definition


def to_public(session: Session, run: WorkflowRun) -> WorkflowRunPublic:
    """Load and assemble one run's public graph."""
    return _public_runs(session, [run])[0]


def _public_runs(session: Session, runs: list[WorkflowRun]) -> list[WorkflowRunPublic]:
    nodes_by_run: dict[uuid.UUID, list[NodeRun]] = {run.uid: [] for run in runs}
    nodes = node_run_crud.list_for_runs(session, list(nodes_by_run))
    for node in nodes:
        nodes_by_run[node.run_uid].append(node)
    item_statistics: dict[str, dict[str, int]] = {}
    child_rows = run_crud.count_children_by_state(session, [node.uid for node in nodes if node.node_type == 'foreach'])
    for parent_uid, status, has_started, settled, count in child_rows:
        counts = item_statistics.setdefault(
            str(parent_uid), {'total': 0, 'pending': 0, 'running': 0, 'succeeded': 0, 'failed': 0}
        )
        counts['total'] += count
        state = status.value if settled else ('running' if has_started else 'pending')
        counts[state] += count
    parents = run_crud.get_parent_details(
        session, [run.parent_node_uid for run in runs if run.parent_node_uid is not None]
    )
    return [_to_public(run, nodes_by_run[run.uid], item_statistics, parents.get(run.parent_node_uid)) for run in runs]


def _to_public(
    run: WorkflowRun,
    nodes: list[NodeRun],
    item_statistics: dict[str, dict[str, int]],
    parent_details: tuple[uuid.UUID, str, str] | None,
) -> WorkflowRunPublic:
    graph_nodes = inject_run_data(
        run.workflow,
        {
            node.definition_node_id: NodeRunData(
                uid=node.uid,
                run_uid=run.uid,
                status=node.status,
                create_time=node.create_time,
                start_time=node.start_time,
                end_time=node.end_time,
                item_statistics=item_statistics.get(str(node.uid)),
            ).model_dump(mode='json')
            for node in nodes
        },
        build_preview_data(run, nodes),
    )
    record = WorkflowRunRecord.model_validate(run).model_dump()
    if parent_details is not None:
        record['parent_run_uid'], record['parent_run_name'], record['parent_node_name'] = parent_details
    return WorkflowRunPublic(
        **record,
        nodes=graph_nodes,
        edges=run.workflow.edges,
        interface=run.workflow.interface,
        node_statistics=_node_statistics(nodes),
    )


def list_runs(
    session: Session,
    *,
    owner_id: int | None,
    project_id: int | None = None,
    status: RunStatus | None = None,
    offset: int = 0,
    limit: int = 20,
) -> PaginatedWorkflowRuns:
    """Return workflow runs for the requested owner and project."""
    runs, total = run_crud.get_page(
        session, owner_id=owner_id, project_id=project_id, status=status, offset=offset, limit=limit
    )
    return PaginatedWorkflowRuns(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=_public_runs(session, runs),
    )


def run_statistics(session: Session, *, owner_id: int | None, project_id: int | None = None) -> WorkflowRunStatistics:
    """Aggregate workflow-run states."""
    counts = run_crud.count_by_status(session, owner_id=owner_id, project_id=project_id)
    return WorkflowRunStatistics(
        total=sum(counts.values()),
        pending=counts.get(RunStatus.PENDING, 0),
        running=counts.get(RunStatus.RUNNING, 0),
        succeeded=counts.get(RunStatus.SUCCEEDED, 0),
        failed=counts.get(RunStatus.FAILED, 0),
    )


def package_project_workflow_runs(
    session: Session,
    *,
    project_id: int,
    workflow_uid: uuid.UUID,
    owner_id: int | None,
) -> bytes:
    """Build a zip containing only current-generation outputs."""
    current_runs = run_crud.list_for_workflow(
        session, project_id=project_id, workflow_uid=workflow_uid, owner_id=owner_id
    )
    if not current_runs:
        raise WorkflowRunNotFoundError(_NO_RUN)

    sample_uids = [run.sample_uid for run in current_runs if run.sample_uid is not None]
    sample_names = {sample.uid: sample.sample_name for sample in sample_crud.get_by_uids(session, sample_uids)}

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        for run in current_runs:
            run_dir = Path(run.base_dir)
            if not run_dir.exists():
                continue
            root_name = (
                f'{to_safe_path_segment(sample_names[run.sample_uid])}-{run.sample_uid}'
                if run.sample_uid
                else 'project'
            )
            for directory, subdirectories, filenames in run_dir.walk():
                subdirectories[:] = [name for name in subdirectories if name != '.venv']
                for filename in filenames:
                    file_path = directory / filename
                    if file_path.is_file():
                        archive.write(file_path, arcname=Path(root_name) / file_path.relative_to(run_dir))
    return buffer.getvalue()
