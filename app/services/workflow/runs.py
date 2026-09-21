"""Creation, rerun and read models for durable workflow runs."""

import io
import shutil
import uuid
import zipfile
from datetime import timedelta
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from app.core.config import get_settings
from app.crud import project_file_mapping as project_file_mapping_crud
from app.crud.resource import sample as sample_crud
from app.models.resource import Sample
from app.models.tools import ToolDefine
from app.models.user import User
from app.models.workflow import NodeDependency, NodeRun, Workflow, WorkflowOutbox, WorkflowRun
from app.schemas.project_workflow import WorkflowRunResult
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.user import CurrentUser
from app.schemas.workflow_run import (
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
from app.services.workflow.compiler import CompileContext, compile_run
from app.services.workflow.subgraphs import inject_run_data
from app.utils.input_format import to_safe_path_segment
from app.utils.system import random_uid, utc_now

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
    session.execute(delete(WorkflowOutbox).where(WorkflowOutbox.run_uid == run.uid))
    session.execute(delete(NodeDependency).where(NodeDependency.run_uid == run.uid))
    session.execute(delete(NodeRun).where(NodeRun.run_uid == run.uid))
    session.flush()


def _compile_new_run(
    session: Session,
    run: WorkflowRun,
    workflow: WorkflowDefinition,
    *,
    sample_name: str | None,
) -> None:
    run.workflow = workflow.model_copy(deep=True)
    run.status = RunStatus.PENDING
    run.settled = False
    run.start_time = None
    run.end_time = None
    compile_run(
        session,
        run,
        workflow,
        CompileContext(
            project_id=run.project_id,
            sample_uid=run.sample_uid,
            sample_name=sample_name,
            output_base=run.base_dir,
        ),
    )
    if run.settled:
        run.end_time = utc_now()


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
        workflow=workflow.model_copy(deep=True),
    )
    try:
        session.add(run)
        session.flush()
        _compile_new_run(session, run, workflow, sample_name=None)
        session.commit()
        session.refresh(run)
    except Exception:
        session.rollback()
        raise
    return run


def _find_project_run(
    session: Session,
    project_id: int,
    workflow_uid: uuid.UUID,
    sample_uid: uuid.UUID | None,
) -> WorkflowRun | None:
    statement = select(WorkflowRun).where(
        WorkflowRun.project_id == project_id,
        WorkflowRun.workflow_uid == workflow_uid,
    )
    statement = (
        statement.where(WorkflowRun.sample_uid.is_(None))
        if sample_uid is None
        else statement.where(WorkflowRun.sample_uid == sample_uid)
    )
    return session.scalar(statement.with_for_update())


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
    existing = _find_project_run(session, project_id, workflow.uid, sample_uid)
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
        session.add(run)
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
    _compile_new_run(session, run, workflow.workflow, sample_name=sample_name)
    return run, old_base_dir


def create_project_runs(
    session: Session,
    *,
    project_id: int,
    project_name: str,
    workflow: Workflow,
    samples: list[Sample | None],
    current_user: CurrentUser,
    run_name_prefix: str | None,
) -> WorkflowRunResult:
    """Atomically create or rerun all requested sample/project generations."""
    old_output_dirs: list[str] = []
    workflow_runs: list[WorkflowRun] = []
    try:
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
        for run in workflow_runs:
            session.refresh(run)
    except Exception:
        session.rollback()
        raise

    for old_output_dir in old_output_dirs:
        old_path = Path(old_output_dir)
        if old_path.exists():
            shutil.rmtree(old_path)
    return WorkflowRunResult(run_uids=[run.uid for run in workflow_runs], count=len(workflow_runs))


def get_run(session: Session, run_uid: uuid.UUID) -> WorkflowRun | None:
    """Load one workflow run."""
    return session.get(WorkflowRun, run_uid)


def get_node(session: Session, node_uid: uuid.UUID) -> NodeRun | None:
    """Load one current-generation node."""
    node = session.get(NodeRun, node_uid)
    if node is None:
        return None
    run = session.get(WorkflowRun, node.run_uid)
    return node if run is not None and node.generation == run.generation else None


def _to_node_record(node: NodeRun, run: WorkflowRun, owner_username: str) -> NodeRunRecord:
    return NodeRunRecord(
        uid=node.uid,
        run_uid=node.run_uid,
        run_name=run.name,
        generation=node.generation,
        definition_node_id=node.definition_node_id,
        node_type=node.node_type,
        name=node.name,
        status=node.status,
        owner_id=run.owner_id,
        owner_username=owner_username,
        project_id=run.project_id,
        workflow_uid=run.workflow_uid,
        sample_uid=run.sample_uid,
        create_time=node.create_time,
        start_time=node.start_time,
        end_time=node.end_time,
    )


def to_node_public(session: Session, node: NodeRun) -> NodeRunPublic:
    """Build task-center details for one current-generation node execution."""
    run = session.get(WorkflowRun, node.run_uid)
    if run is None or node.generation != run.generation:
        message = 'Node run does not belong to the current workflow generation'
        raise WorkflowRunNotFoundError(message)
    owner = session.get(User, run.owner_id)
    tool = session.get(ToolDefine, node.tool_uid) if node.tool_uid is not None else None
    record = _to_node_record(node, run, owner.username if owner is not None else str(run.owner_id))
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


def _node_list_filters(
    *,
    owner_id: int | None,
    project_id: int | None,
    status: NodeRunStatus | None,
) -> list[ColumnElement[bool]]:
    filters = [NodeRun.generation == WorkflowRun.generation]
    if owner_id is not None:
        filters.append(WorkflowRun.owner_id == owner_id)
    if project_id is not None:
        filters.append(WorkflowRun.project_id == project_id)
    if status is not None:
        filters.append(NodeRun.status == status)
    return filters


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
    filters = _node_list_filters(owner_id=owner_id, project_id=project_id, status=status)
    statement = (
        select(NodeRun, WorkflowRun, User.username)
        .join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid)
        .join(User, User.id == WorkflowRun.owner_id)
        .where(*filters)
        .order_by(NodeRun.create_time.desc(), NodeRun.uid.desc())
        .offset(offset)
        .limit(limit)
    )
    count_statement = (
        select(func.count()).select_from(NodeRun).join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid).where(*filters)
    )
    rows = session.execute(statement).all()
    total = session.scalar(count_statement) or 0
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
    filters = _node_list_filters(owner_id=owner_id, project_id=None, status=None)
    filters.append(NodeRun.create_time >= utc_now() - timedelta(hours=hours))
    statement = (
        select(NodeRun, WorkflowRun, User.username)
        .join(WorkflowRun, WorkflowRun.uid == NodeRun.run_uid)
        .join(User, User.id == WorkflowRun.owner_id)
        .where(*filters)
        .order_by(NodeRun.create_time.desc(), NodeRun.uid.desc())
        .limit(limit)
    )
    return [
        _to_node_record(node, run, owner_username) for node, run, owner_username in session.execute(statement).all()
    ]


def list_node_runs_for_run(session: Session, run: WorkflowRun) -> list[NodeRunRecord]:
    """Return current-generation node executions in workflow definition order."""
    owner = session.get(User, run.owner_id)
    owner_username = owner.username if owner is not None else str(run.owner_id)
    nodes = session.scalars(
        select(NodeRun)
        .where(NodeRun.run_uid == run.uid, NodeRun.generation == run.generation)
        .order_by(NodeRun.position_index, NodeRun.uid)
    )
    return [_to_node_record(node, run, owner_username) for node in nodes]


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


def to_public(session: Session, run: WorkflowRun) -> WorkflowRunPublic:
    """Build a current-generation graph with NodeRun data injected by stable definition ID."""
    nodes = list(
        session.scalars(
            select(NodeRun)
            .where(NodeRun.run_uid == run.uid, NodeRun.generation == run.generation)
            .order_by(NodeRun.position_index)
        )
    )
    graph_nodes = inject_run_data(
        run.workflow,
        {
            node.definition_node_id: NodeRunData(
                uid=node.uid,
                status=node.status,
                create_time=node.create_time,
                start_time=node.start_time,
                end_time=node.end_time,
            ).model_dump(mode='json')
            for node in nodes
        },
    )
    return WorkflowRunPublic(
        **WorkflowRunRecord.model_validate(run).model_dump(),
        nodes=graph_nodes,
        edges=run.workflow.edges,
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
    filters = []
    if owner_id is not None:
        filters.append(WorkflowRun.owner_id == owner_id)
    if project_id is not None:
        filters.append(WorkflowRun.project_id == project_id)
    if status is not None:
        filters.append(WorkflowRun.status == status)
    statement = select(WorkflowRun).where(*filters).order_by(WorkflowRun.create_time.desc()).offset(offset).limit(limit)
    count_statement = select(func.count()).select_from(WorkflowRun).where(*filters)
    runs = list(session.scalars(statement))
    total = session.scalar(count_statement) or 0
    return PaginatedWorkflowRuns(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[to_public(session, run) for run in runs],
    )


def run_statistics(session: Session, *, owner_id: int | None, project_id: int | None = None) -> WorkflowRunStatistics:
    """Aggregate workflow-run states."""
    filters = []
    if owner_id is not None:
        filters.append(WorkflowRun.owner_id == owner_id)
    if project_id is not None:
        filters.append(WorkflowRun.project_id == project_id)
    rows = session.execute(select(WorkflowRun.status, func.count()).where(*filters).group_by(WorkflowRun.status))
    counts: dict[RunStatus, int] = {}
    for row in rows:
        counts[row[0]] = row[1]
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
    filters = [
        WorkflowRun.project_id == project_id,
        WorkflowRun.workflow_uid == workflow_uid,
    ]
    if owner_id is not None:
        filters.append(WorkflowRun.owner_id == owner_id)
    current_runs = list(session.scalars(select(WorkflowRun).where(*filters).order_by(WorkflowRun.sample_uid)))
    if not current_runs:
        raise WorkflowRunNotFoundError(_NO_RUN)

    sample_uids = [run.sample_uid for run in current_runs if run.sample_uid is not None]
    sample_names = {
        sample.uid: sample.sample_name for sample in session.scalars(select(Sample).where(Sample.uid.in_(sample_uids)))
    }

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
            for file_path in run_dir.rglob('*'):
                if file_path.is_file():
                    archive.write(file_path, arcname=Path(root_name) / file_path.relative_to(run_dir))
    return buffer.getvalue()
