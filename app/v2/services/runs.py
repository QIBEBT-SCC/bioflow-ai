"""Creation, rerun and read models for V2 workflow runs."""

import io
import shutil
import uuid
import zipfile
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.crud import project_file_mapping as project_file_mapping_crud
from app.crud.resource import sample as sample_crud
from app.models.resource import Sample
from app.models.workflow import Workflow
from app.schemas.project_workflow import WorkflowRunResult
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.user import CurrentUser
from app.utils.input_format import to_safe_path_segment
from app.utils.system import random_uid, utc_now
from app.v2.models import NodeDependencyV2, NodeRunV2, WorkflowOutboxV2, WorkflowRunV2
from app.v2.schemas.workflow import (
    NodeRunData,
    NodeRunStatistics,
    NodeRunStatus,
    PaginatedWorkflowRuns,
    RunStatus,
    WorkflowRunPublic,
    WorkflowRunRecord,
    WorkflowRunStatistics,
)
from app.v2.services.compiler import CompileContext, compile_run

STORAGE_PATH = get_settings().persistent.storage.STORAGE_PATH
_NO_V2_RUN = 'No V2 run exists for this workflow'


class WorkflowRunConflictError(RuntimeError):
    """A V2 run cannot be restarted until its active generation settles."""


class WorkflowRunNotFoundError(RuntimeError):
    """No current V2 generation exists for a requested workflow package."""


def _project_output_root(project_id: int, workflow: Workflow, sample: Sample | None) -> Path:
    segment = f'{to_safe_path_segment(workflow.name)}-{workflow.uid}'
    root = Path(STORAGE_PATH).absolute() / 'proj' / str(project_id) / 'runs-v2' / segment
    if sample is None:
        return root / 'project'
    sample_segment = f'{to_safe_path_segment(sample.sample_name)}-{sample.uid}'
    return root / sample_segment


def _clear_generation(session: Session, run: WorkflowRunV2) -> None:
    session.execute(delete(WorkflowOutboxV2).where(WorkflowOutboxV2.run_uid == run.uid))
    session.execute(delete(NodeDependencyV2).where(NodeDependencyV2.run_uid == run.uid))
    session.execute(delete(NodeRunV2).where(NodeRunV2.run_uid == run.uid))
    session.flush()


def _compile_new_run(
    session: Session,
    run: WorkflowRunV2,
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
) -> WorkflowRunV2:
    """Create a fresh ad-hoc V2 run; graph IDs remain unchanged."""
    run_uid = random_uid()
    output_root = (
        Path(STORAGE_PATH).absolute()
        / f'{current_user.username}({current_user.id})'
        / 'temporarily-v2'
        / to_safe_path_segment(template_name)
        / str(run_uid)
    )
    run = WorkflowRunV2(
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
) -> WorkflowRunV2 | None:
    statement = select(WorkflowRunV2).where(
        WorkflowRunV2.project_id == project_id,
        WorkflowRunV2.workflow_uid == workflow_uid,
    )
    statement = (
        statement.where(WorkflowRunV2.sample_uid.is_(None))
        if sample_uid is None
        else statement.where(WorkflowRunV2.sample_uid == sample_uid)
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
) -> tuple[WorkflowRunV2, str | None]:
    sample_uid = sample.uid if sample else None
    sample_name = sample.sample_name if sample else None
    existing = _find_project_run(session, project_id, workflow.uid, sample_uid)
    output_root = _project_output_root(project_id, workflow, sample)
    old_base_dir: str | None = None

    if existing is not None:
        if not existing.settled:
            label = sample_name or project_name
            message = f'{label} already has an unsettled V2 workflow run'
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
        run = WorkflowRunV2(
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
    runs: list[WorkflowRunV2] = []
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
            runs.append(run)
            if old_base_dir:
                old_output_dirs.append(old_base_dir)
        session.commit()
        for run in runs:
            session.refresh(run)
    except Exception:
        session.rollback()
        raise

    for old_output_dir in old_output_dirs:
        old_path = Path(old_output_dir)
        if old_path.exists():
            shutil.rmtree(old_path)
    return WorkflowRunResult(run_uids=[run.uid for run in runs], count=len(runs))


def get_run(session: Session, run_uid: uuid.UUID) -> WorkflowRunV2 | None:
    """Load one V2 run."""
    return session.get(WorkflowRunV2, run_uid)


def get_node(session: Session, node_uid: uuid.UUID) -> NodeRunV2 | None:
    """Load one current-generation V2 node."""
    node = session.get(NodeRunV2, node_uid)
    if node is None:
        return None
    run = session.get(WorkflowRunV2, node.run_uid)
    return node if run is not None and node.generation == run.generation else None


def _node_statistics(nodes: list[NodeRunV2]) -> NodeRunStatistics:
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


def to_public(session: Session, run: WorkflowRunV2) -> WorkflowRunPublic:
    """Build a current-generation graph with NodeRun data injected by stable definition ID."""
    nodes = list(
        session.scalars(
            select(NodeRunV2)
            .where(NodeRunV2.run_uid == run.uid, NodeRunV2.generation == run.generation)
            .order_by(NodeRunV2.position_index)
        )
    )
    by_definition = {node.definition_node_id: node for node in nodes}
    graph_nodes = []
    for definition in run.workflow.nodes:
        node_dict = definition.model_dump(mode='json')
        node_run = by_definition.get(definition.id)
        if node_run is not None:
            node_dict.setdefault('data', {})['run_data'] = NodeRunData(
                uid=node_run.uid,
                status=node_run.status,
                create_time=node_run.create_time,
                start_time=node_run.start_time,
                end_time=node_run.end_time,
            ).model_dump(mode='json')
        graph_nodes.append(node_dict)
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
    offset: int = 0,
    limit: int = 20,
) -> PaginatedWorkflowRuns:
    """Return only V2 runs; V1 history is intentionally invisible."""
    filters = []
    if owner_id is not None:
        filters.append(WorkflowRunV2.owner_id == owner_id)
    if project_id is not None:
        filters.append(WorkflowRunV2.project_id == project_id)
    statement = (
        select(WorkflowRunV2).where(*filters).order_by(WorkflowRunV2.create_time.desc()).offset(offset).limit(limit)
    )
    count_statement = select(func.count()).select_from(WorkflowRunV2).where(*filters)
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
    """Aggregate V2 run states."""
    filters = []
    if owner_id is not None:
        filters.append(WorkflowRunV2.owner_id == owner_id)
    if project_id is not None:
        filters.append(WorkflowRunV2.project_id == project_id)
    rows = session.execute(select(WorkflowRunV2.status, func.count()).where(*filters).group_by(WorkflowRunV2.status))
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
    """Build a zip containing only current-generation V2 outputs."""
    filters = [
        WorkflowRunV2.project_id == project_id,
        WorkflowRunV2.workflow_uid == workflow_uid,
    ]
    if owner_id is not None:
        filters.append(WorkflowRunV2.owner_id == owner_id)
    current_runs = list(session.scalars(select(WorkflowRunV2).where(*filters).order_by(WorkflowRunV2.sample_uid)))
    if not current_runs:
        raise WorkflowRunNotFoundError(_NO_V2_RUN)

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
