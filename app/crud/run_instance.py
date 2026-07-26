import uuid
from collections.abc import Sequence

from sqlalchemy import case, delete, func, select
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.orm.attributes import flag_modified

from app.models.workflow import RunInstance, TaskInstance, TaskMonitor
from app.schemas.common import Statistics
from app.schemas.react_flow import WorkflowDefinition
from app.schemas.run_instance import RunUpdate
from app.schemas.workflow_config import Status
from app.utils.identifiers import UUIDLike, as_uuid


def insert(session: Session, run: RunInstance) -> RunInstance:
    """Insert a new run instance into the database.

    Args:
        session: Database session
        run: Run instance to insert

    Returns:
        The inserted run instance
    """
    session.add(run)
    session.flush()
    session.refresh(run)
    return run


def update(session: Session, uid: UUIDLike, run_update: RunUpdate) -> RunInstance | None:
    """Update an existing run instance in the database.

    Args:
        session: Database session
        uid: UID of the run instance to update
        run_update: Data to update

    Returns:
        The updated run instance
    """
    db_run = get(session, uid)
    if db_run is None:
        return None
    update_data = run_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_run, field, value)
    session.add(db_run)
    session.flush()
    session.refresh(db_run)
    return db_run


def get_run_status(session: Session, user_id: int) -> Statistics:
    """Get statistics of run instances by their status."""
    stats = (
        select(
            func.count().label('total'),
            func.coalesce(func.sum(case((RunInstance.status == Status.SUCCESS, 1), else_=0)), 0).label('success'),
            func.coalesce(func.sum(case((RunInstance.status == Status.ERROR, 1), else_=0)), 0).label('error'),
            func.coalesce(func.sum(case((RunInstance.status == Status.RUNNING, 1), else_=0)), 0).label('running'),
            func.coalesce(func.sum(case((RunInstance.status == Status.WAITING, 1), else_=0)), 0).label('waiting'),
        )
        .select_from(RunInstance)
        .where(RunInstance.owner_id == user_id)
    )

    # pyrefly: ignore [no-matching-overload]
    total, success, error, running, waiting = session.execute(stats).one()
    return Statistics(total=total, success=success, error=error, running=running, waiting=waiting)


def get_runs_page(session: Session, limit: int, offset: int, user_id: int) -> tuple[Sequence[RunInstance], int]:
    """Return a page of a user's run instances together with the total count."""
    page_stmt = (
        select(RunInstance)
        .where(RunInstance.owner_id == user_id)
        # pyrefly: ignore [bad-argument-type]
        .options(selectinload(RunInstance.owner), selectinload(RunInstance.tasks))
        .order_by(RunInstance.create_time.desc())
        .limit(limit)
        .offset(offset)
    )
    count_stmt = select(func.count()).select_from(RunInstance).where(RunInstance.owner_id == user_id)
    return session.scalars(page_stmt).all(), session.scalar(count_stmt) or 0


def get(session: Session, uid: UUIDLike) -> RunInstance | None:
    """Get a run instance by its UID.

    Args:
        session: Database session
        uid: UID of the run instance

    Returns:
        The requested run instance
    """
    statement = (
        select(RunInstance)
        .where(RunInstance.uid == as_uuid(uid))
        .options(selectinload(RunInstance.owner), selectinload(RunInstance.tasks))
    )
    return session.scalar(statement)


def get_run_task_stats(session: Session, uid: UUIDLike) -> Statistics:
    """Get task statistics for a specific run instance."""
    statement = (
        select(
            func.count().label('total'),
            func.coalesce(func.sum(case((TaskInstance.status == Status.SUCCESS, 1), else_=0)), 0).label('success'),
            func.coalesce(func.sum(case((TaskInstance.status == Status.ERROR, 1), else_=0)), 0).label('error'),
            func.coalesce(func.sum(case((TaskInstance.status == Status.RUNNING, 1), else_=0)), 0).label('running'),
            func.coalesce(func.sum(case((TaskInstance.status == Status.WAITING, 1), else_=0)), 0).label('waiting'),
        )
        .select_from(TaskInstance)
        .where(TaskInstance.instance_uid == as_uuid(uid))
    )

    # pyrefly: ignore [no-matching-overload]
    total, success, error, running, waiting = session.execute(statement).one()
    return Statistics(total=total, success=success, error=error, running=running, waiting=waiting)


def get_runs_by_project(session: Session, project_id: int, limit: int = 20, offset: int = 0) -> Sequence[RunInstance]:
    """Get all run instances for a specific project."""
    statement = (
        select(RunInstance)
        .where(RunInstance.project_id == project_id)
        .options(selectinload(RunInstance.owner), selectinload(RunInstance.tasks))
        .order_by(RunInstance.create_time.desc())
        .limit(limit)
        .offset(offset)
    )

    return session.scalars(statement).all()


def get_nonterminal_runs_by_user(session: Session, user_id: int) -> Sequence[RunInstance]:
    """Get all non-terminal run instances for a user."""
    statement = (
        select(RunInstance)
        .where(RunInstance.owner_id == user_id)
        .where(RunInstance.status.not_in([Status.SUCCESS, Status.ERROR]))
        .options(selectinload(RunInstance.tasks))
    )
    return session.scalars(statement).all()


def get_nonterminal_runs_by_project(session: Session, project_id: int) -> Sequence[RunInstance]:
    """Get all non-terminal run instances for a project."""
    statement = (
        select(RunInstance)
        .where(RunInstance.project_id == project_id)
        .where(RunInstance.status.not_in([Status.SUCCESS, Status.ERROR]))
        .options(selectinload(RunInstance.tasks))
    )
    return session.scalars(statement).all()


def get_runs_by_project_page(
    session: Session, project_id: int, limit: int = 20, offset: int = 0
) -> tuple[Sequence[RunInstance], int]:
    """Return a page of a project's run instances together with the total count."""
    count_stmt = select(func.count()).select_from(RunInstance).where(RunInstance.project_id == project_id)
    return get_runs_by_project(session, project_id, limit, offset), session.scalar(count_stmt) or 0


def get_runs_by_project_and_workflow(
    session: Session, project_id: int, workflow_uid: uuid.UUID
) -> Sequence[RunInstance]:
    """Get all run instances for a specific project and workflow combination.

    Args:
        session: Database session
        project_id: Project ID
        workflow_uid: Workflow template UID

    Returns:
        A sequence of run instances ordered by creation time descending
    """
    statement = (
        select(RunInstance)
        .where(RunInstance.project_id == project_id)
        .where(RunInstance.workflow_uid == workflow_uid)
        .order_by(RunInstance.create_time.desc())
    )

    return session.scalars(statement).all()


def get_runs_by_sample(
    session: Session,
    sample_uid: UUIDLike,
    limit: int = 20,
    offset: int = 0,
) -> Sequence[RunInstance]:
    """Get all run instances for a specific sample.

    Args:
        session: Database session
        sample_uid: Sample UID
        limit: Max number of instances to return
        offset: Offset for pagination

    Returns:
        List of run instances
    """
    statement = (
        select(RunInstance)
        .where(RunInstance.sample_uid == as_uuid(sample_uid))
        .options(selectinload(RunInstance.owner), selectinload(RunInstance.tasks))
        .order_by(RunInstance.create_time.desc())
        .limit(limit)
        .offset(offset)
    )

    return session.scalars(statement).all()


def get_project_run_stats(session: Session, project_id: int) -> Statistics:
    """Get statistics of run instances by status for a project.

    Args:
        session: Database session
        project_id: Project ID

    Returns:
        Statistics: Counts of total, success, error, running, waiting run instances.
    """
    stats = (
        select(
            func.count().label('total'),
            func.coalesce(func.sum(case((RunInstance.status == Status.SUCCESS, 1), else_=0)), 0).label('success'),
            func.coalesce(func.sum(case((RunInstance.status == Status.ERROR, 1), else_=0)), 0).label('error'),
            func.coalesce(func.sum(case((RunInstance.status == Status.RUNNING, 1), else_=0)), 0).label('running'),
            func.coalesce(func.sum(case((RunInstance.status == Status.WAITING, 1), else_=0)), 0).label('waiting'),
        )
        .select_from(RunInstance)
        .where(RunInstance.project_id == project_id)
    )

    # pyrefly: ignore [no-matching-overload]
    total, success, error, running, waiting = session.execute(stats).one()
    return Statistics(total=total, success=success, error=error, running=running, waiting=waiting)


def get_latest_run_by_sample_and_workflow(
    session: Session, sample_uid: uuid.UUID, workflow_uid: uuid.UUID
) -> RunInstance | None:
    """Get the latest run instance for a specific sample and workflow combination.

    Args:
        session: Database session
        sample_uid: Sample UID
        workflow_uid: Workflow template UID

    Returns:
        The latest run instance, or None if not found
    """
    statement = (
        select(RunInstance)
        .where(RunInstance.sample_uid == sample_uid)
        .where(RunInstance.workflow_uid == workflow_uid)
        .options(selectinload(RunInstance.tasks))
        .order_by(RunInstance.create_time.desc())
        .limit(1)
    )
    return session.scalar(statement)


def get_latest_run_by_project_and_workflow(
    session: Session, project_id: int, workflow_uid: uuid.UUID
) -> RunInstance | None:
    """Get the latest run instance for a specific project and workflow combination (project level).

    Args:
        session: Database session
        project_id: Project ID
        workflow_uid: Workflow UID

    Returns:
        The latest project-level run instance, or None if not found
    """
    statement = (
        select(RunInstance)
        .where(RunInstance.project_id == project_id)
        .where(RunInstance.workflow_uid == workflow_uid)
        .where(RunInstance.sample_uid.is_(None))
        .options(selectinload(RunInstance.tasks))
        .order_by(RunInstance.create_time.desc())
        .limit(1)
    )
    return session.scalar(statement)


def clear_run_for_rerun(session: Session, run: RunInstance, new_workflow: WorkflowDefinition) -> None:
    """Clear execution data of a run instance for rerun.

    - Delete all TaskInstance records (cascading delete TaskMonitor)
    - Reset run status to WAITING
    - Clear start_time and end_time

    Args:
        session: Database session
        run: Run instance to clear
        new_workflow: New workflow definition
    """
    # 1. Delete task monitor logs (TaskMonitor)
    subquery = select(TaskInstance.uid).where(TaskInstance.instance_uid == run.uid)
    session.execute(delete(TaskMonitor).where(TaskMonitor.task_uid.in_(subquery)))

    # 2. Delete task instances (TaskInstance)
    session.execute(delete(TaskInstance).where(TaskInstance.instance_uid == run.uid))
    session.expire(run)

    # 3. Reset workflow definition (RunInstance)
    run.workflow = new_workflow
    flag_modified(run, 'workflow')

    # 4. Reset run status
    run.status = Status.WAITING
    run.start_time = None
    run.end_time = None
    session.add(run)
    session.flush()
