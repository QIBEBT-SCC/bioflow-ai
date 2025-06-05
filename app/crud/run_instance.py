from sqlalchemy import Select, Subquery
from sqlmodel import Session, select, col, func, case

from app.models.workflow import RunInstance, Status, TaskInstance
from app.schemas.run_instance import RunUpdate, TaskStat, RunPublic
from app.schemas.user import UserPublic


def insert(session: Session, run: RunInstance) -> RunInstance:
    session.add(run)
    session.commit()
    session.refresh(run)

    return run


def update(session: Session, uid: str, run_update: RunUpdate) -> RunInstance:
    db_run = get(session, uid)

    db_run.sqlmodel_update(run_update.model_dump(exclude_unset=True))
    session.add(db_run)
    session.commit()
    session.refresh(db_run)
    return db_run


def get_run_counts(session: Session, user_id: int):
    statement: Select = select(func.count()).select_from(RunInstance).where(col(RunInstance.owner_id) == user_id)

    return session.exec(statement).one()


def get_all_runs(session: Session, limit: int, offset: int, user_id: int) -> list[RunPublic]:
    """Get run instances with their task statistics in a single query."""
    # Create a subquery that counts tasks by status for each run instance
    task_stats: Subquery = (
        select(  # type: ignore
            TaskInstance.instance_uid,
            func.count().label('total'),
            func.sum(case((col(TaskInstance.status) == Status.SUCCESS, 1), else_=0)).label('success'),
            func.sum(case((col(TaskInstance.status) == Status.ERROR, 1), else_=0)).label('error'),
            func.sum(case((col(TaskInstance.status) == Status.RUNNING, 1), else_=0)).label('running'),
            func.sum(case((col(TaskInstance.status) == Status.WAITING, 1), else_=0)).label('waiting'),
        )
        .group_by(TaskInstance.instance_uid)
        .subquery()
    )

    # Main query to get run instances with their stats
    statement = (
        select(  # type: ignore
            RunInstance,
            task_stats.c.total,
            task_stats.c.success,
            task_stats.c.error,
            task_stats.c.running,
            task_stats.c.waiting,
        )
        .where(RunInstance.owner_id == user_id)
        .outerjoin(task_stats, col(RunInstance.uid) == task_stats.c.instance_uid)
        .order_by(col(RunInstance.create_time).desc())
        .offset(offset)
        .limit(limit)
    )

    results = session.exec(statement).all()

    return [
        RunPublic(
            uid=run.uid,
            name=run.name,
            owner=UserPublic.model_validate(run.owner.model_dump()),
            status=run.status,
            task_stats=TaskStat(total=total, success=success, error=error, running=running, waiting=waiting),
            create_time=run.create_time,
            start_time=run.start_time,
            end_time=run.end_time,
        )
        for run, total, success, error, running, waiting in results
    ]


def get_run_status(session: Session, user_id: int) -> TaskStat:
    """Get statistics of run instances by their status."""
    stats: Select = (
        select(  # type: ignore
            func.count().label('total'),
            func.sum(case((RunInstance.status == Status.SUCCESS, 1), else_=0)).label('success'),
            func.sum(case((RunInstance.status == Status.ERROR, 1), else_=0)).label('error'),
            func.sum(case((RunInstance.status == Status.RUNNING, 1), else_=0)).label('running'),
            func.sum(case((RunInstance.status == Status.WAITING, 1), else_=0)).label('waiting'),
        )
        .select_from(RunInstance)
        .where(col(RunInstance.owner_id) == user_id)
    )

    total, success, error, running, waiting = session.exec(stats).one()
    return TaskStat(total=total, success=success, error=error, running=running, waiting=waiting)


def get(session: Session, uid: str) -> RunInstance:
    statement: Select = select(RunInstance).where(RunInstance.uid == uid)
    return session.exec(statement).one()
