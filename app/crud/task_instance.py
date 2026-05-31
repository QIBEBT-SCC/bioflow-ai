from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import joinedload
from sqlmodel import Session, and_, col, func, or_, select

from app.models.workflow import TaskInstance
from app.schemas.task_instance import TaskUpdate


def insert(session: Session, task: TaskInstance) -> TaskInstance:
    """Insert a new task instance into the database."""
    session.add(task)
    session.commit()
    session.refresh(task)

    return task


def update(session: Session, uid: str, task_update: TaskUpdate) -> TaskInstance:
    """Update an existing task instance in the database."""
    db_task = get(session, uid)

    update_data = task_update.model_dump(exclude_unset=True)
    db_task.sqlmodel_update(update_data)
    session.add(db_task)
    session.commit()
    session.refresh(db_task)
    return db_task


def get_all(session: Session, limit: int, offset: int, user_id: int) -> Sequence[TaskInstance]:
    """Get a paginated list of task instances for a specific user."""
    statement = (
        select(TaskInstance)
        .where(TaskInstance.owner_id == user_id)
        .where(TaskInstance.name != 'read_file')
        .options(joinedload(TaskInstance.owner), joinedload(TaskInstance.run_instance))  # type: ignore
        .order_by(col(TaskInstance.uid).desc())
        .limit(limit)
        .offset(offset)
    )
    return session.exec(statement).all()


def get_all_by_time(session: Session, user_id: int, delta: timedelta) -> Sequence[TaskInstance]:
    """Get task instances for a user within a recent time window."""
    now = datetime.now(UTC)
    since_time = now - delta
    statement = (
        select(TaskInstance)
        .where(
            TaskInstance.owner_id == user_id,
            or_(
                and_(
                    col(TaskInstance.start_time).isnot(None),
                    col(TaskInstance.start_time) >= since_time,
                    col(TaskInstance.start_time) <= now,
                ),
                and_(
                    col(TaskInstance.end_time).isnot(None),
                    col(TaskInstance.end_time) >= since_time,
                    col(TaskInstance.end_time) <= now,
                ),
            ),
        )
        .where(TaskInstance.name != 'read_file')
        .options(joinedload(TaskInstance.owner))  # type: ignore
    )
    return session.exec(statement).all()


def get_count(session: Session, user_id: int) -> int:
    """Get the total count of task instances for a specific user."""
    statement = select(func.count()).select_from(TaskInstance).where(col(TaskInstance.owner_id) == user_id)

    return session.exec(statement).one()


def get(session: Session, uid: str) -> TaskInstance:
    """Get a task instance by its UID."""
    statement = (
        select(TaskInstance)
        .where(TaskInstance.uid == uid)
        .options(joinedload(TaskInstance.owner), joinedload(TaskInstance.run_instance), joinedload(TaskInstance.tool))  # type: ignore
    )
    return session.exec(statement).one()
