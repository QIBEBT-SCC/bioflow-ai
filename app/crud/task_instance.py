from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.models.workflow import TaskInstance
from app.schemas.task_instance import TaskUpdate
from app.utils.identifiers import UUIDLike, as_uuid


def insert(session: Session, task: TaskInstance) -> TaskInstance:
    """Insert a new task instance into the database."""
    session.add(task)
    session.flush()
    session.refresh(task)
    return task


def update(session: Session, uid: UUIDLike, task_update: TaskUpdate) -> TaskInstance | None:
    """Update an existing task instance in the database."""
    db_task = get(session, uid)
    if db_task is None:
        return None

    update_data = task_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_task, field, value)
    session.add(db_task)
    session.flush()
    session.refresh(db_task)
    return db_task


def get_all(session: Session, limit: int, offset: int, user_id: int | None) -> Sequence[TaskInstance]:
    """Get a paginated list of task instances for a specific user."""
    statement = (
        select(TaskInstance)
        .where(TaskInstance.track_progress.is_(True))
        .options(joinedload(TaskInstance.owner), joinedload(TaskInstance.run_instance))
        .order_by(TaskInstance.uid.desc())
        .limit(limit)
        .offset(offset)
    )
    if user_id is not None:
        statement = statement.where(TaskInstance.owner_id == user_id)
    return session.scalars(statement).all()


def get_all_by_time(session: Session, user_id: int | None, delta: timedelta) -> Sequence[TaskInstance]:
    """Get task instances for a user within a recent time window."""
    now = datetime.now(UTC)
    since_time = now - delta
    statement = (
        select(TaskInstance)
        .where(
            or_(
                and_(
                    TaskInstance.start_time.is_not(None),
                    TaskInstance.start_time >= since_time,
                    TaskInstance.start_time <= now,
                ),
                and_(
                    TaskInstance.end_time.is_not(None),
                    TaskInstance.end_time >= since_time,
                    TaskInstance.end_time <= now,
                ),
            ),
        )
        .where(TaskInstance.track_progress.is_(True))
        .options(joinedload(TaskInstance.owner))
    )
    if user_id is not None:
        statement = statement.where(TaskInstance.owner_id == user_id)
    return session.scalars(statement).all()


def get_count(session: Session, user_id: int | None) -> int:
    """Get the total count of task instances for a specific user."""
    statement = (
        select(func.count())
        .select_from(TaskInstance)
        .where(TaskInstance.track_progress.is_(True))
    )
    if user_id is not None:
        statement = statement.where(TaskInstance.owner_id == user_id)
    return session.scalar(statement) or 0


def get(session: Session, uid: UUIDLike) -> TaskInstance | None:
    """Get a task instance by its UID."""
    statement = (
        select(TaskInstance)
        .where(TaskInstance.uid == as_uuid(uid))
        .options(joinedload(TaskInstance.owner), joinedload(TaskInstance.run_instance), joinedload(TaskInstance.tool))
    )
    return session.scalar(statement)
