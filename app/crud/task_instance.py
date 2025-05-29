from datetime import datetime, UTC, timedelta

from loguru import logger
from sqlalchemy import Select, func
from sqlmodel import Session, select, or_, and_, col

from app.models.workflow import TaskInstance, TaskUpdate


def insert(session: Session, task: TaskInstance) -> TaskInstance:
    session.add(task)
    session.commit()
    session.refresh(task)

    return task


def update(session: Session, uid: str, task_update: TaskUpdate) -> TaskInstance:
    db_task = get(session, uid)
    logger.debug(db_task)

    update_data = task_update.model_dump(exclude_unset=True)
    logger.debug(update_data)
    db_task.sqlmodel_update(update_data)
    logger.debug(db_task)
    session.add(db_task)
    session.commit()
    session.refresh(db_task)
    return db_task


def get_all(session: Session, limit: int, offset: int, user_id: int) -> list[TaskInstance]:
    statement: Select = (
        select(TaskInstance)
        .where(TaskInstance.owner_id == user_id)
        .order_by(col(TaskInstance.uid).desc())
        .limit(limit)
        .offset(offset)
    )
    return session.exec(statement).all()


def get_all_by_time(session: Session, user_id: int, delta: timedelta) -> list[TaskInstance]:
    now = datetime.now(UTC)
    since_time = now - delta
    statement: Select = (
        select(TaskInstance)
        .where(
            TaskInstance.owner_id == user_id,
            or_(
                and_(
                    TaskInstance.start_time is not None,
                    TaskInstance.start_time >= since_time,
                    TaskInstance.start_time <= now,
                ),
                and_(
                    TaskInstance.end_time is not None,
                    TaskInstance.end_time >= since_time,
                    TaskInstance.end_time <= now,
                ),
                and_(
                    TaskInstance.start_time is not None,
                    TaskInstance.start_time >= since_time,
                    TaskInstance.start_time <= now,
                    TaskInstance.end_time is None,
                ),
            ),
        )
        .where(TaskInstance.name != 'read_file')
    )
    return session.exec(statement).all()


def get_count(session: Session, user_id: int) -> int:
    statement: Select = (
        select(func.count()).select_from(TaskInstance).where(col(TaskInstance.owner_id) == user_id)
    )

    return session.exec(statement).one()


def get(session: Session, uid: str) -> TaskInstance:
    statement: Select = select(TaskInstance).where(TaskInstance.uid == uid)
    return session.exec(statement).one()
