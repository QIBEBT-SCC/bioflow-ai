from loguru import logger
from sqlalchemy import Select
from sqlmodel import Session, select

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
        select(TaskInstance).where(TaskInstance.owner_id == user_id).limit(limit).offset(offset)
    )
    return session.exec(statement).all()


def get(session: Session, uid: str) -> TaskInstance:
    statement: Select = select(TaskInstance).where(TaskInstance.uid == uid)
    return session.exec(statement).one()
