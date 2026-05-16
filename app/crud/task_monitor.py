from typing import TYPE_CHECKING

from sqlmodel import Session, select

from app.models.workflow import TaskMonitor

if TYPE_CHECKING:
    from sqlalchemy import Select


def insert_list(session: Session, monitor_list: list[TaskMonitor]) -> None:
    """Insert a list of task monitor records into the database."""
    session.add_all(monitor_list)
    session.commit()


def get(session: Session, task_uid: str) -> list[TaskMonitor]:
    """Return all monitor records for a given task, ordered by time."""
    statement: Select = select(TaskMonitor).where(TaskMonitor.task_uid == task_uid).order_by(TaskMonitor.time)
    return session.exec(statement).all()
