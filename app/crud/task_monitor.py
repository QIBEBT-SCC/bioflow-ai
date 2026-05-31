from collections.abc import Sequence

from sqlmodel import Session, col, select

from app.models.workflow import TaskMonitor


def insert_list(session: Session, monitor_list: list[TaskMonitor]) -> None:
    """Insert a list of task monitor records into the database."""
    session.add_all(monitor_list)
    session.commit()


def get(session: Session, task_uid: str) -> Sequence[TaskMonitor]:
    """Return all monitor records for a given task, ordered by time."""
    statement = select(TaskMonitor).where(TaskMonitor.task_uid == task_uid).order_by(col(TaskMonitor.time))
    return session.exec(statement).all()
