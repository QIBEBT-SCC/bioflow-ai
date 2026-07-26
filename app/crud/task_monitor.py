from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.workflow import TaskMonitor
from app.utils.identifiers import UUIDLike, as_uuid


def insert_list(session: Session, monitor_list: list[TaskMonitor]) -> None:
    """Insert a list of task monitor records into the database."""
    session.add_all(monitor_list)
    session.flush()


def get(session: Session, task_uid: UUIDLike) -> Sequence[TaskMonitor]:
    """Return all monitor records for a given task, ordered by time."""
    statement = select(TaskMonitor).where(TaskMonitor.task_uid == as_uuid(task_uid)).order_by(TaskMonitor.time)
    return session.scalars(statement).all()
