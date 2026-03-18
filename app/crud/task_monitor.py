from sqlalchemy import Select
from sqlmodel import Session, select

from app.models.workflow import TaskMonitor


def insert_list(session: Session, monitor_list: list[TaskMonitor]):
    session.add_all(monitor_list)
    session.commit()


def get(session: Session, task_uid: str):
    statement: Select = select(TaskMonitor).where(TaskMonitor.task_uid == task_uid).order_by(TaskMonitor.time)
    return session.exec(statement).all()
