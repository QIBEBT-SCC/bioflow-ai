"""Transaction boundaries for workflow run, task, and monitor persistence."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session

import app.crud.run_instance as run_crud
import app.crud.task_instance as task_crud
import app.crud.task_monitor as monitor_crud
from app.models.workflow import RunInstance, TaskInstance, TaskMonitor
from app.schemas.run_instance import RunUpdate
from app.schemas.task_instance import TaskUpdate
from app.utils.identifiers import UUIDLike

_RUN_NOT_FOUND = 'Run not found'
_TASK_NOT_FOUND = 'Task not found'


class WorkflowRuntimeError(Exception):
    """Base exception for workflow runtime persistence."""


class WorkflowRuntimeNotFoundError(WorkflowRuntimeError):
    """A requested run or task instance does not exist."""


@contextmanager
def _committing(session: Session) -> Generator[None, None, None]:
    """Commit a service transaction or roll it back on failure."""
    try:
        yield
        session.commit()
    except Exception:
        session.rollback()
        raise


def create_run(session: Session, run: RunInstance) -> RunInstance:
    """Persist a run instance."""
    with _committing(session):
        run_crud.insert(session, run)
    session.refresh(run)
    return run


def update_run(session: Session, uid: UUIDLike, data: RunUpdate) -> RunInstance:
    """Update a run instance."""
    with _committing(session):
        run = run_crud.update(session, uid, data)
        if run is None:
            raise WorkflowRuntimeNotFoundError(_RUN_NOT_FOUND)
    session.refresh(run)
    return run


def create_task(session: Session, task: TaskInstance) -> TaskInstance:
    """Persist a task instance before it is submitted to Celery."""
    with _committing(session):
        task_crud.insert(session, task)
    session.refresh(task)
    return task


def update_task(session: Session, uid: UUIDLike, data: TaskUpdate) -> TaskInstance:
    """Update a task instance."""
    with _committing(session):
        task = task_crud.update(session, uid, data)
        if task is None:
            raise WorkflowRuntimeNotFoundError(_TASK_NOT_FOUND)
    session.refresh(task)
    return task


def insert_monitors(session: Session, monitors: list[TaskMonitor]) -> None:
    """Persist a batch of task-monitor records."""
    with _committing(session):
        monitor_crud.insert_list(session, monitors)
