"""Runtime task and run status reconciliation."""

from collections.abc import Iterable

from celery import states

from app.models.workflow import RunInstance, Status, TaskInstance
from app.schemas.common import Statistics
from script.worker import celery_app

TERMINAL_STATUSES = {Status.SUCCESS, Status.ERROR}

_CELERY_STATE_TO_STATUS = {
    states.PENDING: Status.WAITING,
    states.RECEIVED: Status.WAITING,
    states.STARTED: Status.RUNNING,
    states.RETRY: Status.RUNNING,
    states.SUCCESS: Status.SUCCESS,
    states.FAILURE: Status.ERROR,
    states.REVOKED: Status.ERROR,
}


def refresh_task_status(task: TaskInstance) -> TaskInstance:
    """Apply a task's effective runtime status in memory."""
    refresh_task_statuses([task])
    return task


def get_task_status(task_uid: str, db_status: Status | int = Status.WAITING) -> Status:
    """Return the effective runtime status for a task without touching the database."""
    normalized_db_status = _normalize_status(db_status)
    if normalized_db_status in TERMINAL_STATUSES:
        return normalized_db_status

    celery_status = _get_task_status_from_celery(task_uid)
    return celery_status if celery_status is not None else normalized_db_status


def refresh_task_statuses(tasks: Iterable[TaskInstance]) -> dict[str, Status]:
    """Apply non-terminal task statuses from Celery in memory and return them."""
    effective_statuses: dict[str, Status] = {}

    for task in tasks:
        effective_status = get_task_status(str(task.uid), task.status)
        if not isinstance(task.status, Status) or task.status != effective_status:
            task.status = effective_status

        effective_statuses[str(task.uid)] = effective_status

    return effective_statuses


def refresh_run_status(run: RunInstance) -> Statistics:
    """Apply a run's effective runtime status in memory and return task statistics."""
    refresh_task_statuses(run.tasks)
    stats = build_task_statistics(run.tasks)

    normalized_run_status = _normalize_status(run.status)
    if not isinstance(run.status, Status):
        run.status = normalized_run_status

    if normalized_run_status not in TERMINAL_STATUSES:
        effective_run_status = _get_run_status_from_stats(stats)
        run.status = effective_run_status

    return stats


def apply_effective_nonterminal_run_stats(stats: Statistics, runs: Iterable[RunInstance]) -> Statistics:
    """Adjust aggregate run statistics using effective Celery state for non-terminal runs."""
    result = stats.model_copy()
    for run in runs:
        original_status = _normalize_status(run.status)
        refresh_run_status(run)
        effective_status = _normalize_status(run.status)

        if original_status == effective_status:
            continue

        _decrement_status(result, original_status)
        _increment_status(result, effective_status)

    return result


def build_task_statistics(tasks: Iterable[TaskInstance]) -> Statistics:
    """Build task statistics from effective in-memory task statuses."""
    task_list = list(tasks)
    return Statistics(
        total=len(task_list),
        waiting=sum(1 for task in task_list if task.status == Status.WAITING),
        running=sum(1 for task in task_list if task.status == Status.RUNNING),
        success=sum(1 for task in task_list if task.status == Status.SUCCESS),
        error=sum(1 for task in task_list if task.status == Status.ERROR),
    )


def _get_task_status_from_celery(task_uid: str) -> Status | None:
    result = celery_app.AsyncResult(task_uid)
    return _CELERY_STATE_TO_STATUS.get(result.state)


def _get_run_status_from_stats(stats: Statistics) -> Status:
    if stats.error > 0:
        return Status.ERROR
    if stats.total > 0 and stats.success == stats.total:
        return Status.SUCCESS
    if stats.running > 0:
        return Status.RUNNING
    return Status.WAITING


def _normalize_status(status: Status | int) -> Status:
    """Convert integer values loaded from SQL integer columns to ``Status``."""
    return status if isinstance(status, Status) else Status(status)


def _decrement_status(stats: Statistics, status: Status | int) -> None:
    field_name = _normalize_status(status).name.lower()
    setattr(stats, field_name, max(getattr(stats, field_name) - 1, 0))


def _increment_status(stats: Statistics, status: Status | int) -> None:
    field_name = _normalize_status(status).name.lower()
    setattr(stats, field_name, getattr(stats, field_name) + 1)
