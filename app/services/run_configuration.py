"""Resolve execution options for new and repeated workflow runs."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.workflow import TaskInstance

AUTO_SUMMARY_TASK_NAME = 'upstream_summary'


def resolve_auto_summary(
    *,
    requested: bool | None,
    previous_tasks: Iterable['TaskInstance'] | None,
    workflow_default: bool,
) -> bool:
    """Resolve auto-summary without losing the previous run's explicit choice.

    An explicit request always wins. A rerun without an override inherits the
    option that was actually instantiated previously. Only a brand-new run
    falls back to the workflow template default.
    """
    if requested is not None:
        return requested
    if previous_tasks is not None:
        return any(task.name == AUTO_SUMMARY_TASK_NAME for task in previous_tasks)
    return workflow_default
