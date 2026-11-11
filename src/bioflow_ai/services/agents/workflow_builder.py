"""Prepare workflow-builder input from the project planner's accepted outputs."""

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

SAMPLES_FILENAME = 'SAMPLES.md'
PLAN_FILENAME = 'PLAN.md'
BUILD_REQUEST = 'Now start building the actual workflow(s) based on the above information.'
MISSING_PLAN_ERROR = 'workflow-builder requires SAMPLES.md and PLAN.md. Run project-planner first.'


class MissingProjectPlanError(FileNotFoundError):
    """Raised when a project has not been prepared by the project planner."""


def has_project_plan(workdir: Path) -> bool:
    """Whether the project directory holds both planner outputs."""
    return (workdir / SAMPLES_FILENAME).is_file() and (workdir / PLAN_FILENAME).is_file()


def read_project_plan(workdir: Path) -> tuple[str, str]:
    """Read the sample summary and analysis plan written by the project planner.

    Returns:
        The ``SAMPLES.md`` and ``PLAN.md`` contents.

    Raises:
        MissingProjectPlanError: Either file is missing.
    """
    if not has_project_plan(workdir):
        raise MissingProjectPlanError(MISSING_PLAN_ERROR)
    return (
        (workdir / SAMPLES_FILENAME).read_text(encoding='utf-8'),
        (workdir / PLAN_FILENAME).read_text(encoding='utf-8'),
    )


def plan_handoff_messages(samples: str, plan: str, request: HumanMessage | None = None) -> list[BaseMessage]:
    """Present the prepared sample summary and plan as the assistant's own earlier work.

    The plan is the planner's output rather than a user instruction, so the builder may set aside
    steps the user never asked for. Shared by web runs and offline evaluation.

    Args:
        samples: ``SAMPLES.md`` contents.
        plan: ``PLAN.md`` contents.
        request: The user's build request; a generic one is used when omitted.
    """
    return [
        AIMessage(content="I'll first review the data and design an analysis plan."),
        AIMessage(
            content='[Process collapsed]\n'
            'Below is a brief description of the sample to be analyzed and the proposed workflow plan:\n'
            f'<SampleInfo>\n{samples}\n</SampleInfo>\n'
            f'<Plan>\n{plan}\n</Plan>'
        ),
        request or HumanMessage(content=BUILD_REQUEST),
    ]


def prepare_workflow_builder_input(
    history: list[BaseMessage],
    current_message: HumanMessage,
    project_root: Path,
) -> dict[str, Any]:
    """Keep the conversation and insert the planner's documents before the current build request.

    Like the fixer's diagnosis report, these files are the only hand-off between planner and builder runs.
    """
    samples, plan = read_project_plan(project_root)
    previous = [message for message in history if message is not current_message]
    return {'messages': [*previous, *plan_handoff_messages(samples, plan, current_message)]}
