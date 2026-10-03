"""Message formatting helpers for workflow-builder tools."""

from bioflow_ai.llm.states import WorkflowDraft
from bioflow_ai.llm.workflow_builder.serialization import format_execution_scope
from bioflow_ai.models.workflow import Workflow


def format_workflow_drafts(drafts: list[WorkflowDraft], active_workflow_id: str | None) -> str:
    """Format draft summaries for the workflow manager tool."""
    if not drafts:
        return 'No workflow drafts exist yet. Use create_workflow_draft before building a workflow.'

    lines = ['Workflow drafts:']
    for draft in drafts:
        active_marker = ' (active)' if draft.draft_id == active_workflow_id else ''
        description = f' - {draft.description}' if draft.description else ''
        lines.append(
            f'- {draft.draft_id}{active_marker}: {draft.name}'
            f' [{draft.execution_scope}, {len(draft.nodes)} nodes, {len(draft.edges)} edges]{description}'
        )
    return '\n'.join(lines)


def format_saved_workflows(workflows: list[Workflow]) -> str:
    """Format saved workflow records for the tool response."""
    saved_lines = '\n'.join(
        f'- {workflow.name} ({format_execution_scope(workflow.execution_scope)}): {workflow.uid}'
        for workflow in workflows
    )
    return f'Saved {len(workflows)} workflow(s) to the database:\n{saved_lines}'
