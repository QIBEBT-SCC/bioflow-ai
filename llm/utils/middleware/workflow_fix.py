"""Mutation policy for the workflow-fixer agent."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

MUTATING_ACTIONS = {
    'tool_node_manager': {'add', 'update_args', 'create'},
    'resource_node_manager': {'create_db', 'add'},
    'utility_node_manager': {
        'add_processor',
        'add_exist_code',
        'add_new_code',
        'add_note',
        'add_downstream_summary',
    },
    'workflow_manager': {
        'create_workflow_draft',
        'switch_workflow_draft',
        'delete_workflow_draft',
        'update_workflow_metadata',
        'delete_node',
        'fix_edge',
        'save_all_workflows',
    },
}
FORBIDDEN_DRAFT_ACTIONS = {
    'create_workflow_draft',
    'switch_workflow_draft',
    'delete_workflow_draft',
    'update_workflow_metadata',
    'save_all_workflows',
}


class WorkflowFixGuardMiddleware(AgentMiddleware):
    """Permit mutations only for a diagnosis that recommends patching the workflow."""

    @staticmethod
    def _rejection(request: ToolCallRequest) -> ToolMessage | None:
        tool_name = request.tool_call['name']
        args = request.tool_call.get('args')
        args = args if isinstance(args, dict) else {}
        action = args.get('action')

        if tool_name == 'workflow_manager' and action in FORBIDDEN_DRAFT_ACTIONS:
            return ToolMessage(
                content=(
                    'Fix uses one preloaded draft. Creating, switching, deleting, or saving drafts and changing '
                    'workflow metadata is disabled. Use finalize_workflow_fix when finished.'
                ),
                tool_call_id=request.tool_call['id'],
            )

        is_mutation = action in MUTATING_ACTIONS.get(tool_name, set())
        if tool_name == 'finalize_workflow_fix':
            is_mutation = args.get('mode') == 'save'
        if not is_mutation:
            return None

        state = request.state
        classification = state.get('diagnosis_classification', 'unknown')
        recommended_action = state.get('recommended_action', 'need_more_evidence')
        if recommended_action == 'patch_workflow' and classification in {'workflow_structure', 'workflow_usage'}:
            return None
        return ToolMessage(
            content=(
                f'Workflow mutation is forbidden for diagnosis classification={classification!r}, '
                f'recommended_action={recommended_action!r}. Inspect the workflow if useful, then call '
                'finalize_workflow_fix with mode="no_change" and give the user a concrete recommendation.'
            ),
            tool_call_id=request.tool_call['id'],
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Reject forbidden synchronous mutations before invoking a manager."""
        return self._rejection(request) or handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Reject forbidden asynchronous mutations before invoking a manager."""
        rejection = self._rejection(request)
        return rejection if rejection is not None else await handler(request)
