"""Editing-scope policy for the workflow-fixer agent."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

FORBIDDEN_DRAFT_ACTIONS = {
    'create_workflow_draft',
    'switch_workflow_draft',
    'delete_workflow_draft',
    'update_workflow_metadata',
    'save_all_workflows',
}


class WorkflowFixScopeMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Prevent unsupported draft and workflow-metadata operations."""

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

        return None

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
