"""Keep a delegated subgraph builder inside its single preloaded draft."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from bioflow_ai.llm.workflow_builder.subgraphs import INPUT_NODE_ID

FORBIDDEN_WORKFLOW_ACTIONS = {
    'create_workflow_draft',
    'switch_workflow_draft',
    'delete_workflow_draft',
    'update_workflow_metadata',
    'save_all_workflows',
}


class SubgraphBuilderScopeMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Prevent a child from changing parent drafts or recursively creating children."""

    @staticmethod
    def _rejection(request: ToolCallRequest) -> ToolMessage | None:
        tool_name = request.tool_call['name']
        args = request.tool_call.get('args')
        args = args if isinstance(args, dict) else {}
        action = args.get('action')
        params = args.get('params')
        params = params if isinstance(params, dict) else {}
        forbidden = (
            (tool_name == 'workflow_manager' and action in FORBIDDEN_WORKFLOW_ACTIONS)
            or (tool_name == 'workflow_manager' and action == 'delete_node' and params.get('node_id') == INPUT_NODE_ID)
            or (tool_name == 'subgraph_node_manager' and action == 'create')
        )
        if forbidden:
            return ToolMessage(
                content='The subgraph builder has one preloaded draft. Use its interface manager to edit and save it.',
                tool_call_id=request.tool_call['id'],
            )
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Reject forbidden synchronous calls."""
        return self._rejection(request) or handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Reject forbidden asynchronous calls."""
        rejection = self._rejection(request)
        return rejection if rejection is not None else await handler(request)
