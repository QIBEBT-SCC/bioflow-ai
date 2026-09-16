from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult, StateT, hook_config
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT

from llm.context import BaseContext

PARENT_HANDOFF_PROMPT = """## Parent-agent handoff
You are running as a sub-agent. If any instruction tells you to call `ask_human`, call `return_to_parent`
instead. Include the complete blocking issue, relevant execution context, and a concrete recommended next action. If a
human decision or new input is genuinely required, also include the exact question the parent agent should ask. The
parent may instead resolve the issue by replanning from your verified findings. This ends your current run and returns
the request to the parent agent; do not try to continue afterward.
"""


@tool('return_to_parent', parse_docstring=True)
async def return_to_parent(request: str, runtime: ToolRuntime[BaseContext]) -> Command:
    """Return a blocking request to the supervising agent and stop this sub-agent.

    Args:
        request: Complete issue context, recommended next action, and any exact question that still requires user input.
    """
    if not runtime.context.delegate_human_requests:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='`return_to_parent` is only available in delegated runs. Call `ask_human` instead.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    return Command(
        update={
            'return_to_parent': True,
            'parent_request': request,
            'messages': [
                ToolMessage(
                    content='The blocking request has been returned to the parent agent.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class ParentHandoffState(AgentState, total=False):
    return_to_parent: bool
    parent_request: str


class ParentHandoffMiddleware(AgentMiddleware):
    """Swap human interrupts for an early, structured parent handoff in delegated runs."""

    tools = [return_to_parent]  # noqa: RUF012
    state_schema = ParentHandoffState

    @staticmethod
    def _tool_name(candidate: BaseTool | dict[str, Any]) -> str | None:
        if isinstance(candidate, dict):
            name = candidate.get('name')
            return name if isinstance(name, str) else None
        return candidate.name

    @staticmethod
    def _prepare_request(request: ModelRequest) -> ModelRequest:
        delegated = getattr(request.runtime.context, 'delegate_human_requests', False)
        hidden_tool = 'ask_human' if delegated else 'return_to_parent'
        tools = [
            candidate for candidate in request.tools if ParentHandoffMiddleware._tool_name(candidate) != hidden_tool
        ]
        if not delegated:
            return request.override(tools=tools)

        existing_content = list(request.system_message.content_blocks) if request.system_message is not None else []
        # pyrefly: ignore [no-matching-overload]
        system_message = SystemMessage(content=[*existing_content, {'type': 'text', 'text': PARENT_HANDOFF_PROMPT}])
        return request.override(tools=tools, system_message=system_message)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Expose the human tool appropriate for the current invocation mode."""
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Expose the human tool appropriate for the current invocation mode."""
        return await handler(self._prepare_request(request))

    @hook_config(can_jump_to=['end'])
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """End immediately after a sub-agent returns a blocking request."""
        if state.get('return_to_parent', False):
            return {'jump_to': 'end'}
        return None

    @hook_config(can_jump_to=['end'])
    async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """End immediately after a sub-agent returns a blocking request."""
        return self.before_model(state, runtime)
