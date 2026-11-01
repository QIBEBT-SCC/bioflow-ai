"""Read queued user corrections at safe boundaries of a workflow build."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from bioflow_ai.llm.context import ProjectContext
from bioflow_ai.llm.states import UserInputState
from bioflow_ai.services.agents.sessions import (
    WORKFLOW_INPUT_RUN_KEY,
    QueuedAgentInput,
    acknowledge_inputs,
    close_inputs,
    pending_inputs,
)


def _run_uid() -> uuid.UUID | None:
    value = get_config().get('configurable', {}).get(WORKFLOW_INPUT_RUN_KEY)
    return uuid.UUID(value) if value else None


def _deliver(state: UserInputState, inputs: list[QueuedAgentInput]) -> dict[str, Any]:
    messages = [
        HumanMessage(
            id=str(item.message_uid),
            content=(
                '<user_update>\n'
                'The user added this instruction during the current workflow build. '
                'Apply it to the remaining work and revise the plan or Todo list when needed.\n'
                f'{item.text}\n</user_update>'
            ),
        )
        for item in inputs
    ]
    return {
        'messages': messages,
        'user_input_cursor': inputs[-1].event_id,
        'user_input_messages': [*state.get('user_input_messages', []), *messages],
    }


def _protect_request(request: ModelRequest) -> ModelRequest:
    protected = cast('list[HumanMessage]', request.state.get('user_input_messages', []))
    present = {message.id for message in request.messages}
    missing = [message for message in protected if message.id not in present]
    return request.override(messages=[*request.messages, *missing]) if missing else request


class PendingUserInputMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Deliver only to build coordinators, after their outstanding tools have returned.

    Args:
        close_inbox: Atomically stop accepting input when the agent produces its final answer.
    """

    state_schema = UserInputState

    def __init__(self, *, close_inbox: bool = False) -> None:
        super().__init__()
        self.close_inbox = close_inbox

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        """Restore corrections that a later hook compacted out of the history."""
        return handler(_protect_request(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelCallResult:
        """Restore compacted corrections in the async model request."""
        return await handler(_protect_request(request))

    def before_model(self, state: UserInputState, runtime: Runtime[ProjectContext]) -> dict[str, Any] | None:
        """Append new inputs in server acceptance order."""
        run_uid = _run_uid()
        if run_uid is None:
            return None
        inputs = pending_inputs(run_uid, state.get('user_input_cursor', 0))
        return _deliver(state, inputs) if inputs else None

    async def abefore_model(self, state: UserInputState, runtime: Runtime[ProjectContext]) -> dict[str, Any] | None:
        """Keep database reads off the worker's event loop."""
        return await asyncio.to_thread(self.before_model, state, runtime)

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: UserInputState, runtime: Runtime[ProjectContext]) -> dict[str, Any] | None:
        """Confirm delivery and catch inputs submitted during a final model response.

        This hook runs last among the agent's after-model hooks, so a final answer reaching it ends the loop.
        """
        run_uid = _run_uid()
        if run_uid is None:
            return None
        update: dict[str, Any] = {}
        cursor = state.get('user_input_cursor', 0)
        if cursor > state.get('user_input_acked', 0):
            acknowledge_inputs(run_uid, cursor)
            update['user_input_acked'] = cursor
        last_message = state['messages'][-1]
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            inputs = close_inputs(run_uid, cursor) if self.close_inbox else pending_inputs(run_uid, cursor)
            if inputs:
                return {**update, **_deliver(state, inputs), 'jump_to': 'model'}
        return update or None

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state: UserInputState, runtime: Runtime[ProjectContext]) -> dict[str, Any] | None:
        """Keep database writes off the worker's event loop."""
        return await asyncio.to_thread(self.after_model, state, runtime)
