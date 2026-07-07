from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from langgraph.typing import ContextT
from loguru import logger

from llm.states import TestRunState

SPECIFIC_TOOLS_PARALLEL_PROMPT_TEMPLATE = """
## Tool Call Restriction
The following tools must be called alone in a separate response. They cannot be called in parallel with any other tool:
{tool_list}
"""


class DisableParallelToolCallsMiddleware(AgentMiddleware):
    def __init__(
        self,
        disabled_tools: Optional[set[str]] = None,
        disabled_tool_actions: Optional[dict[str, set[str]]] = None,
    ):
        self.disabled_tools = disabled_tools
        self.disabled_tool_actions = (
            {tool_name: set(actions) for tool_name, actions in disabled_tool_actions.items()}
            if disabled_tool_actions is not None
            else None
        )

    @staticmethod
    def _get_tool_action(tool_call: dict[str, Any]) -> str | None:
        args = tool_call.get('args')
        if not isinstance(args, dict):
            return None

        action = args.get('action')
        if not isinstance(action, str):
            return None

        return action

    def _get_parallel_restriction(self, tool_call: dict[str, Any]) -> str | None:
        tool_name = tool_call['name']

        if self.disabled_tools and tool_name in self.disabled_tools:
            return f"Tool '{tool_name}'"

        if not self.disabled_tool_actions:
            return None

        disabled_actions = self.disabled_tool_actions.get(tool_name)
        if not disabled_actions:
            return None

        action = self._get_tool_action(tool_call)
        if action in disabled_actions:
            return f"Tool '{tool_name}' action '{action}'"

        return None

    # 部分模型不支持, 用这种方法反而会报错。改为直接使用工具提示来限制。
    # @staticmethod
    # def _disable_parallel(request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
    #     return request.override(model_settings={**request.model_settings, 'parallel_tool_calls': False})
    #
    # def _disable_specific_tools_parallel(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
    #     if not self.disabled_tools:
    #         return request.override(model_settings={**request.model_settings, 'parallel_tool_calls': False})
    #
    #     tool_list = '\n'.join(f'- {t}' for t in sorted(self.disabled_tools))
    #     prompt = SPECIFIC_TOOLS_PARALLEL_PROMPT_TEMPLATE.format(tool_list=tool_list)
    #     new_content = [*list(request.system_message.content_blocks), {'type': 'text', 'text': prompt}]
    #     return request.override(
    #         model_settings={**request.model_settings, 'parallel_tool_calls': False},
    #         system_message=SystemMessage(content=new_content),
    #     )
    #
    # def wrap_model_call(
    #     self,
    #     request: ModelRequest[ContextT],
    #     handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    # ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
    #     request = self._disable_specific_tools_parallel(request)
    #     return handler(request)
    #
    # async def awrap_model_call(
    #     self,
    #     request: ModelRequest[ContextT],
    #     handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    # ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
    #     request = self._disable_specific_tools_parallel(request)
    #     return await handler(request)

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        last_message = state['messages'][-1]
        if not isinstance(last_message, AIMessage):
            return None

        if not last_message.tool_calls or len(last_message.tool_calls) == 1:
            return None

        if self.disabled_tool_actions is None and not self.disabled_tools:
            # 全局禁止并行调用
            error_msg = 'Parallel tool calls detected; this invocation has been invalidated. Only one tool call is allowed at a time!'
            return {
                'jump_to': 'model',
                'messages': [ToolMessage(content=error_msg, tool_call_id=tc['id']) for tc in last_message.tool_calls],
            }

        # 检查是否有禁止并行的工具或 manager action 被并行调用
        restricted_calls = {
            tc['id']: restriction
            for tc in last_message.tool_calls
            if (restriction := self._get_parallel_restriction(tc))
        }
        if not restricted_calls:
            return None

        # 有禁止并行的工具或 action 混入了并行调用, 截断本次调用
        messages = []
        for tc in last_message.tool_calls:
            if restriction := restricted_calls.get(tc['id']):
                msg = f'{restriction} must be called alone and cannot be used in parallel with other tools.'
            else:
                msg = 'This tool invocation has been invalidated due to the presence of tools that do not allow parallel calls.'
            messages.append(ToolMessage(content=msg, tool_call_id=tc['id']))

        return {'jump_to': 'model', 'messages': messages}

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return self.after_model(state, runtime)


class TokenLimitMiddleware(AgentMiddleware):
    def __init__(self, max_token: int = 12000):
        self.max_token = max_token

    def before_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:  # noqa: D102
        return {'checked': False}

    def before_model(self, state: TestRunState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: D102
        total_tokens = count_tokens_approximately(state['messages'])
        if not state['checked'] and total_tokens > self.max_token:
            logger.warning(f'Up to max token {total_tokens}')
            is_approved = interrupt({'max_token': total_tokens})

            return {'checked': is_approved}

        return None
