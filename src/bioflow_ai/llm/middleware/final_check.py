from pathlib import Path
from typing import Any, NotRequired, cast

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.typing import ContextT
from loguru import logger

from bioflow_ai.llm.context import FolderContext


class CheckState(AgentState):
    check_failed_count: NotRequired[int]


class MaxCheckError(Exception):
    pass


def _handle_check_failed(
    check_failed_count: int,
    messages: list[BaseMessage],
    max_retries: int = 3,
) -> dict[str, Any]:
    """统一处理检查失败的逻辑.

    Args:
        check_failed_count: 当前失败次数
        messages: 要返回的消息列表
        max_retries: 最大重试次数, 默认为3

    Returns:
        包含跳转信息和更新后的状态

    Raises:
        Exception: 当失败次数达到最大重试次数时
    """
    if check_failed_count >= max_retries:
        err_msg = f'Check failed {max_retries} times.'
        raise MaxCheckError(err_msg)

    return {
        'jump_to': 'model',
        'check_failed_count': check_failed_count + 1,
        'messages': messages,
    }


class CheckTaskReportMiddleware[ContextT: FolderContext](AgentMiddleware[AgentState[Any], ContextT]):
    state_schema = CheckState

    def __init__(
        self,
        error_time: int = 3,
    ) -> None:
        super().__init__()

        self.error_time = error_time

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: CheckState, runtime: Runtime[FolderContext]) -> dict[str, Any] | None:  # noqa: D102
        last_message = cast('AIMessage', state['messages'][-1])
        if last_message.tool_calls:
            return None

        check_failed_count = state.get('check_failed_count', 0)

        summary_md = Path(runtime.context.output_path) / 'summary.md'
        if summary_md.exists():
            logger.success('Check passed')
            return None

        logger.warning('Check failed. summary.md not fount')
        return _handle_check_failed(
            check_failed_count,
            [
                HumanMessage(
                    content='The summary.md file is not found in the data/output directory. Please ensure the report has been written to this file and that the file name is correct, including its case sensitivity.'
                )
            ],
            self.error_time,
        )

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state: CheckState, runtime: Runtime[FolderContext]) -> dict[str, Any] | None:  # noqa: D102
        return self.after_model(state, runtime)


class StateUpdateSchema(AgentState):
    updated: NotRequired[bool]


class CheckStateExistMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """正常来说, 模型只能由ask human确认之后退出, 不会自己走到end loop, 也就是说只要模型自己结束了循环一定是出问题了."""

    state_schema = StateUpdateSchema

    def __init__(self, state_name: str, tool_name: str):
        self.state_name = state_name
        self.tool_name = tool_name

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: D102
        return {'updated': False}

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:  # noqa: D102
        last_message = state['messages'][-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return None

        # 如果这是一条不带工具调用的信息, 那么说明模型直接退出了
        # 此时检查: 1、state_name 有内容; 2、updated 为真
        if state.get(self.state_name) and state['updated']:
            return None

        return {
            'jump_to': 'model',
            'messages': [
                HumanMessage(
                    content=f'Before concluding the task, first invoke the `{self.tool_name}` tool to write or updated {self.state_name.replace("_", " ")}.'
                )
            ],
        }

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:  # noqa: D102
        return self.after_model(state, runtime)


class RequireToolCallMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """拦截不带工具调用的纯文本回复.

    适用于只能通过工具 (如 confirm_plan 设置 should_end) 结束的智能体: 正常流程中模型不会直接输出纯文本,
    一旦出现即说明它试图跳过流程提前结束, 此时附上提醒打回模型. 连续失败超过 ``max_retries`` 次时抛出 MaxCheckError.
    """

    state_schema = CheckState

    def __init__(self, reminder: str, max_retries: int = 3) -> None:
        super().__init__()

        self.reminder = reminder
        self.max_retries = max_retries

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: CheckState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: D102
        last_message = state['messages'][-1]
        check_failed_count = state.get('check_failed_count', 0)

        if getattr(last_message, 'tool_calls', None):
            return {'check_failed_count': 0} if check_failed_count else None

        logger.warning('Model replied without a tool call, sending it back.')
        return _handle_check_failed(check_failed_count, [HumanMessage(content=self.reminder)], self.max_retries)

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state: CheckState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: D102
        return self.after_model(state, runtime)
