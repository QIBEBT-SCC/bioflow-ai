from typing import Any, NotRequired, TypedDict, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.runtime import Runtime
from loguru import logger
from pydantic import BaseModel, Field

from llm.states import StructuredResponse, ToolGeneratorState, UserContext
from llm.utils.formatters import tool_config_formatter
from llm.utils.token import store_token_usage


class CheckResult(BaseModel):
    checked: bool = Field(description='Has the FinalResponse you see passed the format check')
    reason: str = Field(description='Reasons for Pass/Fail check')


class CheckState(TypedDict):
    check_failed_count: NotRequired[int]


def _handle_check_failed(
    check_failed_count: int,
    messages: list[BaseMessage],
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    统一处理检查失败的逻辑

    Args:
        check_failed_count: 当前失败次数
        messages: 要返回的消息列表
        max_retries: 最大重试次数，默认为3

    Returns:
        包含跳转信息和更新后的状态

    Raises:
        Exception: 当失败次数达到最大重试次数时
    """
    if check_failed_count >= max_retries:
        raise Exception(f'Check failed {max_retries} times.')

    return {
        'jump_to': 'model',
        'check_failed_count': check_failed_count + 1,
        'messages': messages,
    }


class CheckReportMiddleware(AgentMiddleware):
    state_schema = CheckState

    def __init__(
        self,
        model: BaseChatModel,
        check_prompt: str,
    ) -> None:
        """
        Args:
            model: 用于检查任务完成状态的模型
            check_prompt: 检查提示词
        """
        super().__init__()

        prompt = ChatPromptTemplate(
            messages=[
                ('system', check_prompt),
                ('human', '<FinalResponse>\n{final_response}\n</FinalResponse>'),
            ]
        )
        self.check_chain = prompt | model.with_structured_output(CheckResult, include_raw=True)
        self.model_metadata = model.metadata

    def after_model(self, state: StateT, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        last_message = cast(AIMessage, state['messages'][-1])

        if last_message.tool_calls:
            return None

        check_failed_count = state.get('check_failed_count', 0)

        if not last_message.content:
            # 生成结果为空，重新生成
            logger.warning('Check failed. Empty message.')
            return _handle_check_failed(
                check_failed_count,
                messages=[RemoveMessage(id=last_message.id)],
            )

        check_result: StructuredResponse[CheckResult] = self.check_chain.invoke(  # type: ignore
            {'final_response': last_message.content}
        )
        store_token_usage(
            'check_report',
            token_usage=check_result['raw'].usage_metadata,
            user_id=runtime.context.user_id,
            **self.model_metadata,
        )
        if not check_result['parsed'].checked:
            # 检查未通过，将原因传回去
            logger.warning('Check failed')
            return _handle_check_failed(
                check_failed_count,
                messages=[HumanMessage(content=check_result['parsed'].reason)],
            )

        logger.success('Check passed')
        return None

    async def aafter_model(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        last_message = cast(AIMessage, state['messages'][-1])

        if last_message.tool_calls:
            return None

        check_failed_count = state.get('check_failed_count', 0)

        if not last_message.content:
            logger.warning('Check failed. Empty message.')
            return _handle_check_failed(
                check_failed_count,
                messages=[RemoveMessage(id=last_message.id)],
            )

        check_result: StructuredResponse[CheckResult] = await self.check_chain.ainvoke(  # type: ignore
            {'final_response': last_message.content}
        )
        store_token_usage(
            'check_report',
            token_usage=check_result['raw'].usage_metadata,
            user_id=runtime.context.user_id,
            **self.model_metadata,
        )
        if not check_result['parsed'].checked:
            logger.warning('Check failed')
            return _handle_check_failed(
                check_failed_count,
                messages=[HumanMessage(content=check_result['parsed'].reason)],
            )

        logger.success('Check passed')
        return None


class CheckGenerateMiddleware(AgentMiddleware):
    state_schema = CheckState

    def __init__(
        self,
        model: BaseChatModel,
        check_prompt: str,
    ) -> None:
        """
        Args:
            model: 用于检查任务完成状态的模型
            check_prompt: 检查提示词
        """
        super().__init__()

        prompt = ChatPromptTemplate(
            messages=[
                ('system', check_prompt),
                MessagesPlaceholder(variable_name='history'),
                ('human', '<TemplateOutput>\n{tool_config}\n</TemplateOutput>'),
            ]
        )
        self.check_chain = prompt | model.with_structured_output(CheckResult, include_raw=True)
        self.model_metadata = model.metadata

    def after_model(self, state: ToolGeneratorState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        last_message = cast(AIMessage, state['messages'][-1])
        if last_message.tool_calls:
            return None

        check_failed_count = state.get('check_failed_count', 0)

        if 'file_mounts' not in state or not state['file_mounts']:
            return _handle_check_failed(
                check_failed_count,
                messages=[
                    HumanMessage(
                        content='You have not set up file mounts yet. Use the `set_file_mounts` tool to configure them.'
                    )
                ],
            )

        if 'command_template' not in state or not state['command_template']:
            return _handle_check_failed(
                check_failed_count,
                messages=[
                    HumanMessage(
                        content='You have not set up command template yet. Use the `set_command_detail` tool to configure it.'
                    )
                ],
            )

        # 保留前两条消息，分别是用户原始需求和运行报告
        init_messages = state['messages'][:2]

        tool_config = tool_config_formatter(state)

        check_result: StructuredResponse[CheckResult] = self.check_chain.invoke(  # type: ignore
            {'history': init_messages, 'tool_config': tool_config}
        )
        store_token_usage(
            'check_tool_generate',
            token_usage=check_result['raw'].usage_metadata,
            user_id=runtime.context.user_id,
            **self.model_metadata,
        )
        if not check_result['parsed'].checked:
            # 检查未通过，将原因传回去
            logger.warning('Check failed')
            return _handle_check_failed(
                check_failed_count,
                messages=[HumanMessage(content=check_result['parsed'].reason)],
            )

        logger.success('Check passed')
        return None

    async def aafter_model(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        last_message = cast(AIMessage, state['messages'][-1])
        if last_message.tool_calls:
            return None

        check_failed_count = state.get('check_failed_count', 0)

        if 'file_mounts' not in state or not state['file_mounts']:
            return _handle_check_failed(
                check_failed_count,
                messages=[
                    HumanMessage(
                        content='You have not set up file mounts yet. Use the `set_file_mounts` tool to configure them.'
                    )
                ],
            )

        if 'command_template' not in state or not state['command_template']:
            return _handle_check_failed(
                check_failed_count,
                messages=[
                    HumanMessage(
                        content='You have not set up command template yet. Use the `set_command_detail` tool to configure it.'
                    )
                ],
            )

        init_messages = state['messages'][:2]
        tool_config = tool_config_formatter(state)

        check_result: StructuredResponse[CheckResult] = await self.check_chain.ainvoke(  # type: ignore
            {'history': init_messages, 'tool_config': tool_config}
        )
        store_token_usage(
            'check_tool_generate',
            token_usage=check_result['raw'].usage_metadata,
            user_id=runtime.context.user_id,
            **self.model_metadata,
        )
        if not check_result['parsed'].checked:
            logger.warning('Check failed')
            return _handle_check_failed(
                check_failed_count,
                messages=[HumanMessage(content=check_result['parsed'].reason)],
            )

        logger.success('Check passed')
        return None
