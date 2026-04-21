from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT, hook_config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from langgraph.typing import ContextT
from loguru import logger
from pydantic import BaseModel, Field

from llm.states import FolderContext, StructuredResponse, UserContext
from llm.utils.token import store_token_usage


class CheckResult(BaseModel):
    checked: bool = Field(description='Has the FinalResponse you see passed the format check')
    reason: str = Field(description='Reasons for Pass/Fail check')


class CheckState(TypedDict):
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


class CheckReportMiddleware(AgentMiddleware):
    state_schema = CheckState

    def __init__(
        self,
        model: BaseChatModel,
        check_prompt: str,
    ) -> None:
        """对模型输出结果进行检查.

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

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return {'check_failed_count': 0}

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: StateT, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        last_message = cast('AIMessage', state['messages'][-1])

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

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        last_message = cast('AIMessage', state['messages'][-1])

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
        error_time: int = 3,
    ) -> None:
        super().__init__()

        self.error_time = error_time

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return {'check_failed_count': 0}

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: StateT, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        last_message = cast('AIMessage', state['messages'][-1])
        if last_message.tool_calls:
            return None

        check_failed_count = state.get('check_failed_count', 0)

        if 'file_mounts' not in state or not state['file_mounts']:
            return _handle_check_failed(
                check_failed_count,
                [
                    HumanMessage(
                        content='You have not set up file mounts yet. Use the `set_file_mounts` tool to configure them.'
                    )
                ],
                self.error_time,
            )

        if 'command_template' not in state or not state['command_template']:
            return _handle_check_failed(
                check_failed_count,
                [
                    HumanMessage(
                        content='You have not set up command template yet. Use the `set_command_detail` tool to configure it.'
                    )
                ],
                self.error_time,
            )

        # complete_command = build_command_template(
        #     command_template=state['command_template'],
        #     dynamic_params=state['dynamic_params'],
        #     immutable_static_params=state['immutable_static_params'],
        #     modifiable_static_params=state['modifiable_static_params'],
        #     file_mounts=state['file_mounts'],
        # )
        # template_params = {field_name for _, field_name, _, _ in Formatter().parse(complete_command) if field_name}
        # input_files = {mount.name for mount in state['file_mounts'] if mount.file_type == 'INPUT'}
        #
        # # # 检查1：template_params中的变量名是否只有input_files中值的以及output_dir和sample_name
        # allowed_params = input_files | {'output_dir', 'sample_name'}
        # invalid_params = template_params - allowed_params
        # if invalid_params:
        #     return _handle_check_failed(
        #         check_failed_count,
        #         [
        #             HumanMessage(
        #                 content=f'Invalid parameter(s) in command template: "{", ".join(invalid_params)}". Invoke the `set_command_detail` tool again based on the rules to make corrections.'
        #             )
        #         ],
        #         self.error_time,
        #     )
        #
        # # 检查2：input_files中的值是否在template_params中均存在
        # missing_inputs = input_files - template_params
        # if missing_inputs:
        #     return _handle_check_failed(
        #         check_failed_count,
        #         [
        #             HumanMessage(
        #                 content=f'Input file(s) not used in command: "{", ".join(missing_inputs)}". '
        #                 f'All input files must be referenced in the command. Check if the command_template and parameter settings are correct, and call the `set_command_detail` tool again to correct them.'
        #             )
        #         ],
        #         self.error_time,
        #     )

        logger.success('Check passed')
        return None

    @hook_config(can_jump_to=['model'])
    async def aafter_model(self, state: StateT, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


class CheckTaskReportMiddleware(AgentMiddleware):
    state_schema = CheckState

    def __init__(
        self,
        error_time: int = 3,
    ) -> None:
        super().__init__()

        self.error_time = error_time

    @hook_config(can_jump_to=['model'])
    def after_model(self, state: StateT, runtime: Runtime[FolderContext]) -> dict[str, Any] | None:
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
    async def aafter_model(self, state: StateT, runtime: Runtime[FolderContext]) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


class StateUpdateSchema(TypedDict):
    updated: NotRequired[bool]


class CheckStateExistMiddleware(AgentMiddleware):
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

        if last_message.tool_calls:
            return None

        # 如果这是一条不带工具调用的信息, 那么说明模型直接退出了
        # 此时检查: 1、state_name 有内容; 2、updated 为真
        if state.get(self.state_name) and state['updated']:  # type: ignore
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
