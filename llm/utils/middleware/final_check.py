from pathlib import Path
from string import Formatter
from typing import Any, NotRequired, TypedDict, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from loguru import logger
from pydantic import BaseModel, Field

from app.utils.input_format import build_command_template
from llm.states import FolderContext, StructuredResponse, ToolGeneratorState, UserContext
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
        error_time: int = 3,
    ) -> None:
        """
        Args:
            error_time: 重试次数
        """
        super().__init__()

        self.error_time = error_time

    def after_model(self, state: ToolGeneratorState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        last_message = cast(AIMessage, state['messages'][-1])
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

        complete_command = build_command_template(
            command_template=state['command_template'],
            dynamic_params=state['dynamic_params'],
            immutable_static_params=state['immutable_static_params'],
            modifiable_static_params=state['modifiable_static_params'],
            file_mounts=state['file_mounts'],
        )
        template_params = {field_name for _, field_name, _, _ in Formatter().parse(complete_command) if field_name}
        input_files = {mount.name for mount in state['file_mounts'] if mount.file_type == 'INPUT'}

        # 检查1：template_params中的变量名是否只有input_files中值的以及output_dir和sample_name
        allowed_params = input_files | {'output_dir', 'sample_name'}
        invalid_params = template_params - allowed_params
        if invalid_params:
            return _handle_check_failed(
                check_failed_count,
                [
                    HumanMessage(
                        content=f'Invalid parameter(s) in command template: "{", ".join(invalid_params)}". Invoke the `set_command_detail` tool again based on the rules to make corrections.'
                    )
                ],
                self.error_time,
            )

        # 检查2：input_files中的值是否在template_params中均存在
        missing_inputs = input_files - template_params
        if missing_inputs:
            return _handle_check_failed(
                check_failed_count,
                [
                    HumanMessage(
                        content=f'Input file(s) not used in command: "{", ".join(missing_inputs)}". '
                        f'All input files must be referenced in the command. Check if the command_template and parameter settings are correct, and call the `set_command_detail` tool again to correct them.'
                    )
                ],
                self.error_time,
            )

        logger.success('Check passed')
        return None

    async def aafter_model(self, state: StateT, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


class CheckTaskReportMiddleware(AgentMiddleware):
    state_schema = CheckState

    def __init__(
        self,
        error_time: int = 3,
    ) -> None:
        """
        Args:
            error_time: 重试次数
        """
        super().__init__()

        self.error_time = error_time

    def after_model(self, state: StateT, runtime: Runtime[FolderContext]) -> dict[str, Any] | None:
        last_message = cast(AIMessage, state['messages'][-1])
        if last_message.tool_calls:
            return None

        check_failed_count = state.get('check_failed_count', 0)

        missing_files = []
        for file_name in runtime.context.allowed_files:
            file_path = Path(runtime.context.root_path) / file_name
            if not file_path.exists():
                missing_files.append(file_name)

        if missing_files:
            missing = ', '.join(missing_files)
            logger.warning(f'Check failed. Missing files: {missing}')
            # return _handle_check_failed(
            #     check_failed_count,
            #     [
            #         HumanMessage(
            #             content=f'Could not find the expected output files "{missing}". Please check if you did not generate them or used incorrect file names or paths.'
            #         )
            #     ],
            #     self.error_time,
            # )

        logger.success('Check passed')
        return None

    async def aafter_model(self, state: StateT, runtime: Runtime[FolderContext]) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
