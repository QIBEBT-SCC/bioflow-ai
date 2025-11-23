from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from loguru import logger
from pydantic import BaseModel, Field

from app.utils.container.podman import remove_container
from llm.states import TestRunContext, TestRunState

DEFAULT_CHECK = """Based on the task description above and the conversation history below, please judge whether the current task has been completed.

Consider:
1. Have all the objectives mentioned in the task system prompt been achieved?
2. Have all necessary steps been executed?
3. Has the user's request been fully addressed?

Return True only if the task is completely finished."""
DEFAULT_CONTINUE = """Please continue to call the tool to complete the task."""


class DisableParallelToolCallsMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        request.model_settings['parallel_tool_calls'] = False
        return handler(request)

    async def awrap_model_call(self, request, handler):
        request.model_settings['parallel_tool_calls'] = False
        return await handler(request)


class CheckResult(BaseModel):
    has_end: bool = Field(
        description='Whether the task has been completed. True indicates the task is completed, False indicates the task is not completed and needs to continue execution'
    )


class CheckReportMiddleware(AgentMiddleware):
    def __init__(
        self,
        model: BaseChatModel,
        task_description: str,
        check_prompt: str = DEFAULT_CHECK,
        continue_message: str = DEFAULT_CONTINUE,
    ) -> None:
        """
        Args:
            model: 用于检查任务完成状态的模型
            task_description: 主智能体的任务描述
            check_prompt: 检查提示词
            continue_message: 任务未完成时的提示消息
        """
        super().__init__()

        self.continue_message = continue_message

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=f'{check_prompt}\n\n<TaskDescription>\n{task_description}\n</TaskDescription>'),
                MessagesPlaceholder(variable_name='history'),
            ]
        )
        self.check_chain = prompt | model.with_structured_output(CheckResult)

    def after_model(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state['messages'][-1]

        if not last_message.tool_calls:
            response = self.check_chain.invoke({'history': state['messages']})
            has_end = response.has_end

            if not has_end:
                return {
                    'jump_to': 'model',
                    'messages': [HumanMessage(content=self.continue_message)],
                }

        return None

    async def aafter_model(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state['messages'][-1]

        if not last_message.tool_calls:
            response = await self.check_chain.ainvoke({'history': state['messages']})
            has_end = response.has_end

            if not has_end:
                return {
                    'jump_to': 'model',
                    'messages': [HumanMessage(content=self.continue_message)],
                }

        return None


class TokenLimitMiddleware(AgentMiddleware):
    def __init__(self, max_token: int = 12000):
        self.max_token = max_token

    def before_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        return {'checked': False}

    def before_model(self, state: TestRunState, runtime: Runtime) -> dict[str, Any] | None:
        total_tokens = count_tokens_approximately(state['messages'])
        if not state['checked']:
            if total_tokens > self.max_token:
                logger.warning(f'Up to max token {total_tokens}')
                is_approved = interrupt({'max_token': total_tokens})

                return {'checked': is_approved}

        return None


class SaveMessagesMiddleware(AgentMiddleware):
    def __init__(self, dataset_file: str = 'test_run.parquet'):
        self.dataset_file = dataset_file

    def _save_messages(self, state: MessagesState, run_id: str):
        import polars as pl

        message_dicts = [m.model_dump_json() for m in state['messages']]
        new_data = {
            'question': state['messages'][0].content,
            'test_run_id': run_id,
            'messages': message_dicts,
        }
        df_new = pl.DataFrame([new_data])

        if Path(self.dataset_file).exists():
            df_existing = pl.read_parquet(self.dataset_file)
            df_existing = df_existing.vstack(df_new)
            df_existing.write_parquet(self.dataset_file, compression='snappy')
        else:
            df_new.write_parquet(self.dataset_file, compression='snappy')

        logger.debug(f'Saved new conversation record, containing {len(message_dicts)} conversations.')

    def after_agent(self, state: MessagesState, runtime: Runtime[TestRunContext]) -> dict[str, Any] | None:
        if runtime.context.save_message:
            self._save_messages(state, runtime.context.test_run_id)

    async def aafter_agent(self, state: MessagesState, runtime: Runtime[TestRunContext]) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)


class ClearContainerMiddleware(AgentMiddleware):
    def after_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        if 'container_id' in state and state['container_id']:
            remove_container(state['container_id'])
            logger.debug('Cleare residual containers')

    async def aafter_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
