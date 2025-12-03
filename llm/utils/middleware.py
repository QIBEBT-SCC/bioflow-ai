from pathlib import Path
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings
from llm.states import TestRunContext, TestRunState


class DisableParallelToolCallsMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        request.model_settings['parallel_tool_calls'] = False
        return handler(request)

    async def awrap_model_call(self, request, handler):
        request.model_settings['parallel_tool_calls'] = False
        return await handler(request)


class CheckResult(BaseModel):
    checked: bool = Field(description='Has the FinalResponse you see passed the format check')
    reason: str = Field(description='Reasons for Pass/Fail check')


class CheckReportMiddleware(AgentMiddleware):
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
        self.check_chain = prompt | model.with_structured_output(CheckResult)

    def after_model(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        last_message = cast(AIMessage, state['messages'][-1])

        if last_message.tool_calls:
            return None

        should_retry = (
            not last_message.content or not self.check_chain.invoke({'final_response': last_message.content}).checked
        )

        if should_retry:
            logger.warning('Check failed')
            return {
                'jump_to': 'model',
                'messages': [RemoveMessage(id=last_message.id)],
            }

        logger.debug('Check passed')
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

    def after_agent(self, state: MessagesState, runtime: Runtime[TestRunContext]) -> dict[str, Any] | None:
        if runtime.context.save_message:
            import polars as pl

            message_dicts = [m.model_dump_json() for m in state['messages']]
            new_data = {
                'question': state['messages'][0].content,
                'answer': state['messages'][-1].content,
                'test_run_id': runtime.context.test_run_id,
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

    async def aafter_agent(self, state: MessagesState, runtime: Runtime[TestRunContext]) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)


class ClearContainerMiddleware(AgentMiddleware):
    def after_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        if 'container_id' in state and state['container_id']:
            if get_settings().docker.CONTAINER == 'docker':
                from app.utils.container.docker import remove_container

                remove_container(state['container_id'])
            else:
                from app.utils.container.podman import remove_container

                remove_container(state['container_id'])

            logger.debug(f'Remove {get_settings().docker.CONTAINER} containers')

    async def aafter_agent(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
