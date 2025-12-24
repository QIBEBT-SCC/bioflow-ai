from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from loguru import logger
from pydantic import BaseModel, Field

from llm.states import StructuredResponse, UserContext
from llm.utils.token import store_token_usage


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
        self.check_chain = prompt | model.with_structured_output(CheckResult, include_raw=True)
        self.model_metadata = model.metadata

    def after_model(self, state: StateT, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        last_message = cast(AIMessage, state['messages'][-1])

        if last_message.tool_calls:
            return None

        if not last_message.content:
            # 生成结果为空，重新生成
            logger.warning('Check failed. Empty message.')
            return {
                'jump_to': 'model',
                'messages': [RemoveMessage(id=last_message.id)],
            }

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
            return {
                'jump_to': 'model',
                'messages': [HumanMessage(content=check_result['parsed'].reason)],
            }

        logger.debug('Check passed')
        return None

    async def aafter_model(self, state: StateT, runtime: Runtime) -> dict[str, Any] | None:
        last_message = cast(AIMessage, state['messages'][-1])

        if last_message.tool_calls:
            return None

        if not last_message.content:
            logger.warning('Check failed. Empty message.')
            return {
                'jump_to': 'model',
                'messages': [RemoveMessage(id=last_message.id)],
            }

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
            return {
                'jump_to': 'model',
                'messages': [HumanMessage(content=check_result['parsed'].reason)],
            }

        logger.debug('Check passed')
        return None
