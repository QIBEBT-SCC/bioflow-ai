from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from langgraph.typing import ContextT
from loguru import logger
from pydantic import BaseModel, Field


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

        if not last_message.content:
            # 生成结果为空，重新生成
            logger.warning('Check failed. Empty message.')
            return {
                'jump_to': 'model',
                'messages': [RemoveMessage(id=last_message.id)],
            }

        check_result = self.check_chain.invoke({'final_response': last_message.content})
        if not check_result.checked:
            # 检查未通过，将原因传回去
            logger.warning('Check failed')
            return {
                'jump_to': 'model',
                'messages': [HumanMessage(content=check_result.reason)],
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

        check_result = await self.check_chain.invoke({'final_response': last_message.content})
        if not check_result.checked:
            logger.warning('Check failed')
            return {
                'jump_to': 'model',
                'messages': [HumanMessage(content=check_result.reason)],
            }

        logger.debug('Check passed')
        return None
