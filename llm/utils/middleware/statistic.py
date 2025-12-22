from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import UsageMetadata
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime
from sqlmodel import Session

import app.crud.setting.llm_statistic as statistic_crud
from app.core.session import engine
from app.models.setting import LLMStatistic


class TokenStatistic(AgentMiddleware):
    def __init__(
        self,
        agent_name: str,
        model_id: int,
        setting_key: str,
        input_price: float,
        output_price: float,
        cache_read_price: float,
    ):
        self.agent_name = agent_name
        self.model_id = model_id
        self.setting_key = setting_key
        self.input_price = input_price
        self.output_price = output_price
        self.cache_read_price = cache_read_price

    def after_model(self, state: MessagesState, runtime: Runtime) -> dict[str, Any] | None:
        token_usage: UsageMetadata = state['messages'][-1].usage_metadata
        input_tokens = token_usage.get('input_tokens', 0)
        output_tokens = token_usage.get('output_tokens', 0)
        cache_read = token_usage['input_token_details'].get('cache_read', 0)
        price = (
            (input_tokens - cache_read) * self.input_price
            + cache_read * self.cache_read_price
            + output_tokens * self.output_price
        ) / 1000000

        statistic_data = LLMStatistic(
            agent_name=self.agent_name,
            model_id=self.model_id,
            setting_key=self.setting_key,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            price=price,
        )
        with Session(engine) as session:
            statistic_crud.insert(session, statistic_data)
