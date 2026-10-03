from decimal import Decimal

from langchain_core.messages import UsageMetadata
from loguru import logger
from sqlalchemy.orm import Session

from bioflow_ai.infrastructure.database import engine
from bioflow_ai.models.setting import LLMStatistic
from bioflow_ai.services import llm_setting as setting_service


def store_token_usage(
    agent_name: str,
    user_id: int,
    thread_id: str | None,
    *,
    setting_key: str,
    model_id: int,
    input_price: Decimal,
    output_price: Decimal,
    cache_read_price: Decimal,
    token_usage: UsageMetadata | None,
) -> None:
    """Persist token usage metrics and associated billing cost.

    Args:
        agent_name: Identifier for the agent whose tokens were consumed.
        model_id: Primary key of the LLM model generating the usage.
        setting_key: Configuration key tying the statistic to a specific setting.
        user_id: Identifier for the user associated with this usage.
        thread_id: Session id.
        token_usage: Usage metadata containing token counts and cache details.
        input_price: Price per regular input token, expressed in micro-units.
        output_price: Price per output token, expressed in micro-units.
        cache_read_price: Discounted price per cached input token, in micro-units.

    Returns:
        None
    """
    if token_usage:
        input_tokens = token_usage.get('input_tokens', 0)
        output_tokens = token_usage.get('output_tokens', 0)
        cache_read = token_usage['input_token_details'].get('cache_read', 0)
        price = (
            (input_tokens - cache_read) * input_price + cache_read * cache_read_price + output_tokens * output_price
        ) / Decimal(1000000)

        statistic_data = LLMStatistic(
            agent_name=agent_name,
            model_id=model_id,
            setting_key=setting_key,
            user_id=user_id,
            thread=thread_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            price=price,
        )
        with Session(engine) as session:
            setting_service.record_statistic(session, statistic_data)
    else:
        logger.warning('token usage is empty')
