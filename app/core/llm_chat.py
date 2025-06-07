from langchain_deepseek import ChatDeepSeek

from app.core.config import get_settings


def deepseek_r1() -> ChatDeepSeek:
    chat = ChatDeepSeek(
        model='deepseek-reasoner', api_key=get_settings().llm.DEEPSEEK_API, disabled_params={'tool_choice': None}
    )
    return chat
