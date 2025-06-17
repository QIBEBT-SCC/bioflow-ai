from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from app.core.config import get_settings


def deepseek_r1_structure() -> ChatDeepSeek:
    chat = ChatDeepSeek(
        model='deepseek-reasoner', api_key=get_settings().llm.DEEPSEEK_API, disabled_params={'tool_choice': None}
    )
    return chat


def deepseek_v3() -> ChatDeepSeek:
    chat = ChatDeepSeek(model='deepseek-chat', api_key=get_settings().llm.DEEPSEEK_API)
    return chat


def gemini_25_flash() -> ChatOpenAI:
    """gemini 2.5 flash.

    目前尚不支持并行工具调用
    """

    llm = ChatOpenAI(
        model='gemini-2.5-flash-preview-05-20',
        base_url='https://aihubmix.com/v1',
        api_key=get_settings().llm.AI_HUB_MIX_KEY,
    )

    return llm


def gpt_41_mini() -> ChatOpenAI:
    llm = ChatOpenAI(
        model='gpt-4.1-mini', base_url='https://aihubmix.com/v1', api_key=get_settings().llm.AI_HUB_MIX_KEY
    )

    return llm
