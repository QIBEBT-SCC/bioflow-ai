from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
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


def doubao_seed_16() -> ChatOpenAI:
    llm = ChatOpenAI(
        model='doubao-seed-1-6-250615',
        base_url='https://ark.cn-beijing.volces.com/api/v3',
        api_key=get_settings().llm.VOLCENGINE_API,
    )
    return llm


def gemini_25_flash() -> ChatGoogleGenerativeAI:
    """gemini 2.5 flash."""

    # llm = ChatGoogleGenerativeAI(
    #     model='gemini-2.5-flash',
    #     client_options={
    #         'api_endpoint': 'https://aihubmix.com/gemini',
    #         'api_key': get_settings().llm.AI_HUB_MIX_KEY.get_secret_value(),
    #     },
    #     transport='rest',
    # )
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=get_settings().llm.AI_STUDIO_KEY)

    return llm


def gemini_25_pro() -> ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro', google_api_key=get_settings().llm.AI_STUDIO_KEY)

    return llm


def gpt_41_mini() -> ChatOpenAI:
    llm = ChatOpenAI(
        model='gpt-4.1-mini', base_url='https://aihubmix.com/v1', api_key=get_settings().llm.AI_HUB_MIX_KEY
    )

    return llm


def gpt_o4_mini() -> ChatOpenAI:
    llm = ChatOpenAI(model='o4-mini', base_url='https://aihubmix.com/v1', api_key=get_settings().llm.AI_HUB_MIX_KEY)

    return llm
