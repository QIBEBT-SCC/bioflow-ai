from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.config import get_settings


def deepseek_r1_structure() -> ChatDeepSeek:
    chat = ChatDeepSeek(
        model='deepseek-reasoner', api_key=get_settings().llm.DEEPSEEK_KEY, disabled_params={'tool_choice': None}
    )
    return chat


def deepseek_v3() -> ChatDeepSeek:
    chat = ChatDeepSeek(model='deepseek-chat', api_key=get_settings().llm.DEEPSEEK_KEY)
    return chat


def doubao_seed_16() -> ChatOpenAI:
    llm = ChatOpenAI(
        model='doubao-seed-1-6-250615',
        base_url='https://ark.cn-beijing.volces.com/api/v3',
        api_key=get_settings().llm.VOLCENGINE_KEY,
    )
    return llm


def gemini_25_flash() -> ChatGoogleGenerativeAI:
    if get_settings().llm.TRANSMIT:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            client_options={
                'api_endpoint': 'https://aihubmix.com/gemini',
                'api_key': get_settings().llm.AI_HUB_MIX_KEY.get_secret_value(),
            },
            transport='rest',
        )
    else:
        llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=get_settings().llm.AI_STUDIO_KEY)

    return llm


def gemini_25_flash_no_think() -> ChatGoogleGenerativeAI:
    if get_settings().llm.TRANSMIT:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            client_options={
                'api_endpoint': 'https://aihubmix.com/gemini',
                'api_key': get_settings().llm.AI_HUB_MIX_KEY.get_secret_value(),
            },
            transport='rest',
            thinking_budget=0,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash', google_api_key=get_settings().llm.AI_STUDIO_KEY, thinking_budget=0, max_retries=2
        )
    return llm


def gemini_25_pro() -> ChatGoogleGenerativeAI:
    if get_settings().llm.TRANSMIT:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-pro',
            client_options={
                'api_endpoint': 'https://aihubmix.com/gemini',
                'api_key': get_settings().llm.AI_HUB_MIX_KEY.get_secret_value(),
            },
            transport='rest',
        )
    else:
        llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro', google_api_key=get_settings().llm.AI_STUDIO_KEY)

    return llm


def gpt_41_mini() -> ChatOpenAI:
    if get_settings().llm.TRANSMIT:
        llm = ChatOpenAI(
            model='gpt-4.1-mini', base_url='https://aihubmix.com/v1', api_key=get_settings().llm.AI_HUB_MIX_KEY
        )
    else:
        llm = ChatOpenAI(model='gpt-4.1-mini', api_key=get_settings().llm.OPENAI_KEY)

    return llm


def gpt_o4_mini() -> ChatOpenAI:
    if get_settings().llm.TRANSMIT:
        llm = ChatOpenAI(model='o4-mini', base_url='https://aihubmix.com/v1', api_key=get_settings().llm.AI_HUB_MIX_KEY)
    else:
        llm = ChatOpenAI(model='o4-mini', api_key=get_settings().llm.OPENAI_KEY)

    return llm
