from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_qwen3 import Qwen3Embeddings

from app.core.config import get_settings
from llm.utils.cache import cache_model


def deepseek_r1_structure(temperature: float = 0.7) -> BaseChatModel:
    if base_url := get_settings().llm.DEEPSEEK.BASE_URL:
        provider = get_settings().llm.DEEPSEEK.PROVIDER
        llm = ChatOpenAI(
            model=f'{provider}/deepseek-reasoner' if provider else 'deepseek-reasoner',
            base_url=base_url,
            api_key=get_settings().llm.DEEPSEEK.API_KEY,
            temperature=temperature,
        )
    else:
        llm = ChatDeepSeek(
            model='deepseek-reasoner',
            api_key=get_settings().llm.DEEPSEEK.API_KEY,
            temperature=temperature,
            disabled_params={'tool_choice': None},
        )

    return llm


def deepseek_v3(temperature: float = 0.7) -> ChatDeepSeek:
    if base_url := get_settings().llm.DEEPSEEK.BASE_URL:
        provider = get_settings().llm.DEEPSEEK.PROVIDER
        llm = ChatOpenAI(
            model=f'{provider}/deepseek-chat' if provider else 'deepseek-chat',
            base_url=base_url,
            api_key=get_settings().llm.DEEPSEEK.API_KEY,
            temperature=temperature,
        )
    else:
        llm = ChatDeepSeek(model='deepseek-chat', api_key=get_settings().llm.DEEPSEEK.API_KEY, temperature=temperature)

    return llm


def gemini_25_flash(temperature: float = 0.7) -> BaseChatModel:
    if base_url := get_settings().llm.GEMINI.BASE_URL:
        # llm = ChatGoogleGenerativeAI(
        #     model='gemini-2.5-flash',
        #     client_options={'api_endpoint': base_url},
        #     google_api_key=get_settings().llm.GEMINI.API_KEY,
        #     temperature=temperature,
        #     transport='rest',
        # )
        provider = get_settings().llm.GEMINI.PROVIDER
        llm = ChatOpenAI(
            model=f'{provider}/gemini-2.5-flash' if provider else 'gemini-2.5-flash',
            base_url=base_url,
            api_key=get_settings().llm.GEMINI.API_KEY,
            temperature=temperature,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash', google_api_key=get_settings().llm.GEMINI.API_KEY, temperature=temperature
        )

    return llm


def gemini_25_pro(temperature: float = 0.7) -> BaseChatModel:
    if base_url := get_settings().llm.GEMINI.BASE_URL:
        provider = get_settings().llm.GEMINI.PROVIDER
        llm = ChatOpenAI(
            model=f'{provider}/gemini-2.5-pro' if provider else 'gemini-2.5-pro',
            base_url=base_url,
            api_key=get_settings().llm.GEMINI.API_KEY,
            temperature=temperature,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-pro', google_api_key=get_settings().llm.GEMINI.API_KEY, temperature=temperature
        )

    return llm


def gpt_41_mini(temperature: float = 0.7) -> ChatOpenAI:
    if base_url := get_settings().llm.GPT.BASE_URL:
        provider = get_settings().llm.GPT.PROVIDER
        llm = ChatOpenAI(
            model=f'{provider}/gpt-4.1-mini' if provider else 'gpt-4.1-mini',
            base_url=base_url,
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
        )
    else:
        llm = ChatOpenAI(model='gpt-4.1-mini', api_key=get_settings().llm.GPT.API_KEY, temperature=temperature)

    return llm


def gpt_o4_mini(temperature: float = 0.7) -> ChatOpenAI:
    if base_url := get_settings().llm.GPT.BASE_URL:
        provider = get_settings().llm.GPT.PROVIDER
        llm = ChatOpenAI(
            model=f'{provider}/o4-mini' if provider else 'o4-mini',
            base_url=base_url,
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
        )
    else:
        llm = ChatOpenAI(model='o4-mini', api_key=get_settings().llm.GPT.API_KEY, temperature=temperature)

    return llm

def gpt_5_mini(temperature: float = 0.7) -> ChatOpenAI:
    if base_url := get_settings().llm.GPT.BASE_URL:
        provider = get_settings().llm.GPT.PROVIDER
        llm = ChatOpenAI(
            model=f'{provider}/gpt-5-mini' if provider else 'gpt-5-mini',
            base_url=base_url,
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
        )
    else:
        llm = ChatOpenAI(model='gpt-5-mini', api_key=get_settings().llm.GPT.API_KEY, temperature=temperature)

    return llm

@cache_model()
def qwen3_embedding() -> Qwen3Embeddings:
    embedding = Qwen3Embeddings(use_modelscope=True)
    return embedding
