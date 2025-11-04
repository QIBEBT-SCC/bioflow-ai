from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_qwen3 import Qwen3Embeddings

from app.core.config import get_settings
from llm.llm_client import ChatOpenAIReasoning
from llm.utils.cache import cache_model


def gemini_25_flash(temperature: float = 0.7, disable_streaming: bool = False) -> BaseChatModel:
    if base_url := get_settings().llm.GEMINI.BASE_URL:
        # llm = ChatGoogleGenerativeAI(
        #     model='gemini-2.5-flash',
        #     client_options={'api_endpoint': base_url},
        #     google_api_key=get_settings().llm.GEMINI.API_KEY,
        #     temperature=temperature,
        #     transport='rest',
        # )
        provider = get_settings().llm.GEMINI.PROVIDER
        llm = ChatOpenAIReasoning(
            model=f'{provider}/gemini-2.5-flash' if provider else 'gemini-2.5-flash',
            base_url=base_url,
            api_key=get_settings().llm.GEMINI.API_KEY,
            provider='google',
            temperature=temperature,
            disable_streaming=disable_streaming,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            google_api_key=get_settings().llm.GEMINI.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
        )

    return llm


def gemini_25_pro(temperature: float = 0.7, disable_streaming: bool = False) -> BaseChatModel:
    if base_url := get_settings().llm.GEMINI.BASE_URL:
        provider = get_settings().llm.GEMINI.PROVIDER
        llm = ChatOpenAIReasoning(
            model=f'{provider}/gemini-2.5-pro' if provider else 'gemini-2.5-pro',
            base_url=base_url,
            api_key=get_settings().llm.GEMINI.API_KEY,
            provider='google',
            temperature=temperature,
            disable_streaming=disable_streaming,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-pro', google_api_key=get_settings().llm.GEMINI.API_KEY, temperature=temperature
        )

    return llm


def gpt_o4_mini(temperature: float = 0.7, disable_streaming: bool = False) -> ChatOpenAI:
    if base_url := get_settings().llm.GPT.BASE_URL:
        provider = get_settings().llm.GPT.PROVIDER
        llm = ChatOpenAIReasoning(
            model=f'{provider}/o4-mini' if provider else 'o4-mini',
            base_url=base_url,
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
        )
    else:
        llm = ChatOpenAIReasoning(
            model='o4-mini',
            base_url='https://api.openai.com/v1',
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
        )

    return llm


def gpt_5(temperature: float = 0.7, disable_streaming: bool = False, reasoning_effort: str = 'low') -> ChatOpenAI:
    if base_url := get_settings().llm.GPT.BASE_URL:
        provider = get_settings().llm.GPT.PROVIDER
        llm = ChatOpenAIReasoning(
            model=f'{provider}/gpt-5' if provider else 'gpt-5',
            base_url=base_url,
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
            reasoning_effort=reasoning_effort,
        )
    else:
        llm = ChatOpenAIReasoning(
            model='gpt-5',
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
            reasoning_effort=reasoning_effort,
        )

    return llm


def gpt_5_mini(temperature: float = 0.7, disable_streaming: bool = False, reasoning_effort: str = 'low') -> ChatOpenAI:
    if base_url := get_settings().llm.GPT.BASE_URL:
        provider = get_settings().llm.GPT.PROVIDER
        llm = ChatOpenAIReasoning(
            model=f'{provider}/gpt-5-mini' if provider else 'gpt-5-mini',
            base_url=base_url,
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
            reasoning_effort=reasoning_effort,
        )
    else:
        llm = ChatOpenAIReasoning(
            model='gpt-5-mini',
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
            reasoning_effort=reasoning_effort,
        )

    return llm


def grok_4_fast_livesearch(temperature: float = 0.7, disable_streaming: bool = False) -> ChatOpenAIReasoning:
    provider = get_settings().llm.GROK.PROVIDER
    llm = ChatOpenAIReasoning(
        model=f'{provider}/grok-4-fast-reasoning' if provider else 'grok-4-fast-reasoning',
        base_url=get_settings().llm.GROK.BASE_URL,
        api_key=get_settings().llm.GROK.API_KEY,
        provider='xai',
        extra_body={
            'search_parameters': {
                'mode': 'on',
                'return_citations': True,
                'max_search_results': 10,
                'sources': ['web'],
            }
        },
        temperature=temperature,
        disable_streaming=disable_streaming,
    )

    return llm


def deepseek_v3(temperature: float = 0.7, disable_streaming: bool = False) -> ChatOpenAI:
    provider = get_settings().llm.DEEPSEEK.PROVIDER
    llm = ChatOpenAIReasoning(
        model=f'{provider}/deepseek-chat' if provider else 'deepseek-chat',
        base_url=get_settings().llm.DEEPSEEK.BASE_URL,
        api_key=get_settings().llm.DEEPSEEK.API_KEY,
        provider='deepseek',
        temperature=temperature,
        disable_streaming=disable_streaming,
    )

    return llm


def deepseek_r1(temperature: float = 0.7, disable_streaming: bool = False) -> ChatOpenAI:
    provider = get_settings().llm.DEEPSEEK.PROVIDER
    llm = ChatOpenAIReasoning(
        model=f'{provider}/deepseek-reasoner' if provider else 'deepseek-reasoner',
        base_url=get_settings().llm.DEEPSEEK.BASE_URL,
        api_key=get_settings().llm.DEEPSEEK.API_KEY,
        provider='deepseek',
        temperature=temperature,
        disable_streaming=disable_streaming,
    )

    return llm


def glm_45_air(temperature: float = 0.7, disable_streaming: bool = False) -> ChatOpenAI:
    provider = get_settings().llm.GLM.PROVIDER
    llm = ChatOpenAIReasoning(
        model=f'{provider}/glm-4.5-air' if provider else 'glm-4.5-air',
        base_url=get_settings().llm.GLM.BASE_URL,
        api_key=get_settings().llm.GLM.API_KEY,
        provider='zai',
        temperature=temperature,
        disable_streaming=disable_streaming,
    )

    return llm


def glm_45(temperature: float = 0.7, disable_streaming: bool = False) -> ChatOpenAI:
    provider = get_settings().llm.GLM.PROVIDER
    llm = ChatOpenAIReasoning(
        model=f'{provider}/glm-4.5' if provider else 'glm-4.5',
        base_url=get_settings().llm.GLM.BASE_URL,
        api_key=get_settings().llm.GLM.API_KEY,
        provider='zai',
        temperature=temperature,
        disable_streaming=disable_streaming,
    )

    return llm


def glm_46(temperature: float = 0.7, disable_streaming: bool = False) -> ChatOpenAI:
    provider = get_settings().llm.GLM.PROVIDER
    llm = ChatOpenAIReasoning(
        model=f'{provider}/glm-4.6' if provider else 'glm-4.6',
        base_url=get_settings().llm.GLM.BASE_URL,
        api_key=get_settings().llm.GLM.API_KEY,
        provider='zai',
        temperature=temperature,
        disable_streaming=disable_streaming,
    )

    return llm


def qwen3_30b_a3b_think(temperature: float = 0.7, disable_streaming: bool = False) -> ChatOpenAIReasoning:
    llm = ChatOpenAIReasoning(
        model='qwen3-30b-a3b-thinking-2507',
        base_url=get_settings().llm.VLLM.BASE_URL,
        api_key=get_settings().llm.VLLM.API_KEY,
        provider='qwen',
        temperature=temperature,
        disable_streaming=disable_streaming,
    )

    return llm


@cache_model()
def qwen3_embedding() -> Qwen3Embeddings:
    embedding = Qwen3Embeddings(model='Qwen/Qwen3-Embedding-4B', use_modelscope=True)
    return embedding
