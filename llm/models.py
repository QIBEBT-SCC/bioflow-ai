from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel, ModelProfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_qwen3 import Qwen3Embeddings
from pydantic import SecretStr
from sqlmodel import Session

import app.crud.setting.llm_model as model_crud
import app.crud.setting.llm_setting as setting_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.setting import ProviderType
from llm.llm_client import ChatDeepseekReasoning, ChatOpenAIReasoning
from llm.utils.cache import cache_model


def _get_model_from_db(model_id: int) -> BaseChatModel:
    with Session(engine) as session:
        model_config = model_crud.get(session, model_id)
        assert model_config.provider.api_key is not None, 'No available api_key has been set yet!'
        proxy = get_settings().network.get_proxy_url() if model_config.provider.use_proxy else None

        match model_config.provider.provider_type:
            case ProviderType.ANTHROPIC:
                model = ChatAnthropic(
                    model=model_config.name,
                    api_key=SecretStr(model_config.provider.api_key),
                    anthropic_proxy=proxy,
                )
            case ProviderType.GOOGLE:
                client_args = {'proxy': proxy} if proxy else None
                model = ChatGoogleGenerativeAI(
                    model=model_config.name,
                    api_key=SecretStr(model_config.provider.api_key),
                    client_args=client_args,
                )
            case ProviderType.OPENAI:
                if model_config.provider.base_url is None:
                    model = ChatOpenAI(
                        model=model_config.name,
                        api_key=SecretStr(model_config.provider.api_key),
                        openai_proxy=proxy,
                    )
                else:
                    if 'deepseek' in model_config.name.lower():
                        model = ChatDeepseekReasoning(
                            model=model_config.name,
                            base_url=model_config.provider.base_url,
                            api_key=SecretStr(model_config.provider.api_key),
                            openai_proxy=proxy,
                        )
                    else:
                        model = ChatOpenAIReasoning(
                            model=model_config.name,
                            base_url=model_config.provider.base_url,
                            api_key=SecretStr(model_config.provider.api_key),
                            openai_proxy=proxy,
                        )
            case _:
                raise ValueError('invalid provider type')

        model.metadata = {
            'model_id': model_id,
            'input_price': model_config.input_price,
            'output_price': model_config.output_price,
            'cache_read_price': model_config.cache_read_price,
        }
        return model


def get_agent_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'agent_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'agent_model'
    return model


def get_chat_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'chat_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'chat_model'
    return model


def get_vision_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'vision_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'vision_model'
    return model


def get_coding_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'coding_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'coding_model'
    return model


def get_simple_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'simple_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'simple_model'
    return model


def get_long_context_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'long_context_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'long_context_model'
    return model


def get_high_performance_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'high_performance_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'high_performance_model'
    return model


def gemini_25_flash(temperature: float = 0.7, disable_streaming: bool = False) -> BaseChatModel:
    if base_url := get_settings().llm.GEMINI.BASE_URL:
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


def gemini_3_pro(temperature: float = 1.0, disable_streaming: bool = False) -> ChatOpenAIReasoning:
    if base_url := get_settings().llm.GEMINI.BASE_URL:
        provider = get_settings().llm.GEMINI.PROVIDER
        llm = ChatOpenAIReasoning(
            model=f'{provider}/gemini-3-pro-preview' if provider else 'gemini-3-pro-preview',
            base_url=base_url,
            api_key=get_settings().llm.GEMINI.API_KEY,
            provider='google',
            temperature=temperature,
            reasoning_effort='low',
            disable_streaming=disable_streaming,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model='gemini-3-pro-preview', google_api_key=get_settings().llm.GEMINI.API_KEY, temperature=temperature
        )

    return llm


def gpt_5(temperature: float = 0.7, disable_streaming: bool = False, reasoning_effort: str = 'low') -> ChatOpenAI:
    if base_url := get_settings().llm.GPT.BASE_URL:
        provider = get_settings().llm.GPT.PROVIDER
        llm = ChatOpenAIReasoning(
            model=f'{provider}/gpt-5.1' if provider else 'gpt-5.1',
            base_url=base_url,
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
            reasoning_effort=reasoning_effort,
            openai_proxy=get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None,
        )
    else:
        llm = ChatOpenAIReasoning(
            model='gpt-5.1',
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
            reasoning_effort=reasoning_effort,
            openai_proxy=get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None,
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
            openai_proxy=get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None,
        )
    else:
        llm = ChatOpenAIReasoning(
            model='gpt-5-mini',
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
            reasoning_effort=reasoning_effort,
            openai_proxy=get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None,
        )

    return llm


def gpt_51_codex_mini(temperature: float = 0.7, disable_streaming: bool = False, reasoning_effort: str = 'low'):
    if base_url := get_settings().llm.GPT.BASE_URL:
        provider = get_settings().llm.GPT.PROVIDER
        llm = ChatOpenAIReasoning(
            model=f'{provider}/gpt-5.1-codex-mini' if provider else 'gpt-5.1-codex-mini',
            base_url=base_url,
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
            reasoning_effort=reasoning_effort,
            openai_proxy=get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None,
        )
    else:
        llm = ChatOpenAIReasoning(
            model='gpt-5.1-codex-mini',
            api_key=get_settings().llm.GPT.API_KEY,
            temperature=temperature,
            disable_streaming=disable_streaming,
            reasoning_effort=reasoning_effort,
            openai_proxy=get_settings().network.get_proxy_url() if get_settings().network.USE_PROXY else None,
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
    llm = ChatDeepseekReasoning(
        model=f'{provider}/deepseek-reasoner' if provider else 'deepseek-reasoner',
        base_url=get_settings().llm.DEEPSEEK.BASE_URL,
        api_key=get_settings().llm.DEEPSEEK.API_KEY,
        extra_body={'thinking': {'type': 'enabled'}},
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
    embedding = Qwen3Embeddings(
        model=get_settings().knowledge_base.EMBEDDING_MODEL,
        use_modelscope=get_settings().knowledge_base.USE_MODELSCOPE,
        device=get_settings().knowledge_base.EMBEDDING_DEVICE,
    )
    return embedding
