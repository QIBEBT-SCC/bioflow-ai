from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from sqlmodel import Session

import app.crud.setting.llm_model as model_crud
import app.crud.setting.llm_setting as setting_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.setting import LLMModel, ModelRole, ProviderType
from llm.llm_client import ChatDeepseekReasoning, ChatOpenAIReasoning
from llm.utils.cache import cache_model

_model_cache: dict[ModelRole, BaseChatModel] = {}


class NoApikeyError(ValueError):
    pass


def _get_model_from_db(session: Session, model_id: int) -> tuple[BaseChatModel, LLMModel]:
    model_config = model_crud.get(session, model_id)

    api_key = model_config.provider.api_key if model_config.provider.api_key else ''
    proxy = get_settings().network.get_proxy_url() if model_config.provider.use_proxy else None

    match model_config.provider.provider_type:
        case ProviderType.ANTHROPIC:
            model = ChatAnthropic(
                model_name=model_config.name,
                api_key=SecretStr(api_key),
                base_url='https://api.anthropic.com'
                if model_config.provider.base_url is None
                else model_config.provider.base_url,
                anthropic_proxy=proxy,
            )
        case ProviderType.GOOGLE:
            client_args = {'proxy': proxy} if proxy else None
            if model_config.provider.base_url is None:
                model = ChatGoogleGenerativeAI(
                    model=model_config.name,
                    api_key=SecretStr(api_key),
                    client_args=client_args,
                )
            else:
                model = ChatGoogleGenerativeAI(
                    model=model_config.name,
                    api_key=SecretStr(api_key),
                    base_url=model_config.provider.base_url,
                    client_args=client_args,
                )
        case ProviderType.OPENAI:
            if model_config.provider.base_url is None:
                model = ChatOpenAI(
                    model=model_config.name,
                    api_key=SecretStr(api_key),
                    openai_proxy=proxy,
                )
            elif 'deepseek' in model_config.name.lower():
                model = ChatDeepseekReasoning(
                    model=model_config.name,
                    base_url=model_config.provider.base_url,
                    api_key=SecretStr(api_key),
                    openai_proxy=proxy,
                )
            else:
                model = ChatOpenAIReasoning(
                    model=model_config.name,
                    base_url=model_config.provider.base_url,
                    api_key=SecretStr(api_key),
                    openai_proxy=proxy,
                )
        case _:
            raise TypeError

    return model, model_config


def invalidate_model_cache(role: ModelRole) -> None:
    """清楚模型配置缓存."""
    _model_cache.pop(role, None)


def get_model(role: ModelRole) -> BaseChatModel:
    """获取指定类型的模型配置."""
    if role in _model_cache:
        return _model_cache[role]

    with Session(engine) as session:
        model_id = setting_crud.get_role(session, role)
        model, model_config = _get_model_from_db(session, model_id)
        model.metadata = {
            'setting_key': role.value,
            'model_id': model_id,
            'input_price': model_config.input_price,
            'output_price': model_config.output_price,
            'cache_read_price': model_config.cache_read_price,
        }
    _model_cache[role] = model
    return model


def get_chat_model() -> BaseChatModel:
    """常规对话模型."""
    return get_model(ModelRole.CHAT)


def get_vision_model() -> BaseChatModel:
    """视觉模型."""
    return get_model(ModelRole.VISION)


def get_agent_model() -> BaseChatModel:
    """用于常规智能体的模型."""
    return get_model(ModelRole.AGENT)


def get_coding_model() -> BaseChatModel:
    """编码模型."""
    return get_model(ModelRole.CODING)


def get_simple_model() -> BaseChatModel:
    """简单任务处理模型."""
    return get_model(ModelRole.SIMPLE)


def get_long_context_model() -> BaseChatModel:
    """长上下文处理模型."""
    return get_model(ModelRole.LONG_CONTEXT)


def get_high_performance_model() -> BaseChatModel:
    """高性能模型."""
    return get_model(ModelRole.HIGH_PERFORMANCE)


def qwen3_embedding() -> OpenAIEmbeddings:
    """加载向量模型."""
    return OpenAIEmbeddings(
        model=get_settings().knowledge_base.EMBEDDING_MODEL,
        base_url=get_settings().knowledge_base.BASE_URL,
        api_key=get_settings().knowledge_base.API_KEY,
    )
