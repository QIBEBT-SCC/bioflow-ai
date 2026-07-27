from typing import Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.session import engine
from app.models.setting import LLMModel, ModelRole, ProviderType, ReasoningEffort, is_deepseek_provider
from app.services import llm_setting as setting_service
from llm.llm_client import ChatDeepseekReasoning, ChatOpenAIReasoning

_model_cache: dict[ModelRole, BaseChatModel] = {}
AnthropicEffort = Literal['low', 'medium', 'high', 'xhigh', 'max']
GoogleThinkingLevel = Literal['minimal', 'low', 'medium', 'high']
GEMINI_25_REASONING_BUDGETS = {
    ReasoningEffort.MINIMAL: 1024,
    ReasoningEffort.LOW: 1024,
    ReasoningEffort.MEDIUM: 8192,
    ReasoningEffort.HIGH: 24576,
}

USER_AGENT = {
    'User-Agent': 'BioFlowAI-Agent/1.0',
    'X-Agent-Name': 'BioFlowAI',
    'X-Agent-Version': '1.0',
}


class NoApikeyError(ValueError):
    pass


def _get_model_from_db(session: Session, model_id: int) -> tuple[BaseChatModel, LLMModel]:
    model_config = setting_service.get_model(session, model_id)
    if model_config.provider is None:
        raise setting_service.LLMProviderNotFoundError

    api_key = model_config.provider.api_key or ''
    proxy = get_settings().network.get_proxy_url() if model_config.provider.use_proxy else None
    reasoning_effort = (
        ReasoningEffort(model_config.reasoning_effort) if model_config.reasoning_effort is not None else None
    )

    match model_config.provider.provider_type:
        case ProviderType.ANTHROPIC:
            anthropic_effort = cast(
                'AnthropicEffort | None',
                reasoning_effort.value if reasoning_effort is not None else None,
            )
            model = ChatAnthropic(
                model_name=model_config.name,
                api_key=SecretStr(api_key),
                base_url='https://api.anthropic.com'
                if model_config.provider.base_url is None
                else model_config.provider.base_url,
                anthropic_proxy=proxy,
                effort=anthropic_effort,
                default_headers=USER_AGENT,
            )
        case ProviderType.GOOGLE:
            from google.genai.client import Client

            thinking_budget = None
            thinking_level = None
            if reasoning_effort is not None:
                if model_config.name.lower().startswith('gemini-2.5'):
                    thinking_budget = GEMINI_25_REASONING_BUDGETS[reasoning_effort]
                else:
                    thinking_level = cast('GoogleThinkingLevel', reasoning_effort.value)
            if model_config.provider.base_url is None:
                model = ChatGoogleGenerativeAI(
                    model=model_config.name,
                    client=Client(
                        api_key=api_key,
                        http_options={
                            'client_args': {'proxy': proxy} if proxy else None,
                            'headers': USER_AGENT,
                        },
                    ),
                    thinking_budget=thinking_budget,
                    thinking_level=thinking_level,
                )
            else:
                model = ChatGoogleGenerativeAI(
                    model=model_config.name,
                    client=Client(
                        api_key=api_key,
                        http_options={
                            'base_url': model_config.provider.base_url,
                            'client_args': {'proxy': proxy} if proxy else None,
                            'headers': USER_AGENT,
                        },
                    ),
                    thinking_budget=thinking_budget,
                    thinking_level=thinking_level,
                )
        case ProviderType.OPENAI:
            effective_extra_body = None
            if is_deepseek_provider(model_config.provider):
                if reasoning_effort is ReasoningEffort.NONE:
                    openai_reasoning_effort = None
                    effective_extra_body = {'thinking': {'type': 'disabled'}}
                else:
                    openai_reasoning_effort = reasoning_effort.value if reasoning_effort is not None else None
                    if reasoning_effort is not None:
                        effective_extra_body = {'thinking': {'type': 'enabled'}}
            else:
                openai_reasoning_effort = reasoning_effort.value if reasoning_effort is not None else None

            if model_config.provider.base_url is None:
                model = ChatOpenAI(
                    model=model_config.name,
                    api_key=SecretStr(api_key),
                    openai_proxy=proxy,
                    reasoning_effort=openai_reasoning_effort,
                    extra_body=effective_extra_body,
                    default_headers=USER_AGENT,
                )
            elif is_deepseek_provider(model_config.provider):
                model = ChatDeepseekReasoning(
                    model=model_config.name,
                    base_url=model_config.provider.base_url,
                    api_key=SecretStr(api_key),
                    openai_proxy=proxy,
                    reasoning_effort=openai_reasoning_effort,
                    extra_body=effective_extra_body,
                    default_headers=USER_AGENT,
                )
            else:
                model = ChatOpenAIReasoning(
                    model=model_config.name,
                    base_url=model_config.provider.base_url,
                    api_key=SecretStr(api_key),
                    openai_proxy=proxy,
                    reasoning_effort=openai_reasoning_effort,
                    extra_body=effective_extra_body,
                    default_headers=USER_AGENT,
                )
        case _:
            raise TypeError

    return model, model_config


def invalidate_model_cache(role: ModelRole | None = None) -> None:
    """清除单个角色或全部模型配置缓存."""
    if role is None:
        _model_cache.clear()
    else:
        _model_cache.pop(role, None)


def get_model(role: ModelRole) -> BaseChatModel:
    """获取指定类型的模型配置."""
    if role in _model_cache:
        return _model_cache[role]

    with Session(engine) as session:
        model_id = setting_service.get_role_model_id(session, role)
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
