import asyncio
import time
from contextlib import suppress
from typing import Any, Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from pydantic import SecretStr
from sqlalchemy.orm import Session

from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.llm_client import ChatDeepseekReasoning, ChatOpenAIReasoning
from bioflow_ai.models.setting import (
    LLMModel,
    LLMProvider,
    ModelRole,
    ProviderType,
    ReasoningEffort,
    is_deepseek_provider,
)
from bioflow_ai.schemas.setting import LLMModelTestResult
from bioflow_ai.services import llm_setting as setting_service
from bioflow_ai.utils.cache import MemoryCache

_model_cache: MemoryCache[ModelRole, BaseChatModel] = MemoryCache()
_model_signatures: dict[ModelRole, tuple[Any, ...]] = {}
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

MODEL_TEST_PROMPT = 'This is a test message. Reply with OK.'
MODEL_TEST_TIMEOUT_SECONDS = 30.0
MAX_TEST_RESPONSE_CHARS = 200
MAX_TEST_ERROR_CHARS = 500


class NoApikeyError(ValueError):
    pass


def _build_chat_model(
    model_config: LLMModel,
    api_key: str,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    provider = model_config.provider
    if provider is None:
        raise setting_service.LLMProviderNotFoundError

    base_url = (provider.base_url or '').strip() or None
    proxy = get_settings().network.get_proxy_url() if provider.use_proxy else None
    reasoning_effort = (
        ReasoningEffort(model_config.reasoning_effort) if model_config.reasoning_effort is not None else None
    )

    match provider.provider_type:
        case ProviderType.ANTHROPIC:
            anthropic_effort = cast(
                'AnthropicEffort | None',
                reasoning_effort.value if reasoning_effort is not None else None,
            )
            anthropic_options: dict[str, Any] = {}
            if timeout is not None:
                anthropic_options['default_request_timeout'] = timeout
            if max_retries is not None:
                anthropic_options['max_retries'] = max_retries
            return ChatAnthropic(
                model_name=model_config.name,
                api_key=SecretStr(api_key),
                base_url='https://api.anthropic.com' if base_url is None else base_url,
                anthropic_proxy=proxy,
                effort=anthropic_effort,
                default_headers=USER_AGENT,
                **anthropic_options,
            )
        case ProviderType.GOOGLE:
            thinking_budget = None
            thinking_level = None
            if reasoning_effort is not None:
                if model_config.name.lower().startswith('gemini-2.5'):
                    thinking_budget = GEMINI_25_REASONING_BUDGETS[reasoning_effort]
                else:
                    thinking_level = cast('GoogleThinkingLevel', reasoning_effort.value)
            google_options: dict[str, Any] = {}
            if timeout is not None:
                google_options['timeout'] = timeout
            if max_retries is not None:
                google_options['max_retries'] = max_retries
            return ChatGoogleGenerativeAI(
                model=model_config.name,
                google_api_key=SecretStr(api_key),
                base_url=base_url,
                additional_headers=USER_AGENT,
                client_args={'proxy': proxy} if proxy else None,
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
                **google_options,
            )
        case ProviderType.OPENAI:
            effective_extra_body = None
            if is_deepseek_provider(provider):
                if reasoning_effort is ReasoningEffort.NONE:
                    openai_reasoning_effort = None
                    effective_extra_body = {'thinking': {'type': 'disabled'}}
                else:
                    openai_reasoning_effort = reasoning_effort.value if reasoning_effort is not None else None
                    if reasoning_effort is not None:
                        effective_extra_body = {'thinking': {'type': 'enabled'}}
            else:
                openai_reasoning_effort = reasoning_effort.value if reasoning_effort is not None else None

            openai_options: dict[str, Any] = {}
            if timeout is not None:
                openai_options['request_timeout'] = timeout
            if max_retries is not None:
                openai_options['max_retries'] = max_retries

            if base_url is None:
                return ChatOpenAI(
                    model=model_config.name,
                    api_key=SecretStr(api_key),
                    openai_proxy=proxy,
                    reasoning_effort=openai_reasoning_effort,
                    extra_body=effective_extra_body,
                    default_headers=USER_AGENT,
                    **openai_options,
                )
            if is_deepseek_provider(provider):
                return ChatDeepseekReasoning(
                    model=model_config.name,
                    base_url=base_url,
                    api_key=SecretStr(api_key),
                    openai_proxy=proxy,
                    reasoning_effort=openai_reasoning_effort,
                    extra_body=effective_extra_body,
                    default_headers=USER_AGENT,
                    **openai_options,
                )
            return ChatOpenAIReasoning(
                model=model_config.name,
                base_url=base_url,
                api_key=SecretStr(api_key),
                openai_proxy=proxy,
                reasoning_effort=openai_reasoning_effort,
                extra_body=effective_extra_body,
                default_headers=USER_AGENT,
                **openai_options,
            )
        case _:
            raise TypeError


def _model_signature(model_config: LLMModel, provider: LLMProvider) -> tuple[Any, ...]:
    """Build a hashable snapshot of all DB and proxy settings affecting a built model."""
    base_url = (provider.base_url or '').strip() or None
    proxy = get_settings().network.get_proxy_url() if provider.use_proxy else None
    return (
        model_config.id,
        model_config.name,
        model_config.reasoning_effort,
        model_config.context_window,
        model_config.input_price,
        model_config.output_price,
        model_config.cache_read_price,
        provider.id,
        provider.name,
        provider.provider_type,
        base_url,
        provider.api_key_ciphertext,
        provider.use_proxy,
        proxy,
    )


def _truncate_text(text: str, max_length: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_length:
        return stripped
    return f'{stripped[: max_length - 1].rstrip()}…'


def _extract_response_text(message: BaseMessage) -> str:
    text = str(message.text).strip()
    if text:
        return text

    if isinstance(message.content, list):
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, dict) and block.get('type') == 'output_text':
                block_text = block.get('text')
                if isinstance(block_text, str):
                    parts.append(block_text)
        return ''.join(parts).strip()

    return ''


def _format_test_error(exc: Exception, *, api_key: str) -> str:
    message = str(exc).strip() or type(exc).__name__
    if api_key:
        message = message.replace(api_key, '***')
    return _truncate_text(message, MAX_TEST_ERROR_CHARS)


async def _close_chat_model(chat_model: BaseChatModel) -> None:
    """Release per-instance HTTP clients created for a one-off model probe.

    ChatOpenAI and ChatAnthropic reuse shared default httpx clients via lru_cache,
    so closing them would poison subsequent requests with the same connection settings.
    """
    if isinstance(chat_model, ChatGoogleGenerativeAI) and chat_model.client is not None:
        with suppress(Exception):
            await chat_model.client.aio.aclose()
        with suppress(Exception):
            chat_model.client.close()


async def probe_model_availability(
    session: Session,
    model_id: int,
    *,
    timeout_seconds: float = MODEL_TEST_TIMEOUT_SECONDS,
) -> LLMModelTestResult:
    """Send a minimal probe message to verify an LLM model is reachable and responding."""
    model_config = setting_service.get_model(session, model_id)
    provider = model_config.provider
    if provider is None:
        raise setting_service.LLMProviderNotFoundError

    model_name = model_config.name
    provider_name = provider.name
    api_key = setting_service.get_provider_api_key(provider)
    if not api_key.strip():
        return LLMModelTestResult(
            model_id=model_id,
            model_name=model_name,
            provider_name=provider_name,
            available=False,
            latency_ms=0,
            error='API key is not configured for this provider.',
        )

    started_at = time.perf_counter()
    chat_model: BaseChatModel | None = None
    try:
        chat_model = _build_chat_model(model_config, api_key, timeout=timeout_seconds, max_retries=0)
        response = await asyncio.wait_for(
            chat_model.ainvoke([HumanMessage(content=MODEL_TEST_PROMPT)]),
            timeout=timeout_seconds,
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        response_text = _extract_response_text(response)
        if not response_text:
            return LLMModelTestResult(
                model_id=model_id,
                model_name=model_name,
                provider_name=provider_name,
                available=False,
                latency_ms=latency_ms,
                error='Model returned an empty response.',
            )
        return LLMModelTestResult(
            model_id=model_id,
            model_name=model_name,
            provider_name=provider_name,
            available=True,
            latency_ms=latency_ms,
            response_text=_truncate_text(response_text, MAX_TEST_RESPONSE_CHARS),
        )
    except TimeoutError:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        return LLMModelTestResult(
            model_id=model_id,
            model_name=model_name,
            provider_name=provider_name,
            available=False,
            latency_ms=latency_ms,
            error=f'Request timed out after {int(timeout_seconds)}s.',
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        error_message = _format_test_error(exc, api_key=api_key)
        logger.warning(
            'LLM model availability test failed for model_id={} ({}/{}): {}',
            model_id,
            provider_name,
            model_name,
            error_message,
        )
        return LLMModelTestResult(
            model_id=model_id,
            model_name=model_name,
            provider_name=provider_name,
            available=False,
            latency_ms=latency_ms,
            error=error_message,
        )
    finally:
        if chat_model is not None:
            await _close_chat_model(chat_model)


def invalidate_model_cache(role: ModelRole | None = None) -> None:
    """清除单个角色或全部模型配置缓存."""
    if role is None:
        _model_cache.clear()
        _model_signatures.clear()
    else:
        _model_cache.delete(role)
        _model_signatures.pop(role, None)


def get_model(role: ModelRole) -> BaseChatModel:
    """获取指定类型的模型配置."""
    with Session(engine) as session:
        model_id = setting_service.get_role_model_id(session, role)
        model_config = setting_service.get_model(session, model_id)
        provider = model_config.provider
        if provider is None:
            raise setting_service.LLMProviderNotFoundError

        signature = _model_signature(model_config, provider)
        cached_model = _model_cache.get(role)
        if cached_model is not None and _model_signatures.get(role) == signature:
            return cached_model

        api_key = setting_service.get_provider_api_key(provider)
        model = _build_chat_model(model_config, api_key)
        model.metadata = {
            'setting_key': role.value,
            'model_id': model_id,
            'model_name': model_config.name,
            'context_window': model_config.context_window,
            'input_price': model_config.input_price,
            'output_price': model_config.output_price,
            'cache_read_price': model_config.cache_read_price,
        }
    _model_cache.set(role, model)
    _model_signatures[role] = signature
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
        request_timeout=get_settings().knowledge_base.TIMEOUT,
    )


def pdf_embedding() -> OpenAIEmbeddings:
    """Send bounded PDF text directly to the configured Qwen embedding endpoint."""
    config = get_settings().knowledge_base
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.BASE_URL,
        api_key=config.API_KEY,
        request_timeout=config.TIMEOUT,
        check_embedding_ctx_length=False,
        chunk_size=get_settings().pdf_parser.BATCH_SIZE,
    )
