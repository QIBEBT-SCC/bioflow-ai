from collections.abc import Mapping, Sequence
from decimal import Decimal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

import bioflow_ai.crud.setting.llm_provider as provider_crud
import bioflow_ai.crud.setting.llm_setting as setting_crud
from bioflow_ai.llm.pricing import find_model_price
from bioflow_ai.models.setting import LLMModel, LLMProvider, ModelRole, ProviderType
from bioflow_ai.services import llm_setting as setting_service

CUSTOM_PROVIDER_KEY = 'custom'
ZERO_PRICE = Decimal(0)


class ProviderPreset(BaseModel):
    """Built-in connection details and model suggestions for an official provider."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    provider_type: ProviderType
    base_url: str | None
    models: tuple[str, ...]


class ModelConfiguration(BaseModel):
    """One model selected in the initialization wizard."""

    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str


class ProviderConfiguration(BaseModel):
    """Provider and model values collected by the initialization wizard."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    provider_type: ProviderType
    api_key: str | None
    base_url: str | None
    use_proxy: bool
    models: tuple[ModelConfiguration, ...]


OFFICIAL_PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        key='deepseek',
        name='DeepSeek',
        provider_type=ProviderType.OPENAI,
        base_url='https://api.deepseek.com',
        models=('deepseek-v4-flash', 'deepseek-v4-pro'),
    ),
    ProviderPreset(
        key='openai',
        name='OpenAI',
        provider_type=ProviderType.OPENAI,
        base_url=None,
        models=('gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'),
    ),
)
PRESET_BY_KEY = {preset.key: preset for preset in OFFICIAL_PROVIDER_PRESETS}
RESERVED_PROVIDER_NAMES = frozenset(preset.name.casefold() for preset in OFFICIAL_PROVIDER_PRESETS)

ROLE_LABELS: Mapping[ModelRole, str] = {
    ModelRole.LONG_CONTEXT: 'Long-context model',
    ModelRole.AGENT: 'Agent model',
    ModelRole.CODING: 'Coding model',
    ModelRole.SIMPLE: 'Simple-task model',
    ModelRole.HIGH_PERFORMANCE: 'High-performance model',
    ModelRole.CHAT: 'Chat model',
    ModelRole.VISION: 'Vision model',
}


def validate_provider_selection(selected: list[str]) -> bool | str:
    """Require at least one provider in the initialization wizard."""
    return True if selected else 'Select at least one AI provider.'


def validate_model_selection(selected: list[str]) -> bool | str:
    """Require at least one model for every selected official provider."""
    return True if selected else 'Select at least one model.'


def validate_required_text(value: str) -> bool | str:
    """Reject blank names while allowing surrounding whitespace in user input."""
    return True if value.strip() else 'This field is required.'


def validate_base_url(value: str) -> bool | str:
    """Require an absolute HTTP(S) URL for an OpenAI-compatible custom provider."""
    parsed = urlparse(value.strip())
    if parsed.scheme in {'http', 'https'} and parsed.netloc:
        return True
    return 'Enter a complete http:// or https:// URL.'


def validate_custom_provider_name(value: str) -> bool | str:
    """Prevent a custom endpoint from silently replacing an official provider."""
    name = value.strip()
    if not name:
        return 'Provider name is required.'
    if name.casefold() in RESERVED_PROVIDER_NAMES:
        return 'A custom provider cannot use an official provider name.'
    return True


def validate_api_key(value: str, *, credential_configured: bool) -> bool | str:
    """Require a key unless the existing provider already has credentials."""
    return True if value or credential_configured else 'API key is required.'


def existing_provider(providers: Sequence[LLMProvider], name: str) -> LLMProvider | None:
    """Find a provider by its case-insensitive configured name."""
    normalized_name = name.casefold()
    return next((provider for provider in providers if provider.name.casefold() == normalized_name), None)


def apply_provider_configurations(
    session: Session,
    providers: Sequence[LLMProvider],
    configurations: Sequence[ProviderConfiguration],
) -> tuple[list[LLMModel], list[str]]:
    """Upsert selected providers/models and return models in wizard order."""
    selected_providers: set[int] = set()
    configured_models: list[LLMModel] = []
    models_without_prices: list[str] = []

    for configuration in configurations:
        provider = existing_provider(providers, configuration.name)
        if provider is None:
            provider = LLMProvider(name=configuration.name, provider_type=configuration.provider_type)
            session.add(provider)
            session.flush()
        provider.name = configuration.name
        provider.provider_type = configuration.provider_type
        provider.base_url = configuration.base_url
        provider.use_proxy = configuration.use_proxy
        provider.is_active = True
        if configuration.api_key is not None:
            setting_service.set_provider_api_key(provider, configuration.api_key)
        selected_providers.add(provider.id)

        existing_models = {model.name: model for model in provider.models}
        selected_model_names = {model.name for model in configuration.models}
        for model in provider.models:
            model.is_active = model.name in selected_model_names

        for model_configuration in configuration.models:
            model = existing_models.get(model_configuration.name)
            price = find_model_price(model_configuration.name)
            if model is None:
                model = LLMModel(
                    provider=provider,
                    name=model_configuration.name,
                    display_name=model_configuration.display_name,
                    input_price=price.input if price is not None else ZERO_PRICE,
                    output_price=price.output if price is not None else ZERO_PRICE,
                    cache_read_price=price.cache_read if price is not None else ZERO_PRICE,
                    is_active=True,
                )
                session.add(model)
            else:
                model.display_name = model_configuration.display_name
                model.is_active = True
                if price is not None:
                    model.input_price = price.input
                    model.output_price = price.output
                    model.cache_read_price = price.cache_read
            if price is None:
                models_without_prices.append(f'{provider.name}/{model.name}')
            configured_models.append(model)

    for provider in providers:
        if provider.id not in selected_providers:
            provider.is_active = False

    session.flush()
    return configured_models, models_without_prices


def apply_role_assignments(
    session: Session,
    assignments: Mapping[ModelRole, LLMModel],
) -> None:
    """Persist the selected model for every system role."""
    for role, model in assignments.items():
        setting_crud.update(session, role, model.id)


def list_providers(session: Session) -> list[LLMProvider]:
    """Read existing provider choices inside the initialization transaction."""
    return list(provider_crud.get_all(session))
