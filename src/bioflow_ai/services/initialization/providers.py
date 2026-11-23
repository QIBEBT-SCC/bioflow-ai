from collections.abc import Collection, Mapping, Sequence
from decimal import Decimal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

import bioflow_ai.crud.setting.llm_provider as provider_crud
import bioflow_ai.crud.setting.llm_setting as setting_crud
from bioflow_ai.llm.pricing import find_model_defaults
from bioflow_ai.models.setting import LLMModel, LLMProvider, LLMSetting, ModelRole, ProviderType
from bioflow_ai.services import llm_setting as setting_service

CUSTOM_PROVIDER_KEY = 'custom'
ZERO_PRICE = Decimal(0)


class ModelPreset(BaseModel):
    """Built-in model metadata matching the database preset catalog."""

    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    input_price: Decimal | None = None
    output_price: Decimal | None = None
    cache_read_price: Decimal | None = None
    context_window: int | None = None


class ProviderPreset(BaseModel):
    """Built-in connection details and model suggestions for an official provider."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    provider_type: ProviderType
    base_url: str | None
    models: tuple[ModelPreset, ...]


class ModelConfiguration(BaseModel):
    """One model selected or entered in the initialization wizard."""

    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    model_id: int | None = None
    input_price: Decimal | None = None
    output_price: Decimal | None = None
    cache_read_price: Decimal | None = None
    context_window: int | None = None


class ProviderConfiguration(BaseModel):
    """Provider and model values collected by the initialization wizard."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    provider_type: ProviderType
    api_key: str | None = None
    credential_configured: bool = False
    base_url: str | None = None
    use_proxy: bool = False
    models: tuple[ModelConfiguration, ...] = ()


def _preset_model(
    name: str,
    display_name: str,
    *,
    input_price: str | None = None,
    output_price: str | None = None,
    cache_read_price: str | None = None,
    context_window: int | None = None,
) -> ModelPreset:
    defaults = find_model_defaults(name)
    return ModelPreset(
        name=name,
        display_name=display_name,
        input_price=(
            Decimal(input_price)
            if input_price is not None
            else (defaults.price.input if defaults is not None else None)
        ),
        output_price=(
            Decimal(output_price)
            if output_price is not None
            else (defaults.price.output if defaults is not None else None)
        ),
        cache_read_price=(
            Decimal(cache_read_price)
            if cache_read_price is not None
            else (defaults.price.cache_read if defaults is not None else None)
        ),
        context_window=(
            context_window
            if context_window is not None
            else (defaults.context_window if defaults is not None else None)
        ),
    )


OFFICIAL_PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        key='openai',
        name='OpenAI',
        provider_type=ProviderType.OPENAI,
        base_url=None,
        models=(
            _preset_model('gpt-6.2-astra', 'GPT-6.2 Astra'),
            _preset_model('gpt-6.2-sol', 'GPT-6.2 Sol'),
            _preset_model('gpt-6.1-terra', 'GPT-6.1 Terra'),
            _preset_model('gpt-6-luna', 'GPT-6 Luna'),
        ),
    ),
    ProviderPreset(
        key='anthropic',
        name='Anthropic',
        provider_type=ProviderType.ANTHROPIC,
        base_url=None,
        models=(
            _preset_model('claude-fable-5-2', 'Claude Fable 5.2'),
            _preset_model('claude-opus-5-5', 'Claude Opus 5.5'),
            _preset_model('claude-sonnet-5-5', 'Claude Sonnet 5.5'),
            _preset_model('claude-haiku-5-5', 'Claude Haiku 5.5'),
        ),
    ),
    ProviderPreset(
        key='google',
        name='Google',
        provider_type=ProviderType.GOOGLE,
        base_url=None,
        models=(
            _preset_model('gemini-4-carbon', 'Gemini 4 Carbon'),
            _preset_model('gemini-4-neon', 'Gemini 4 Neon'),
            _preset_model('gemini-4-hydrogen', 'Gemini 4 Hydrogen'),
            _preset_model('gemini-3.8-flash', 'Gemini 3.8 Flash'),
            _preset_model('gemini-3.5-flash-lite', 'Gemini 3.5 Flash-Lite'),
        ),
    ),
    ProviderPreset(
        key='deepseek',
        name='DeepSeek',
        provider_type=ProviderType.OPENAI,
        base_url='https://api.deepseek.com/v1',
        models=(
            _preset_model('deepseek-v4-pro', 'DeepSeek V4 Pro'),
            _preset_model('deepseek-flash', 'DeepSeek V4.1 Flash'),
        ),
    ),
    ProviderPreset(
        key='openrouter',
        name='OpenRouter',
        provider_type=ProviderType.OPENAI,
        base_url='https://openrouter.ai/api/v1',
        models=(
            _preset_model('openai/gpt-6.2-astra', 'GPT-6.2 Astra'),
            _preset_model('openai/gpt-6.2-sol', 'GPT-6.2 Sol'),
            _preset_model('openai/gpt-6.1-terra', 'GPT-6.1 Terra'),
            _preset_model('openai/gpt-6-luna', 'GPT-6 Luna'),
            _preset_model('anthropic/claude-fable-5.2', 'Claude Fable 5.2'),
            _preset_model('anthropic/claude-opus-5.5', 'Claude Opus 5.5'),
            _preset_model('anthropic/claude-sonnet-5.5', 'Claude Sonnet 5.5'),
            _preset_model('anthropic/claude-haiku-5.5', 'Claude Haiku 5.5'),
            _preset_model('google/gemini-4-carbon', 'Gemini 4 Carbon'),
            _preset_model('google/gemini-4-neon', 'Gemini 4 Neon'),
            _preset_model('google/gemini-4-hydrogen', 'Gemini 4 Hydrogen'),
            _preset_model(
                'google/gemini-3.8-flash',
                'Gemini 3.8 Flash',
                input_price='0.825',
                output_price='4.125',
                cache_read_price='0.0825',
            ),
            _preset_model('google/gemini-3.5-flash-lite', 'Gemini 3.5 Flash-Lite'),
        ),
    ),
)
PRESET_BY_KEY = {preset.key: preset for preset in OFFICIAL_PROVIDER_PRESETS}
PRESET_BY_NAME = {preset.name.casefold(): preset for preset in OFFICIAL_PROVIDER_PRESETS}
RESERVED_PROVIDER_NAMES = frozenset(PRESET_BY_NAME)

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
    """Require at least one model for every selected provider."""
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


def validate_custom_provider_name(value: str, *, existing_names: Collection[str] = ()) -> bool | str:
    """Prevent a custom endpoint from replacing a preset or duplicating another provider."""
    name = value.strip()
    if not name:
        return 'Provider name is required.'
    normalized = name.casefold()
    if normalized in RESERVED_PROVIDER_NAMES:
        return 'A custom provider cannot use a built-in provider name.'
    if normalized in {item.casefold() for item in existing_names}:
        return f'Provider {name} already exists; pick it from the menu to edit.'
    return True


def validate_api_key(value: str, *, credential_configured: bool) -> bool | str:
    """Require a key unless the existing provider already has credentials."""
    return True if value.strip() or credential_configured else 'API key is required.'


def existing_provider(providers: Sequence[LLMProvider], name: str) -> LLMProvider | None:
    """Find a provider by its case-insensitive configured name."""
    normalized_name = name.casefold()
    return next((provider for provider in providers if provider.name.casefold() == normalized_name), None)


def _match_db_model(
    models: Sequence[LLMModel],
    *,
    model_id: int | None = None,
    name: str | None = None,
) -> LLMModel | None:
    if model_id is not None:
        matched_by_id = next((model for model in models if model.id == model_id), None)
        if matched_by_id is not None:
            return matched_by_id
    if name is None:
        return None
    matches = [model for model in models if model.name == name]
    if not matches:
        return None
    return min(matches, key=lambda model: (model.reasoning_effort is not None, not model.is_active, model.id))


def _deduplicate_db_models(models: Sequence[LLMModel]) -> list[LLMModel]:
    chosen: dict[str, LLMModel] = {}
    for model in models:
        existing = chosen.get(model.name)
        if existing is None or (existing.reasoning_effort is not None and model.reasoning_effort is None):
            chosen[model.name] = model
    return list(chosen.values())


def _to_model_configuration(model: LLMModel) -> ModelConfiguration:
    return ModelConfiguration(
        name=model.name,
        display_name=model.display_name,
        model_id=model.id,
        input_price=model.input_price,
        output_price=model.output_price,
        cache_read_price=model.cache_read_price,
        context_window=model.context_window,
    )


def initial_provider_drafts(providers: Sequence[LLMProvider]) -> dict[str, ProviderConfiguration]:
    """Build editable draft configurations from currently enabled database providers."""
    drafts: dict[str, ProviderConfiguration] = {}
    for provider in providers:
        normalized_name = provider.name.casefold()
        preset = PRESET_BY_NAME.get(normalized_name)
        active_models = _deduplicate_db_models([model for model in provider.models if model.is_active])
        if not provider.is_active or not active_models:
            continue
        if preset is not None and not provider.credential_configured:
            continue

        if preset is not None:
            preset_models_by_name = {item.name: item for item in preset.models}
            model_configs = tuple(
                ModelConfiguration(
                    name=model.name,
                    display_name=(
                        preset_models_by_name[model.name].display_name
                        if model.name in preset_models_by_name
                        else model.display_name
                    ),
                    model_id=model.id,
                    input_price=(
                        preset_models_by_name[model.name].input_price
                        if model.name in preset_models_by_name
                        else model.input_price
                    ),
                    output_price=(
                        preset_models_by_name[model.name].output_price
                        if model.name in preset_models_by_name
                        else model.output_price
                    ),
                    cache_read_price=(
                        preset_models_by_name[model.name].cache_read_price
                        if model.name in preset_models_by_name
                        else model.cache_read_price
                    ),
                    context_window=(
                        preset_models_by_name[model.name].context_window
                        if model.name in preset_models_by_name
                        else model.context_window
                    ),
                )
                for model in active_models
            )
            drafts[normalized_name] = ProviderConfiguration(
                key=preset.key,
                name=preset.name,
                provider_type=preset.provider_type,
                credential_configured=provider.credential_configured,
                base_url=preset.base_url,
                use_proxy=provider.use_proxy,
                models=model_configs,
            )
            continue

        drafts[normalized_name] = ProviderConfiguration(
            key=CUSTOM_PROVIDER_KEY,
            name=provider.name,
            provider_type=provider.provider_type,
            credential_configured=provider.credential_configured,
            base_url=provider.base_url,
            use_proxy=provider.use_proxy,
            models=tuple(_to_model_configuration(model) for model in active_models),
        )
    return drafts


def preset_model_candidates(
    preset: ProviderPreset,
    existing: LLMProvider | None,
    draft: ProviderConfiguration | None,
) -> list[ModelConfiguration]:
    """Return ordered preset models followed by any extra models saved on the provider."""
    candidates: list[ModelConfiguration] = []
    seen_names: set[str] = set()
    existing_models = existing.models if existing is not None else ()

    for preset_model in preset.models:
        matched = _match_db_model(existing_models, name=preset_model.name)
        candidates.append(
            ModelConfiguration(
                name=preset_model.name,
                display_name=preset_model.display_name,
                model_id=matched.id if matched is not None else None,
                input_price=preset_model.input_price,
                output_price=preset_model.output_price,
                cache_read_price=preset_model.cache_read_price,
                context_window=preset_model.context_window,
            )
        )
        seen_names.add(preset_model.name)

    if draft is not None:
        for model in draft.models:
            if model.name not in seen_names:
                candidates.append(model)
                seen_names.add(model.name)

    if existing is not None:
        for model in _deduplicate_db_models(existing_models):
            if model.name not in seen_names:
                candidates.append(_to_model_configuration(model))
                seen_names.add(model.name)

    return candidates


def custom_model_candidates(
    existing: LLMProvider | None,
    draft: ProviderConfiguration | None,
) -> list[ModelConfiguration]:
    """Return existing and draft models for a custom provider in display order."""
    candidates: list[ModelConfiguration] = []
    seen_names: set[str] = set()

    if draft is not None:
        for model in draft.models:
            if model.name not in seen_names:
                candidates.append(model)
                seen_names.add(model.name)

    if existing is not None:
        for model in _deduplicate_db_models(existing.models):
            if model.name not in seen_names:
                candidates.append(_to_model_configuration(model))
                seen_names.add(model.name)

    return candidates


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

        existing_models = list(provider.models)
        for existing_model in existing_models:
            existing_model.is_active = False

        for model_configuration in configuration.models:
            model = _match_db_model(
                existing_models,
                model_id=model_configuration.model_id,
                name=model_configuration.name,
            )
            has_preset_prices = (
                model_configuration.input_price is not None
                and model_configuration.output_price is not None
                and model_configuration.cache_read_price is not None
            )
            defaults = None if has_preset_prices else find_model_defaults(model_configuration.name)

            if has_preset_prices:
                resolved_prices = (
                    model_configuration.input_price,
                    model_configuration.output_price,
                    model_configuration.cache_read_price,
                )
            elif defaults is not None:
                resolved_prices = (
                    defaults.price.input,
                    defaults.price.output,
                    defaults.price.cache_read,
                )
            else:
                resolved_prices = None

            resolved_context = model_configuration.context_window or (
                defaults.context_window if defaults is not None else None
            )

            if model is None:
                model = LLMModel(
                    provider=provider,
                    name=model_configuration.name,
                    display_name=model_configuration.display_name,
                    input_price=resolved_prices[0] if resolved_prices is not None else ZERO_PRICE,
                    output_price=resolved_prices[1] if resolved_prices is not None else ZERO_PRICE,
                    cache_read_price=resolved_prices[2] if resolved_prices is not None else ZERO_PRICE,
                    context_window=resolved_context,
                    reasoning_effort=None,
                    is_active=True,
                )
                session.add(model)
                existing_models.append(model)
                if resolved_prices is None:
                    models_without_prices.append(f'{provider.name}/{model.name}')
            else:
                model.display_name = model_configuration.display_name
                model.is_active = True
                if resolved_prices is not None:
                    input_price, output_price, cache_read_price = resolved_prices
                    if input_price is not None:
                        model.input_price = input_price
                    if output_price is not None:
                        model.output_price = output_price
                    if cache_read_price is not None:
                        model.cache_read_price = cache_read_price
                elif (
                    model.input_price <= ZERO_PRICE
                    and model.output_price <= ZERO_PRICE
                    and model.cache_read_price <= ZERO_PRICE
                ):
                    models_without_prices.append(f'{provider.name}/{model.name}')
                if model.context_window is None and resolved_context is not None:
                    model.context_window = resolved_context

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


def role_assignments_for_models(session: Session, models: Sequence[LLMModel]) -> dict[ModelRole, LLMModel]:
    """Return saved or preset role assignments whose models belong to the enabled set."""
    setting = session.get(LLMSetting, 1)
    if setting is None:
        return {}
    candidates = {model.id: model for model in models}
    return {
        role: candidates[model_id]
        for role in ModelRole
        if (model_id := getattr(setting, f'{role.value}_id')) in candidates
    }


def existing_role_assignments(session: Session, providers: Sequence[LLMProvider]) -> dict[ModelRole, LLMModel]:
    """Return existing roles backed by enabled models and providers."""
    setting = session.get(LLMSetting, 1)
    if setting is None:
        return {}
    candidates = {
        model.id: model
        for provider in providers
        if provider.is_active
        and (provider.credential_configured or provider.name.casefold() not in RESERVED_PROVIDER_NAMES)
        for model in provider.models
        if model.is_active
    }
    return {
        role: candidates[model_id]
        for role in ModelRole
        if (model_id := getattr(setting, f'{role.value}_id')) in candidates
    }
