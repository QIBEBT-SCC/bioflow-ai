from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from urllib.parse import urlparse

import questionary
from sqlalchemy.orm import Session

import app.crud.setting.llm_provider as provider_crud
import app.crud.setting.llm_setting as setting_crud
from app.core.session import engine
from app.models.setting import LLMModel, LLMProvider, ModelRole, ProviderType
from app.services import llm_setting as setting_service
from app.utils.init import tui
from app.utils.model_pricing import find_model_price

CUSTOM_PROVIDER_KEY = 'custom'
ZERO_PRICE = Decimal(0)


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    """Built-in connection details and model suggestions for an official provider."""

    key: str
    name: str
    provider_type: ProviderType
    base_url: str | None
    models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """One model selected in the initialization wizard."""

    name: str
    display_name: str


@dataclass(frozen=True, slots=True)
class ProviderConfiguration:
    """Provider and model values collected by the initialization wizard."""

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
    ModelRole.LONG_CONTEXT: '长上下文模型',
    ModelRole.AGENT: '智能体模型',
    ModelRole.CODING: '编码模型',
    ModelRole.SIMPLE: '简单任务模型',
    ModelRole.HIGH_PERFORMANCE: '高性能模型',
    ModelRole.CHAT: '常规对话模型',
    ModelRole.VISION: '视觉模型',
}


def validate_provider_selection(selected: list[str]) -> bool | str:
    """Require at least one provider in the initialization wizard."""
    return True if selected else '请至少选择一个 AI Provider。'


def validate_model_selection(selected: list[str]) -> bool | str:
    """Require at least one model for every selected official provider."""
    return True if selected else '请至少选择一个模型。'


def validate_required_text(value: str) -> bool | str:
    """Reject blank names while allowing surrounding whitespace in user input."""
    return True if value.strip() else '此项不能为空。'


def validate_base_url(value: str) -> bool | str:
    """Require an absolute HTTP(S) URL for an OpenAI-compatible custom provider."""
    parsed = urlparse(value.strip())
    if parsed.scheme in {'http', 'https'} and parsed.netloc:
        return True
    return '请输入完整的 http:// 或 https:// 地址。'


def validate_custom_provider_name(value: str) -> bool | str:
    """Prevent a custom endpoint from silently replacing an official provider."""
    name = value.strip()
    if not name:
        return 'Provider 名称不能为空。'
    if name.casefold() in RESERVED_PROVIDER_NAMES:
        return '自定义 Provider 不能使用官方 Provider 名称。'
    return True


def _validate_api_key(value: str, *, credential_configured: bool) -> bool | str:
    return True if value or credential_configured else 'API key 不能为空。'


def _existing_provider(providers: Sequence[LLMProvider], name: str) -> LLMProvider | None:
    normalized_name = name.casefold()
    return next((provider for provider in providers if provider.name.casefold() == normalized_name), None)


def _prompt_api_key(provider_name: str, existing: LLMProvider | None, *, required: bool) -> str | None:
    credential_configured = existing is not None and existing.credential_configured
    key_hint = ' (留空保留已有值)' if credential_configured else (' (可留空)' if not required else '')
    validate = None
    if required:

        def validate(value: str) -> bool | str:
            return _validate_api_key(value, credential_configured=credential_configured)

    api_key = tui.prompt_password(f'{provider_name} API key{key_hint}', validate=validate).strip()
    return api_key or None


def _prompt_display_name(model_name: str, existing: LLMModel | None = None) -> str:
    default = existing.display_name if existing is not None else model_name
    return tui.prompt_text(
        f'{model_name} 显示名称',
        default=default,
        validate=validate_required_text,
    ).strip()


def _prompt_official_provider(
    preset: ProviderPreset,
    existing: LLMProvider | None,
) -> ProviderConfiguration:
    tui.render_section(preset.name, '选择模型并配置官方服务凭据。')
    api_key = _prompt_api_key(preset.name, existing, required=True)
    use_proxy = tui.prompt_confirm(
        f'{preset.name} 是否使用系统代理?',
        default=existing.use_proxy if existing is not None else False,
    )

    existing_models = {model.name: model for model in existing.models} if existing is not None else {}
    active_suggested_models = (
        {model.name for model in existing.models if model.is_active and model.name in preset.models}
        if existing is not None
        else set()
    )
    choices = [
        questionary.Choice(
            model_name,
            checked=(model_name in active_suggested_models if active_suggested_models else index == 0),
        )
        for index, model_name in enumerate(preset.models)
    ]
    selected_model_names = tui.prompt_checkbox(
        f'选择要启用的 {preset.name} 模型 (空格选择, 回车确认)',
        choices=choices,
        validate=validate_model_selection,
    )
    models = tuple(
        ModelConfiguration(
            name=model_name,
            display_name=_prompt_display_name(model_name, existing_models.get(model_name)),
        )
        for model_name in selected_model_names
    )
    return ProviderConfiguration(
        key=preset.key,
        name=preset.name,
        provider_type=preset.provider_type,
        api_key=api_key,
        base_url=preset.base_url,
        use_proxy=use_proxy,
        models=models,
    )


def _prompt_custom_models(existing: LLMProvider | None) -> tuple[ModelConfiguration, ...]:
    existing_models = {model.name: model for model in existing.models} if existing is not None else {}
    configured: list[ModelConfiguration] = []

    while True:
        default_name = ''
        if not configured and existing_models:
            default_name = next(iter(existing_models))
        model_name = tui.prompt_text(
            '模型 API 名称 (model name)',
            default=default_name,
            validate=validate_required_text,
        ).strip()
        if any(model.name == model_name for model in configured):
            tui.render_notice(f'模型 {model_name} 已添加, 请换一个名称。', style='#f59e0b')
            continue

        configured.append(
            ModelConfiguration(
                name=model_name,
                display_name=_prompt_display_name(model_name, existing_models.get(model_name)),
            )
        )
        if not tui.prompt_confirm('是否继续为此 Provider 添加模型?', default=False):
            return tuple(configured)


def _prompt_custom_provider(providers: Sequence[LLMProvider]) -> ProviderConfiguration:
    tui.render_section('Custom', '配置一个 OpenAI-compatible Provider。')
    provider_name = tui.prompt_text(
        'Provider 名称',
        default='Custom',
        validate=validate_custom_provider_name,
    ).strip()
    existing = _existing_provider(providers, provider_name)
    base_url = (
        tui.prompt_text(
            'Provider base URL',
            default=existing.base_url if existing is not None and existing.base_url is not None else '',
            validate=validate_base_url,
        )
        .strip()
        .rstrip('/')
    )
    api_key = _prompt_api_key(provider_name, existing, required=False)
    use_proxy = tui.prompt_confirm(
        f'{provider_name} 是否使用系统代理?',
        default=existing.use_proxy if existing is not None else False,
    )
    return ProviderConfiguration(
        key=CUSTOM_PROVIDER_KEY,
        name=provider_name,
        provider_type=ProviderType.OPENAI,
        api_key=api_key,
        base_url=base_url,
        use_proxy=use_proxy,
        models=_prompt_custom_models(existing),
    )


def prompt_provider_configurations(providers: Sequence[LLMProvider]) -> list[ProviderConfiguration]:
    """Collect one or more official/custom providers and their models."""
    choices = [
        questionary.Choice(
            preset.name,
            value=preset.key,
            checked=bool(
                (existing := _existing_provider(providers, preset.name))
                and existing.is_active
                and existing.credential_configured
            ),
        )
        for preset in OFFICIAL_PROVIDER_PRESETS
    ]
    custom = _existing_provider(providers, 'Custom')
    choices.append(
        questionary.Choice(
            'Custom (OpenAI-compatible)',
            value=CUSTOM_PROVIDER_KEY,
            checked=bool(custom and custom.is_active),
        )
    )
    selected_keys = tui.prompt_checkbox(
        '选择要启用的 AI Provider (空格选择, 回车确认)',
        choices=choices,
        validate=validate_provider_selection,
    )

    configurations: list[ProviderConfiguration] = []
    for key in selected_keys:
        if key == CUSTOM_PROVIDER_KEY:
            configurations.append(_prompt_custom_provider(providers))
            continue
        preset = PRESET_BY_KEY[key]
        configurations.append(_prompt_official_provider(preset, _existing_provider(providers, preset.name)))
    return configurations


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
        provider = _existing_provider(providers, configuration.name)
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


def prompt_role_assignments(models: Sequence[LLMModel]) -> dict[ModelRole, LLMModel]:
    """Show current role assignments and let users edit them one at a time."""
    assignments = dict.fromkeys(ModelRole, models[0])
    model_labels = [f'{_model_provider_name(model)} / {model.display_name} ({model.name})' for model in models]
    model_by_label = dict(zip(model_labels, models, strict=True))
    done_choice = '完成'
    role_choices = [f'{index}. {label}' for index, label in enumerate(ROLE_LABELS.values(), start=1)]
    role_by_choice = dict(zip(role_choices, ROLE_LABELS, strict=True))

    while True:
        tui.render_model_role_assignments(
            [(role_label, model_labels[models.index(assignments[role])]) for role, role_label in ROLE_LABELS.items()]
        )
        selected_role = tui.prompt_select(
            '选择要修改的角色',
            choices=[done_choice, *role_choices],
            default=done_choice,
        )
        if selected_role == done_choice:
            return assignments

        role = role_by_choice[selected_role]
        current_model = assignments[role]
        selected_label = tui.prompt_select(
            f'为{ROLE_LABELS[role]}选择模型',
            choices=model_labels,
            default=model_labels[models.index(current_model)],
        )
        assignments[role] = model_by_label[selected_label]


def apply_role_assignments(
    session: Session,
    assignments: Mapping[ModelRole, LLMModel],
) -> None:
    """Persist the selected model for every system role."""
    for role, model in assignments.items():
        setting_crud.update(session, role, model.id)


def _model_provider_name(model: LLMModel) -> str:
    if model.provider is None:
        return 'Unknown Provider'
    return model.provider.name


def initialize_llm_providers() -> bool:
    """Interactively create providers, models, prices, and system-role assignments."""
    tui.render_section(
        'AI Provider 与模型',
        '配置官方或 OpenAI-compatible 服务、可用模型以及系统智能体模型。',
    )

    try:
        with Session(engine) as session:
            providers = list(provider_crud.get_all(session))
            configurations = prompt_provider_configurations(providers)
            models, models_without_prices = apply_provider_configurations(session, providers, configurations)
            role_assignments = prompt_role_assignments(models)
            apply_role_assignments(session, role_assignments)
            model_names = ', '.join(f'{_model_provider_name(model)}/{model.display_name}' for model in models)
            session.commit()
    except Exception as exc:
        tui.render_notice(f'AI Provider 与模型配置失败: {exc}', style='#ef4444')
        return False

    provider_names = ', '.join(configuration.name for configuration in configurations)
    tui.render_notice(f'已启用 AI Provider: {provider_names}', style='#10b981')
    tui.render_notice(f'已配置模型: {model_names}', style='#10b981')
    if models_without_prices:
        missing_names = ', '.join(models_without_prices)
        tui.render_notice(
            f'以下模型未能自动识别价格, 当前价格暂记为 0: {missing_names}。'
            '初始化完成后请在 Web UI 的模型设置中补充价格。',
            style='#f59e0b',
        )
    return True
