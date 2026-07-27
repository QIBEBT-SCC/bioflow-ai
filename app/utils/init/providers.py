from dataclasses import dataclass

import questionary
from sqlalchemy.orm import Session

import app.crud.setting.llm_provider as provider_crud
from app.core.session import engine
from app.models.setting import LLMProvider
from app.services import llm_setting as setting_service
from app.utils.init import tui


@dataclass(frozen=True, slots=True)
class ProviderConfiguration:
    """Values collected for a selected LLM provider."""

    name: str
    api_key: str
    base_url: str | None
    use_proxy: bool


def validate_provider_selection(selected: list[str]) -> bool | str:
    """Require at least one provider in the initialization wizard."""
    return True if selected else '请至少选择一个模型 Provider。'


def _validate_api_key(value: str, *, existing_api_key: str | None) -> bool | str:
    return True if value or existing_api_key else 'API key 不能为空。'


def prompt_provider_configurations(providers: list[LLMProvider]) -> list[ProviderConfiguration]:
    """Collect provider credentials and connection options interactively."""
    choices = [
        questionary.Choice(provider.name, checked=provider.is_active and bool(provider.api_key))
        for provider in providers
    ]
    selected_names = tui.prompt_checkbox(
        '选择要启用的模型 Provider (空格选择, 回车确认)',
        choices=choices,
        validate=validate_provider_selection,
    )
    provider_by_name = {provider.name: provider for provider in providers}
    configurations: list[ProviderConfiguration] = []

    for name in selected_names:
        provider = provider_by_name[name]
        tui.render_section(provider.name, '配置模型服务的连接信息。')
        key_hint = ' (留空保留已有值)' if provider.api_key else ''
        api_key = tui.prompt_password(
            f'{provider.name} API key{key_hint}',
            validate=lambda value, existing=provider.api_key: _validate_api_key(value, existing_api_key=existing),
        )
        base_url = tui.prompt_text(
            f'{provider.name} base URL (留空使用官方地址)',
            default=provider.base_url or '',
        ).strip()
        use_proxy = tui.prompt_confirm(f'{provider.name} 是否使用系统代理?', default=provider.use_proxy)
        configurations.append(
            ProviderConfiguration(
                name=name,
                api_key=api_key or provider.api_key or '',
                base_url=base_url or None,
                use_proxy=use_proxy,
            )
        )

    return configurations


def apply_provider_configurations(providers: list[LLMProvider], configurations: list[ProviderConfiguration]) -> None:
    """Apply selected configurations and disable providers not selected by the user."""
    configuration_by_name = {configuration.name: configuration for configuration in configurations}
    for provider in providers:
        configuration = configuration_by_name.get(provider.name)
        provider.is_active = configuration is not None
        if configuration is not None:
            provider.api_key = configuration.api_key
            provider.base_url = configuration.base_url
            provider.use_proxy = configuration.use_proxy


def initialize_llm_providers() -> bool:
    """Prompt for and persist at least one usable LLM provider."""
    tui.render_section('模型 Provider', '至少启用一个模型服务, 并配置 API key、base URL 和代理。')

    try:
        with Session(engine) as session:
            providers = list(provider_crud.get_all(session))
            if not providers:
                tui.render_notice('未找到预置模型 Provider, 请先初始化数据库。', style='#ef4444')
                return False

            configurations = prompt_provider_configurations(providers)
            apply_provider_configurations(providers, configurations)
            setting_service.save_providers(session, providers)
    except Exception as exc:
        tui.render_notice(f'模型 Provider 配置失败: {exc}', style='#ef4444')
        return False

    names = ', '.join(configuration.name for configuration in configurations)
    tui.render_notice(f'已启用模型 Provider: {names}', style='#10b981')
    return True
