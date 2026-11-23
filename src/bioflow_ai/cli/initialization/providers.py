from collections.abc import Mapping, Sequence

import questionary
from sqlalchemy.orm import Session

from bioflow_ai.cli.initialization import tui
from bioflow_ai.models.setting import LLMModel, LLMProvider, ModelRole, ProviderType
from bioflow_ai.services.initialization.providers import (
    CUSTOM_PROVIDER_KEY,
    OFFICIAL_PROVIDER_PRESETS,
    PRESET_BY_KEY,
    ROLE_LABELS,
    ModelConfiguration,
    ProviderConfiguration,
    ProviderPreset,
    apply_provider_configurations,
    apply_role_assignments,
    existing_provider,
    existing_role_assignments,
    list_providers,
    validate_api_key,
    validate_base_url,
    validate_custom_provider_name,
    validate_model_selection,
    validate_provider_selection,
    validate_required_text,
)


def _prompt_api_key(provider_name: str, existing: LLMProvider | None, *, required: bool) -> str | None:
    credential_configured = existing is not None and existing.credential_configured
    key_hint = (
        ' (leave blank to keep the current value)' if credential_configured else (' (optional)' if not required else '')
    )
    validate = None
    if required:

        def validate(value: str) -> bool | str:
            return validate_api_key(value, credential_configured=credential_configured)

    api_key = tui.prompt_password(f'{provider_name} API key{key_hint}', validate=validate).strip()
    return api_key or None


def _prompt_display_name(model_name: str, existing: LLMModel | None = None) -> str:
    default = existing.display_name if existing is not None else model_name
    return tui.prompt_text(
        f'{model_name} display name',
        default=default,
        validate=validate_required_text,
    ).strip()


def _prompt_official_provider(
    preset: ProviderPreset,
    existing: LLMProvider | None,
) -> ProviderConfiguration:
    tui.render_section(preset.name, 'Select models and configure official service credentials.')
    api_key = _prompt_api_key(preset.name, existing, required=True)
    use_proxy = tui.prompt_confirm(
        f'Use the system proxy for {preset.name}?',
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
        f'Select {preset.name} models to enable (Space to select, Enter to confirm)',
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
            'Model API name (model name)',
            default=default_name,
            validate=validate_required_text,
        ).strip()
        if any(model.name == model_name for model in configured):
            tui.render_notice(f'Model {model_name} has already been added. Enter a different name.', style='#f59e0b')
            continue

        configured.append(
            ModelConfiguration(
                name=model_name,
                display_name=_prompt_display_name(model_name, existing_models.get(model_name)),
            )
        )
        if not tui.prompt_confirm('Add another model to this provider?', default=False):
            return tuple(configured)


def _prompt_custom_provider(providers: Sequence[LLMProvider]) -> ProviderConfiguration:
    tui.render_section('Custom', 'Configure an OpenAI-compatible provider.')
    provider_name = tui.prompt_text(
        'Provider name',
        default='Custom',
        validate=validate_custom_provider_name,
    ).strip()
    existing = existing_provider(providers, provider_name)
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
        f'Use the system proxy for {provider_name}?',
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
                (existing := existing_provider(providers, preset.name))
                and existing.is_active
                and existing.credential_configured
            ),
        )
        for preset in OFFICIAL_PROVIDER_PRESETS
    ]
    custom = existing_provider(providers, 'Custom')
    choices.append(
        questionary.Choice(
            'Custom (OpenAI-compatible)',
            value=CUSTOM_PROVIDER_KEY,
            checked=bool(custom and custom.is_active),
        )
    )
    selected_keys = tui.prompt_checkbox(
        'Select AI providers to enable (Space to select, Enter to confirm)',
        choices=choices,
        validate=validate_provider_selection,
    )

    configurations: list[ProviderConfiguration] = []
    for key in selected_keys:
        if key == CUSTOM_PROVIDER_KEY:
            configurations.append(_prompt_custom_provider(providers))
            continue
        preset = PRESET_BY_KEY[key]
        configurations.append(_prompt_official_provider(preset, existing_provider(providers, preset.name)))
    return configurations


def prompt_role_assignments(
    models: Sequence[LLMModel], existing: Mapping[ModelRole, LLMModel]
) -> dict[ModelRole, LLMModel]:
    """Show current role assignments and let users edit them one at a time."""
    assignments = dict.fromkeys(ModelRole, models[0])
    available = {model.id: model for model in models}
    assignments.update({role: available[model.id] for role, model in existing.items() if model.id in available})
    model_labels = [f'{_model_provider_name(model)} / {model.display_name} ({model.name})' for model in models]
    model_by_label = dict(zip(model_labels, models, strict=True))
    done_choice = 'Done'
    role_choices = [f'{index}. {label}' for index, label in enumerate(ROLE_LABELS.values(), start=1)]
    role_by_choice = dict(zip(role_choices, ROLE_LABELS, strict=True))

    while True:
        tui.render_model_role_assignments(
            [(role_label, model_labels[models.index(assignments[role])]) for role, role_label in ROLE_LABELS.items()]
        )
        selected_role = tui.prompt_select(
            'Select a role to change',
            choices=[done_choice, *role_choices],
            default=done_choice,
        )
        if selected_role == done_choice:
            return assignments

        role = role_by_choice[selected_role]
        current_model = assignments[role]
        selected_label = tui.prompt_select(
            f'Select a model for {ROLE_LABELS[role]}',
            choices=model_labels,
            default=model_labels[models.index(current_model)],
        )
        assignments[role] = model_by_label[selected_label]


def _model_provider_name(model: LLMModel) -> str:
    if model.provider is None:
        return 'Unknown Provider'
    return model.provider.name


def initialize_llm_providers(*, reuse_existing: bool) -> bool:
    """Interactively create providers, models, prices, and system-role assignments."""
    tui.render_subsection('Chat Models')

    try:
        from bioflow_ai.infrastructure.database import engine

        with Session(engine) as session:
            providers = list_providers(session)
            roles = existing_role_assignments(session, providers)
            if reuse_existing and len(roles) == len(ModelRole):
                tui.render_model_role_assignments(
                    [
                        (label, f'{_model_provider_name(roles[role])} / {roles[role].display_name}')
                        for role, label in ROLE_LABELS.items()
                    ]
                )
                if not tui.prompt_confirm('Update the saved chat models and role assignments?', default=False):
                    return True
            configurations = prompt_provider_configurations(providers)
            models, models_without_prices = apply_provider_configurations(session, providers, configurations)
            role_assignments = prompt_role_assignments(models, roles)
            apply_role_assignments(session, role_assignments)
            model_names = ', '.join(f'{_model_provider_name(model)}/{model.display_name}' for model in models)
            session.commit()
    except Exception as exc:
        tui.render_notice(f'AI provider and model configuration failed: {exc}', style='#ef4444')
        return False

    provider_names = ', '.join(configuration.name for configuration in configurations)
    tui.render_notice(f'Enabled AI providers: {provider_names}', style='#10b981')
    tui.render_notice(f'Configured models: {model_names}', style='#10b981')
    if models_without_prices:
        missing_names = ', '.join(models_without_prices)
        tui.render_notice(
            f'Pricing could not be detected for these models and is temporarily set to 0: {missing_names}. '
            'After initialization, add pricing in the Web UI model settings.',
            style='#f59e0b',
        )
    return True
