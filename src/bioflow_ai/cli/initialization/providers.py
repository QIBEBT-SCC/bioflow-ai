from collections.abc import Mapping, Sequence

import questionary
from sqlalchemy.orm import Session

from bioflow_ai.cli.initialization import tui
from bioflow_ai.models.setting import LLMModel, LLMProvider, ModelRole, ProviderType
from bioflow_ai.services.initialization.providers import (
    CUSTOM_PROVIDER_KEY,
    OFFICIAL_PROVIDER_PRESETS,
    PRESET_BY_KEY,
    PRESET_BY_NAME,
    ROLE_LABELS,
    ModelConfiguration,
    ProviderConfiguration,
    ProviderPreset,
    apply_provider_configurations,
    apply_role_assignments,
    custom_model_candidates,
    existing_provider,
    existing_role_assignments,
    initial_provider_drafts,
    list_providers,
    preset_model_candidates,
    role_assignments_for_models,
    validate_api_key,
    validate_base_url,
    validate_custom_provider_name,
    validate_required_text,
)


def _prompt_api_key(provider_name: str, *, credential_configured: bool, required: bool) -> str | None:
    key_hint = (
        ' (leave blank to keep the current value)' if credential_configured else (' (optional)' if not required else '')
    )
    validate = None
    if required:

        def validate(value: str) -> bool | str:
            return validate_api_key(value, credential_configured=credential_configured)

    api_key = tui.prompt_password(f'{provider_name} API key{key_hint}', validate=validate).strip()
    return api_key or None


def _format_model_label(name: str, display_name: str) -> str:
    if display_name != name:
        return f'{display_name} ({name})'
    return name


def _format_model_choice(model: ModelConfiguration) -> str:
    return _format_model_label(model.name, model.display_name)


def _prompt_additional_models(
    provider_name: str,
    *,
    existing_models: Sequence[ModelConfiguration],
    require_at_least_one: bool,
) -> tuple[ModelConfiguration, ...]:
    known_names = {model.name for model in existing_models}
    added: list[ModelConfiguration] = []

    while True:
        model_name = tui.prompt_text(
            'Model name (e.g. gpt-6.2-sol, deepseek-flash)',
            validate=validate_required_text,
        ).strip()
        if model_name in known_names:
            tui.render_notice(f'Model {model_name} is already added to {provider_name}.', style='#f59e0b')
            if (
                not added
                and not require_at_least_one
                and not tui.prompt_confirm('Add a different model?', default=False)
            ):
                return ()
            continue

        known_names.add(model_name)
        added.append(
            ModelConfiguration(
                name=model_name,
                display_name=model_name,
            )
        )
        if not tui.prompt_confirm(f'Add another model to {provider_name}?', default=False):
            return tuple(added)


def _prompt_official_provider(
    preset: ProviderPreset,
    existing: LLMProvider | None,
    draft: ProviderConfiguration | None,
) -> ProviderConfiguration:
    tui.render_section(preset.name, 'Configure API credentials and choose preset or custom models.')
    credential_configured = bool(
        (existing is not None and existing.credential_configured) or (draft is not None and draft.credential_configured)
    )
    api_key_input = _prompt_api_key(preset.name, credential_configured=credential_configured, required=True)
    api_key = api_key_input if api_key_input is not None else (draft.api_key if draft is not None else None)

    default_proxy = draft.use_proxy if draft is not None else (existing.use_proxy if existing is not None else False)
    use_proxy = tui.prompt_confirm(
        f'Use the system proxy for {preset.name}?',
        default=default_proxy,
    )

    tui.render_step_divider(
        f'Select Models ({preset.name})',
        f'Provider settings complete. Next, select or add models for {preset.name}.',
    )

    candidates = preset_model_candidates(preset, existing, draft)
    if draft is not None:
        checked_names = {model.name for model in draft.models}
    elif existing is not None and existing.is_active and existing.credential_configured:
        checked_names = {model.name for model in existing.models if model.is_active}
        if not checked_names:
            checked_names = {item.name for item in preset.models}
    else:
        checked_names = {item.name for item in preset.models}

    choices = [
        questionary.Choice(
            _format_model_choice(candidate),
            value=candidate.name,
            checked=candidate.name in checked_names,
        )
        for candidate in candidates
    ]
    selected_names = tui.prompt_checkbox(
        f'Select {preset.name} models to enable (Space to toggle, Enter to confirm)',
        choices=choices,
    )
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    selected_models = [candidate_by_name[name] for name in selected_names]

    if not selected_models:
        tui.render_notice(
            f'No preset models selected for {preset.name}; add at least one custom model.',
            style='#94a3b8',
        )
        selected_models.extend(_prompt_additional_models(preset.name, existing_models=(), require_at_least_one=True))
    elif tui.prompt_confirm(f'Add custom model(s) to {preset.name}?', default=False):
        selected_models.extend(
            _prompt_additional_models(
                preset.name,
                existing_models=selected_models,
                require_at_least_one=False,
            )
        )

    return ProviderConfiguration(
        key=preset.key,
        name=preset.name,
        provider_type=preset.provider_type,
        api_key=api_key,
        credential_configured=credential_configured or api_key is not None,
        base_url=preset.base_url,
        use_proxy=use_proxy,
        models=tuple(selected_models),
    )


def _prompt_new_custom_provider(
    providers: Sequence[LLMProvider],
    drafts: Mapping[str, ProviderConfiguration],
) -> ProviderConfiguration:
    tui.render_section('Custom Provider', 'Configure an OpenAI-compatible provider.')
    existing_names = [provider.name for provider in providers] + [draft.name for draft in drafts.values()]
    default_name = '' if any(name.casefold() == 'custom' for name in existing_names) else 'Custom'
    provider_name = tui.prompt_text(
        'Provider name',
        default=default_name,
        validate=lambda value: validate_custom_provider_name(value, existing_names=existing_names),
    ).strip()
    base_url = (
        tui.prompt_text(
            'Provider base URL',
            validate=validate_base_url,
        )
        .strip()
        .rstrip('/')
    )
    api_key = _prompt_api_key(provider_name, credential_configured=False, required=False)
    use_proxy = tui.prompt_confirm(f'Use the system proxy for {provider_name}?', default=False)
    tui.render_step_divider(
        f'Add Models ({provider_name})',
        f'Provider settings complete. Next, add one or more models for {provider_name}.',
    )
    models = _prompt_additional_models(provider_name, existing_models=(), require_at_least_one=True)
    return ProviderConfiguration(
        key=CUSTOM_PROVIDER_KEY,
        name=provider_name,
        provider_type=ProviderType.OPENAI,
        api_key=api_key,
        credential_configured=api_key is not None,
        base_url=base_url,
        use_proxy=use_proxy,
        models=models,
    )


def _prompt_existing_custom_provider(
    existing: LLMProvider | None,
    draft: ProviderConfiguration | None,
) -> ProviderConfiguration:
    provider_name = draft.name if draft is not None else (existing.name if existing is not None else 'Custom')
    tui.render_section(provider_name, 'Edit OpenAI-compatible provider settings and models.')

    default_url = (
        (draft.base_url if draft is not None else None) or (existing.base_url if existing is not None else None) or ''
    )
    base_url = (
        tui.prompt_text(
            'Provider base URL',
            default=default_url,
            validate=validate_base_url,
        )
        .strip()
        .rstrip('/')
    )
    credential_configured = bool(
        (existing is not None and existing.credential_configured) or (draft is not None and draft.credential_configured)
    )
    api_key_input = _prompt_api_key(provider_name, credential_configured=credential_configured, required=False)
    api_key = api_key_input if api_key_input is not None else (draft.api_key if draft is not None else None)

    default_proxy = draft.use_proxy if draft is not None else (existing.use_proxy if existing is not None else False)
    use_proxy = tui.prompt_confirm(f'Use the system proxy for {provider_name}?', default=default_proxy)

    candidates = custom_model_candidates(existing, draft)
    divider_title = f'Select Models ({provider_name})' if candidates else f'Add Models ({provider_name})'
    divider_desc = (
        f'Provider settings complete. Next, select or add models for {provider_name}.'
        if candidates
        else f'Provider settings complete. Next, add one or more models for {provider_name}.'
    )
    tui.render_step_divider(divider_title, divider_desc)
    if candidates:
        checked_names = (
            {model.name for model in draft.models} if draft is not None else {model.name for model in candidates}
        )
        choices = [
            questionary.Choice(
                _format_model_choice(candidate),
                value=candidate.name,
                checked=candidate.name in checked_names,
            )
            for candidate in candidates
        ]
        selected_names = tui.prompt_checkbox(
            f'Select {provider_name} models to keep enabled (Space to toggle, Enter to confirm)',
            choices=choices,
        )
        candidate_by_name = {candidate.name: candidate for candidate in candidates}
        selected_models = [candidate_by_name[name] for name in selected_names]
        if not selected_models:
            tui.render_notice(
                f'No models selected for {provider_name}; add at least one model.',
                style='#94a3b8',
            )
            selected_models.extend(
                _prompt_additional_models(provider_name, existing_models=(), require_at_least_one=True)
            )
        elif tui.prompt_confirm(f'Add another model to {provider_name}?', default=False):
            selected_models.extend(
                _prompt_additional_models(
                    provider_name,
                    existing_models=selected_models,
                    require_at_least_one=False,
                )
            )
    else:
        selected_models = list(_prompt_additional_models(provider_name, existing_models=(), require_at_least_one=True))

    return ProviderConfiguration(
        key=CUSTOM_PROVIDER_KEY,
        name=provider_name,
        provider_type=ProviderType.OPENAI,
        api_key=api_key,
        credential_configured=credential_configured or api_key is not None,
        base_url=base_url,
        use_proxy=use_proxy,
        models=tuple(selected_models),
    )


def _prompt_configured_provider_action(provider_name: str, *, custom_label: str) -> str:
    return tui.prompt_select(
        f'{provider_name} is already enabled. Choose an action',
        choices=[
            questionary.Choice('Edit credentials & models', value='edit'),
            questionary.Choice(custom_label, value='add_models'),
            questionary.Choice('Disable this provider', value='disable'),
            questionary.Choice('Back', value='back'),
        ],
        default='edit',
    )


def _build_provider_menu(
    providers: Sequence[LLMProvider],
    drafts: Mapping[str, ProviderConfiguration],
) -> list[questionary.Choice]:
    choices: list[questionary.Choice] = []
    if drafts:
        choices.append(questionary.Choice('Done — proceed to role assignment', value='done'))

    for preset in OFFICIAL_PROVIDER_PRESETS:
        draft = drafts.get(preset.name.casefold())
        status = f'[enabled: {len(draft.models)} model(s)]' if draft is not None else '[not configured]'
        choices.append(questionary.Choice(f'{preset.name:<12} {status}', value=f'preset:{preset.key}'))

    seen_custom: set[str] = set()
    for key, draft in drafts.items():
        if draft.key == CUSTOM_PROVIDER_KEY:
            seen_custom.add(key)
            choices.append(
                questionary.Choice(
                    f'{draft.name} [custom · enabled: {len(draft.models)} model(s)]',
                    value=f'custom:{key}',
                )
            )

    for provider in providers:
        normalized = provider.name.casefold()
        if normalized in PRESET_BY_NAME or normalized in seen_custom:
            continue
        seen_custom.add(normalized)
        choices.append(questionary.Choice(f'{provider.name} [custom · disabled]', value=f'custom:{normalized}'))

    choices.append(questionary.Choice('+ Add custom provider (OpenAI-compatible)', value='add_custom'))
    return choices


def prompt_provider_configurations(providers: Sequence[LLMProvider]) -> list[ProviderConfiguration]:
    """Interactively configure one or more built-in or custom AI providers and their models."""
    drafts = initial_provider_drafts(providers)

    while True:
        tui.render_provider_configurations(list(drafts.values()))
        menu_choices = _build_provider_menu(providers, drafts)
        selection = tui.prompt_select(
            'Select an AI provider to configure or add (select Done when finished)',
            choices=menu_choices,
            default='done' if drafts else None,
        )

        if selection == 'done':
            return list(drafts.values())

        if selection == 'add_custom':
            created = _prompt_new_custom_provider(providers, drafts)
            drafts[created.name.casefold()] = created
            continue

        if selection.startswith('preset:'):
            preset_key = selection.removeprefix('preset:')
            preset = PRESET_BY_KEY[preset_key]
            normalized = preset.name.casefold()
            existing = existing_provider(providers, preset.name)
            draft = drafts.get(normalized)

            if draft is not None:
                action = _prompt_configured_provider_action(preset.name, custom_label='Add custom model(s)')
                if action == 'back':
                    continue
                if action == 'disable':
                    drafts.pop(normalized, None)
                    continue
                if action == 'add_models':
                    tui.render_step_divider(
                        f'Add Models ({preset.name})',
                        f'Enter one or more custom model names to add to {preset.name}.',
                    )
                    extra = _prompt_additional_models(
                        preset.name,
                        existing_models=draft.models,
                        require_at_least_one=False,
                    )
                    if extra:
                        drafts[normalized] = draft.model_copy(update={'models': (*draft.models, *extra)})
                    continue

            drafts[normalized] = _prompt_official_provider(preset, existing, draft)
            continue

        if selection.startswith('custom:'):
            normalized = selection.removeprefix('custom:')
            existing = next((provider for provider in providers if provider.name.casefold() == normalized), None)
            draft = drafts.get(normalized)

            if draft is not None:
                action = _prompt_configured_provider_action(draft.name, custom_label='Add model(s)')
                if action == 'back':
                    continue
                if action == 'disable':
                    drafts.pop(normalized, None)
                    continue
                if action == 'add_models':
                    tui.render_step_divider(
                        f'Add Models ({draft.name})',
                        f'Enter one or more model names to add to {draft.name}.',
                    )
                    extra = _prompt_additional_models(
                        draft.name,
                        existing_models=draft.models,
                        require_at_least_one=False,
                    )
                    if extra:
                        drafts[normalized] = draft.model_copy(update={'models': (*draft.models, *extra)})
                    continue

            drafts[normalized] = _prompt_existing_custom_provider(existing, draft)


def prompt_role_assignments(
    models: Sequence[LLMModel], existing: Mapping[ModelRole, LLMModel]
) -> dict[ModelRole, LLMModel]:
    """Show current role assignments and let users edit them one at a time."""
    assignments = dict.fromkeys(ModelRole, models[0])
    available = {model.id: model for model in models}
    assignments.update({role: available[model.id] for role, model in existing.items() if model.id in available})
    model_by_id = {str(model.id): model for model in models}
    label_by_id = {
        str(model.id): f'{_model_provider_name(model)} / {_format_model_label(model.name, model.display_name)}'
        for model in models
    }
    model_choices = [questionary.Choice(label_by_id[str(model.id)], value=str(model.id)) for model in models]
    done_choice = 'Done'
    role_choices = [f'{index}. {label}' for index, label in enumerate(ROLE_LABELS.values(), start=1)]
    role_by_choice = dict(zip(role_choices, ROLE_LABELS, strict=True))

    while True:
        tui.render_model_role_assignments(
            [(role_label, label_by_id[str(assignments[role].id)]) for role, role_label in ROLE_LABELS.items()]
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
        selected_model_id = tui.prompt_select(
            f'Select a model for {ROLE_LABELS[role]}',
            choices=model_choices,
            default=str(current_model.id),
        )
        assignments[role] = model_by_id[selected_model_id]


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
            role_defaults = role_assignments_for_models(session, models)
            role_assignments = prompt_role_assignments(models, role_defaults)
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
