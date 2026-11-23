import asyncio
from pathlib import Path

import click

from bioflow_ai.cli.initialization import tui
from bioflow_ai.cli.initialization.config import create_settings, initialize_embedding
from bioflow_ai.cli.initialization.containers import initialize_project_containers
from bioflow_ai.cli.initialization.database import initialize_preset_database, prepare_database
from bioflow_ai.cli.initialization.providers import initialize_llm_providers
from bioflow_ai.cli.initialization.services import (
    initialize_backend_services,
    resolve_web_ui_port,
    stop_running_services_before_reconfigure,
)
from bioflow_ai.cli.initialization.test_data import initialize_test_data
from bioflow_ai.services.initialization.presets import install_deployment_presets


def _confirm(message: str, *, default: bool) -> bool:
    """Shared confirmation callback for synchronous initialization stages."""
    return tui.prompt_confirm(message, default=default)


async def _confirm_async(message: str, *, default: bool) -> bool:
    """Shared confirmation callback for asynchronous initialization stages."""
    return await tui.prompt_confirm_async(message, default=default)


def run_auto_init() -> None:
    """Run the automatic system initialization workflow."""
    config_path = Path('config.toml')

    tui.render_hero()

    use_existing_config = False
    if config_path.exists():
        overwrite = tui.prompt_confirm(
            f'{config_path} already exists. Overwrite it?',
            default=False,
        )
        use_existing_config = not overwrite

    if use_existing_config:
        from bioflow_ai.core.config import get_settings

        tui.render_notice('Keeping the existing configuration and continuing system initialization.', style='#94a3b8')
        settings = get_settings()
        update_port = None
    else:
        stop_running_services_before_reconfigure(_confirm)
        settings, update_port = create_settings(config_path)

    try:
        install_deployment_presets()
    except OSError as exc:
        message = f'Could not install deployment presets: {exc}'
        raise click.ClickException(message) from exc

    container_ready = initialize_project_containers(settings, _confirm, update_port=update_port)
    if container_ready is False or not prepare_database():
        tui.render_notice(
            'Initialization stopped during service preparation. Fix the reported error and run auto-init again.'
        )
        return

    tui.render_section(
        'Model Configuration', 'Set up the embedding model for vector indexes and chat models for agents.'
    )
    initialize_embedding(config_path)
    provider_ready = initialize_llm_providers(reuse_existing=use_existing_config)

    tui.render_section('Data Initialization', 'Import system presets and optionally download the test dataset.')
    preset_data_ready = asyncio.run(initialize_preset_database(_confirm_async))
    test_data_ready = initialize_test_data(settings.network, settings.persistent.storage, _confirm)
    services_ready = initialize_backend_services(_confirm)

    if any(result is False for result in (provider_ready, preset_data_ready, test_data_ready, services_ready)):
        tui.render_notice('System initialization finished with incomplete steps.', style='#f59e0b')
    else:
        tui.render_system_success(
            web_ui_url=f'http://localhost:{resolve_web_ui_port(update_port)}',
            api_url=f'http://localhost:{settings.network.PORT}',
            services_started=bool(services_ready),
        )
