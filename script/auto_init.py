import asyncio
from pathlib import Path

from app.utils.init import tui
from app.utils.init.config import create_settings
from app.utils.init.containers import initialize_project_containers
from app.utils.init.database import initialize_preset_database
from app.utils.init.test_data import initialize_test_data


def _confirm(message: str, *, default: bool) -> bool:
    """统一初始化业务步骤中的确认回调."""
    return tui.prompt_confirm(message, default=default)


async def _confirm_async(message: str, *, default: bool) -> bool:
    """统一异步初始化业务步骤中的确认回调."""
    return await tui.prompt_confirm_async(message, default=default)


def run_auto_init() -> None:
    """系统自动初始化主流程入口."""
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
        from app.core.config import get_settings

        tui.render_notice('Keeping the existing configuration and continuing system initialization.', style='#94a3b8')
        settings = get_settings()
        update_port = None
    else:
        settings, update_port = create_settings(config_path)

    test_data_ready = initialize_test_data(settings.network, settings.persistent.storage, _confirm)
    container_ready = initialize_project_containers(settings, _confirm, update_port=update_port)
    database_ready = None
    if container_ready is not False:
        database_ready = asyncio.run(initialize_preset_database(_confirm_async))

    provider_ready = None
    if container_ready is not False and database_ready is not False:
        # Import after config.toml has been created so the database engine uses the new settings.
        from app.utils.init.providers import initialize_llm_providers

        provider_ready = initialize_llm_providers()

    if any(result is False for result in (test_data_ready, container_ready, database_ready, provider_ready)):
        tui.render_notice('System initialization finished with incomplete steps.', style='#f59e0b')
    else:
        tui.render_system_success()
