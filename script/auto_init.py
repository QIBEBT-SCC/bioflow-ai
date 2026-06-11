from pathlib import Path

from app.utils.init import tui
from app.utils.init.config import create_settings
from app.utils.init.containers import initialize_project_containers
from app.utils.init.test_data import initialize_test_data


def _confirm(message: str, *, default: bool) -> bool:
    """统一初始化业务步骤中的确认回调."""
    return tui.prompt_confirm(message, default=default)


def run_auto_init() -> None:
    """系统自动初始化主流程入口."""
    config_path = Path('config.toml')

    tui.render_hero()

    use_existing_config = False
    if config_path.exists():
        overwrite = tui.prompt_confirm(
            f'{config_path} 已存在, 是否覆盖?',
            default=False,
        )
        use_existing_config = not overwrite

    if use_existing_config:
        from app.core.config import get_settings

        tui.render_notice('保留现有配置, 将继续执行系统初始化步骤。', style='#94a3b8')
        settings = get_settings()
    else:
        settings = create_settings(config_path)

    test_data_ready = initialize_test_data(settings.network, settings.persistent.storage, _confirm)
    container_ready = initialize_project_containers(settings, _confirm)
    if test_data_ready is False or container_ready is False:
        tui.render_notice('系统初始化流程已结束, 但存在未完成的初始化步骤。', style='#f59e0b')
    else:
        tui.render_system_success()
