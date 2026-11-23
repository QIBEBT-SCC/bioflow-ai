from collections.abc import Awaitable, Callable

from bioflow_ai.cli.initialization import tui
from bioflow_ai.cli.render import create_iterator_renderer
from bioflow_ai.services.initialization import database
from bioflow_ai.services.initialization.database import (
    MAX_PASSWORD_BYTES,
    AdminInitResult,
    PresetDatabaseInitResult,
)


def prepare_database() -> bool:
    """Create the tables needed by the model configuration and data stages."""
    tui.render_section('Database Preparation', 'Create any missing database tables.')
    try:
        from bioflow_ai.infrastructure.database import create_db_and_tables

        create_db_and_tables()
    except Exception as exc:
        tui.render_notice(f'Database preparation failed: {exc}', style='#ef4444')
        return False
    return True


async def init_admin() -> AdminInitResult:
    """Prompt for credentials only when the user table is empty."""
    from pydantic import SecretStr

    from bioflow_ai.schemas.user import UserCreate

    if database.administrator_exists():
        return AdminInitResult(created=False)

    def required(value: str) -> bool | str:
        return True if value.strip() else 'This field is required.'

    email = (await tui.prompt_text_async('Administrator email', validate=required)).strip().casefold()
    username = (await tui.prompt_text_async('Administrator username', default='admin', validate=required)).strip()
    password = await tui.prompt_password_async(
        'Administrator password (any strength, required, maximum 128 UTF-8 bytes)',
        validate=lambda value: (
            required(value)
            if len(value.encode()) <= MAX_PASSWORD_BYTES
            else 'The password cannot exceed 128 UTF-8 bytes.'
        ),
    )
    return await database.init_admin(UserCreate(username=username, email=email, password=SecretStr(password)))


async def init_preset_database_records(
    *, drop_old: bool = True, show_progress: bool = True
) -> PresetDatabaseInitResult:
    """Collect administrator input before running the noninteractive imports."""
    admin = await init_admin()
    return await database.init_preset_database_records(
        admin=admin, drop_old=drop_old, progress=create_iterator_renderer() if show_progress else None
    )


async def initialize_preset_database(confirm: Callable[..., Awaitable[bool]]) -> bool | None:
    """Prompt for and initialize preset SQL records and vector collections."""
    tui.render_section(
        'System Preset Data',
        'Initialize the administrator account, tool database, and tool/image/auxiliary-node vector indexes.',
    )
    if not await confirm('Initialize system preset data and vector indexes?', default=True):
        tui.render_notice('Preset data initialization skipped.', style='#94a3b8')
        return None

    try:
        result = await init_preset_database_records(drop_old=True, show_progress=False)
        await rebuild_auxiliary_node_index(show_progress=False)
    except Exception as exc:
        tui.render_notice(f'Preset data initialization failed: {exc}', style='#ef4444')
        return False

    tui.render_preset_database_summary(result)
    return True


async def init_images(*, drop_old: bool = True, show_progress: bool = True) -> database.PresetImportResult:
    """Import images with CLI-owned progress presentation."""
    return await database.init_images(drop_old=drop_old, progress=create_iterator_renderer() if show_progress else None)


async def init_tools(*, drop_old: bool = True, show_progress: bool = True) -> database.PresetImportResult:
    """Import tools with CLI-owned progress presentation."""
    return await database.init_tools(drop_old=drop_old, progress=create_iterator_renderer() if show_progress else None)


async def rebuild_auxiliary_node_index(*, show_progress: bool = True) -> None:
    """Rebuild the index with CLI-owned progress presentation."""
    await database.rebuild_auxiliary_node_index(progress=create_iterator_renderer() if show_progress else None)
