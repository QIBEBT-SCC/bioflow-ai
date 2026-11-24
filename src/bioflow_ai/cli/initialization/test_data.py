from collections.abc import Callable
from typing import TYPE_CHECKING

from bioflow_ai.cli.initialization import tui
from bioflow_ai.services.initialization.test_data import TESTDATA_SOURCE

if TYPE_CHECKING:
    from bioflow_ai.core.config import NetworkConfig, StorageConfig


from bioflow_ai.services.initialization.test_data import inspect_test_data, sync_test_data


def initialize_test_data(
    network: 'NetworkConfig',
    storage: 'StorageConfig',
    confirm: Callable[..., bool],
) -> bool | None:
    """Prompt for and optionally synchronize the configured test dataset."""
    inspection = inspect_test_data(network, storage)

    tui.render_section('Test Data', 'Optionally synchronize the BioFlowAI test dataset from Hugging Face.')
    tui.console.print(f'[bold]Test data path:[/bold] {inspection.path}')
    tui.console.print(f'[bold]Download proxy:[/bold] {inspection.proxy_text}')

    if inspection.path_is_file:
        tui.render_notice(f'The test data path exists but is not a directory: {inspection.path}', style='#ef4444')
        return False

    if inspection.has_existing_data:
        default_download = False
        prompt = 'The test data directory is not empty. Synchronize or update it?'
    else:
        default_download = True
        prompt = 'Download the test dataset?'

    if not confirm(prompt, default=default_download):
        tui.render_notice('Test data download skipped.', style='#94a3b8')
        return None

    tui.console.print()
    tui.console.print(f'[bold #22d3ee]Syncing:[/bold #22d3ee] {TESTDATA_SOURCE} -> {inspection.path}')

    try:
        plan = sync_test_data(network, storage)
    except ValueError as error:
        tui.render_notice(str(error), style='#ef4444')
        return False
    summary = plan.summary()
    downloads = summary.get('downloads', 0)
    skips = summary.get('skips', 0)
    tui.render_notice(
        f'Test data synchronized to {inspection.path} (downloaded: {downloads}, skipped: {skips}).',
        style='#10b981',
    )
    return True
