import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.utils.init import tui
from app.utils.init.common import proxy_environment, resolve_path

if TYPE_CHECKING:
    from app.core.config import NetworkConfig, StorageConfig

TESTDATA_SOURCE = 'hf://buckets/Aye10032/bioflow-ai-testdata'


@dataclass(frozen=True)
class TestDataInspection:
    path: Path
    proxy_text: str
    has_existing_data: bool
    path_is_file: bool


def inspect_test_data(network: 'NetworkConfig', storage: 'StorageConfig') -> TestDataInspection:
    """Inspect the configured test-data location before prompting the user."""
    test_data_path = resolve_path(storage.TEST_DATA_PATH)
    path_is_file = test_data_path.exists() and not test_data_path.is_dir()
    has_existing_data = test_data_path.exists() and test_data_path.is_dir() and any(test_data_path.iterdir())
    proxy_text = network.get_proxy_url() if network.USE_PROXY else 'No proxy'

    return TestDataInspection(
        path=test_data_path,
        proxy_text=proxy_text,
        has_existing_data=has_existing_data,
        path_is_file=path_is_file,
    )


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

    command = ['uvx', 'hf', 'sync', TESTDATA_SOURCE, str(inspection.path)]
    tui.console.print()
    tui.console.print(f'[bold #22d3ee]Running:[/bold #22d3ee] {" ".join(command)}')

    try:
        inspection.path.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, check=False, env=proxy_environment(network))
    except FileNotFoundError:
        tui.render_notice('uvx was not found. Install uv and try again: https://docs.astral.sh/uv/', style='#ef4444')
        return False

    if result.returncode != 0:
        tui.render_notice(
            f'Test data download failed; uvx hf sync exited with code {result.returncode}.', style='#ef4444'
        )
        return False

    tui.render_notice(f'Test data synchronized to {inspection.path}.', style='#10b981')
    return True
