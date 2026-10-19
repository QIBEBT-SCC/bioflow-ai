import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from bioflow_ai.services.initialization.common import proxy_environment, resolve_path

if TYPE_CHECKING:
    from bioflow_ai.core.config import NetworkConfig, StorageConfig

TESTDATA_SOURCE = 'hf://buckets/Aye10032/bioflow-ai-testdata'


class TestDataInspection(BaseModel):
    model_config = ConfigDict(frozen=True)

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


def sync_test_data(network: 'NetworkConfig', storage: 'StorageConfig') -> None:
    """Synchronize the configured test dataset without collecting user input."""
    inspection = inspect_test_data(network, storage)
    command = ['uvx', 'hf', 'sync', TESTDATA_SOURCE, str(inspection.path)]
    try:
        inspection.path.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, check=False, env=proxy_environment(network))
    except FileNotFoundError as error:
        message = 'uvx was not found. Install uv and try again: https://docs.astral.sh/uv/'
        raise ValueError(message) from error
    if result.returncode != 0:
        message = f'Test data download failed; uvx hf sync exited with code {result.returncode}.'
        raise ValueError(message)
