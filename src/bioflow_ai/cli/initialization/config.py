from pathlib import Path
from typing import TYPE_CHECKING

import click

from bioflow_ai.cli.initialization import tui
from bioflow_ai.services.initialization.config import (
    apply_update_port,
    validate_int,
    write_embedding_settings,
    write_settings,
)
from bioflow_ai.services.initialization.ports import (
    SERVICE_DEFAULT_PORTS,
    UpdatePort,
    create_update_port,
    get_service_label,
)

if TYPE_CHECKING:
    from bioflow_ai.core.config import (
        DockerConfig,
        NetworkConfig,
        PDFParserConfig,
        Settings,
        StorageConfig,
    )


def render_port_preflight(update_port: UpdatePort) -> None:
    """Render automatic port-shift results before config prompts."""
    tui.render_section('Port Preflight', 'Check host ports used by the API and Compose services.')
    changed_ports = update_port.model_dump(exclude_none=True)
    if not changed_ports:
        tui.render_notice('All default ports are available.', style='#10b981')
        return

    for key, port in changed_ports.items():
        tui.render_notice(
            f'{get_service_label(key)} port {SERVICE_DEFAULT_PORTS[key]} is in use; reassigned to {port}.',
            style='#f59e0b',
        )


def ask_network() -> 'NetworkConfig':
    """Prompt for network settings."""
    from bioflow_ai.core.config import NetworkConfig

    tui.render_section('Network Configuration', 'Configure the system-wide network proxy.')

    use_proxy = tui.prompt_confirm('Use a network proxy?', default=False)
    if use_proxy:
        proxy_type = tui.prompt_select('Proxy type', choices=['http', 'socks5'], default='http')
        proxy_host = tui.prompt_text('Proxy host', default='localhost')
        proxy_port = int(tui.prompt_text('Proxy port', default='7890', validate=validate_int))
        return NetworkConfig(
            USE_PROXY=True,
            # ty: ignore[invalid-argument-type]
            PROXY_TYPE=proxy_type,
            PROXY_HOST=proxy_host,
            PROXY_PORT=proxy_port,
        )

    return NetworkConfig(USE_PROXY=False)


def ask_storage() -> 'StorageConfig':
    """Prompt for storage settings."""
    from bioflow_ai.core.config import StorageConfig

    tui.render_section(
        'Storage Paths',
        'Set directories for runtime data, biological databases, reference genomes, and test data.',
    )

    storage_path = tui.prompt_path('Runtime data path (storage_path)', default='./storage')
    bio_db_path = tui.prompt_path('Biological database path (bio_db_path)', default='./storage/bio_dbs')
    genome_path = tui.prompt_path('Reference genome path (genome_path)', default='./storage/genome')
    test_data_path = tui.prompt_path('Test data path (test_data_path)', default='./testdata')

    return StorageConfig(
        STORAGE_PATH=storage_path,
        BIO_DB_PATH=bio_db_path,
        GENOME_PATH=genome_path,
        TEST_DATA_PATH=test_data_path,
    )


def ask_docker() -> 'DockerConfig':
    """Prompt for container runtime settings."""
    from bioflow_ai.core.config import DockerConfig

    tui.render_section('Container Runtime', 'Select Docker or Podman as the container engine.')

    container = tui.prompt_select('Container runtime', choices=['docker', 'podman'], default='docker')

    # ty: ignore[invalid-argument-type]
    return DockerConfig(CONTAINER=container)


def initialize_embedding(config_path: Path) -> None:
    """Keep or update the model used to build vector indexes."""
    from bioflow_ai.core.config import get_settings
    from bioflow_ai.services.initialization.providers import validate_base_url, validate_required_text

    tui.render_subsection('Embedding Model')
    current = get_settings().knowledge_base
    has_key = bool(current.API_KEY.get_secret_value().strip())
    if has_key and not tui.prompt_confirm(
        f'Update the saved embedding model ({current.EMBEDDING_MODEL})?', default=False
    ):
        return

    model = tui.prompt_text('Embedding model name', default=current.EMBEDDING_MODEL, validate=validate_required_text)
    base_url = tui.prompt_text('Embedding API base URL', default=current.BASE_URL, validate=validate_base_url)
    key_prompt = 'Embedding API key (leave blank to keep the current key)' if has_key else 'Embedding API key'
    api_key = tui.prompt_password(key_prompt)
    try:
        write_embedding_settings(config_path, model=model.strip(), base_url=base_url.strip(), api_key=api_key.strip())
    except OSError as exc:
        message = f'Could not save embedding settings: {exc}'
        raise click.ClickException(message) from exc
    get_settings.cache_clear()
    tui.render_notice('Embedding settings saved.', style='#10b981')


def ask_pdf_parser() -> 'PDFParserConfig':
    """Prompt for PDF parser settings."""
    from bioflow_ai.core.config import MinerUConfig, OpenMinerUConfig, PDFParserConfig

    tui.render_section('PDF Parser', 'Select a parsing engine and configure batching and timeouts.')

    engine = tui.prompt_select(
        'PDF parser engine',
        choices=['local', 'open_mineru', 'mineru', 'doc2x', 'grobid'],
        default='local',
    )
    batch_size = int(tui.prompt_text('Batch size (batch_size)', default='5', validate=validate_int))

    # ty: ignore[invalid-argument-type]
    parser = PDFParserConfig(ENGINE=engine, BATCH_SIZE=batch_size)

    if engine == 'mineru':
        api_url = tui.prompt_text('MinerU API URL', default='https://api.mineru.app')
        api_key = tui.prompt_password('MinerU API Key')
        timeout = int(tui.prompt_text('Request timeout (seconds)', default='600', validate=validate_int))
        parser.mineru = MinerUConfig(API_URL=api_url, API_KEY=api_key, TIMEOUT=timeout)
    elif engine == 'open_mineru':
        api_url = tui.prompt_text('Open MinerU API URL', default='http://localhost:8000')
        timeout = int(tui.prompt_text('Request timeout (seconds)', default='600', validate=validate_int))
        parser.open_mineru = OpenMinerUConfig(API_URL=api_url, TIMEOUT=timeout)

    return parser


def create_settings(
    config_path: Path,
    *,
    network: 'NetworkConfig | None' = None,
) -> tuple['Settings', UpdatePort]:
    """Interactively create settings, persist them, and return the settings object."""
    from bioflow_ai.core.config import Settings

    update_port = create_update_port()

    settings = Settings()
    settings.network = network if network is not None else ask_network()
    settings.persistent.storage = ask_storage()
    settings.docker = ask_docker()
    settings.pdf_parser = ask_pdf_parser()

    render_port_preflight(update_port)
    apply_update_port(settings, update_port)

    write_settings(settings, config_path)
    tui.render_success(config_path)
    return settings, update_port
