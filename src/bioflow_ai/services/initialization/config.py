from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w
from pydantic import AmqpDsn, PostgresDsn, RedisDsn, SecretStr

from bioflow_ai.services.initialization import tui
from bioflow_ai.services.initialization.ports import (
    SERVICE_DEFAULT_PORTS,
    UpdatePort,
    create_update_port,
    get_service_label,
)

if TYPE_CHECKING:
    from bioflow_ai.core.config import (
        DockerConfig,
        KnowledgeBaseConfig,
        NetworkConfig,
        PDFParserConfig,
        ResourceConfig,
        Settings,
        StorageConfig,
    )


def validate_int(text: str) -> bool | str:
    """Validate that the input is a positive integer."""
    if text.strip().isdigit() and int(text) > 0:
        return True
    return 'Enter a positive integer.'


def sanitize_config(value: Any) -> Any:
    """Convert settings values into TOML-serializable primitives."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, dict):
        return {k: sanitize_config(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [sanitize_config(v) for v in value]
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def write_settings(settings: Any, config_path: Path) -> None:
    """Write settings to a TOML config file."""
    config_dict = sanitize_config(settings.model_dump(by_alias=True, mode='python', exclude_none=True))
    with config_path.open('wb') as f:
        tomli_w.dump(config_dict, f)


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
            # pyrefly: ignore [bad-argument-type]
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


def ask_resource() -> 'ResourceConfig':
    """Prompt for resource settings."""
    from bioflow_ai.core.config import ResourceConfig

    tui.render_section('Resource Limits', 'Set the default thread and memory limits for analysis tasks.')

    max_thread = int(tui.prompt_text('Maximum threads (max_thread)', default='8', validate=validate_int))
    max_mem = tui.prompt_text('Maximum memory (max_mem, e.g. 4G)', default='4G')

    return ResourceConfig(MAX_THREAD=max_thread, MAX_MEM=max_mem)


def ask_docker() -> 'DockerConfig':
    """Prompt for container runtime settings."""
    from bioflow_ai.core.config import DockerConfig

    tui.render_section('Container Runtime', 'Select Docker or Podman and configure optional registry credentials.')

    container = tui.prompt_select('Container runtime', choices=['docker', 'podman'], default='docker')
    username = tui.prompt_text('docker.io username (optional)', default='')
    password = tui.prompt_password('docker.io password (optional)')

    # pyrefly: ignore [bad-argument-type]
    return DockerConfig(CONTAINER=container, USERNAME=username, PASSWORD=password)


def ask_knowledge_base() -> 'KnowledgeBaseConfig':
    """Prompt for knowledge-base embedding settings."""
    from bioflow_ai.core.config import KnowledgeBaseConfig

    tui.render_section('Knowledge Base Embedding', 'Configure the embedding model used by the knowledge base.')

    model = tui.prompt_text('Embedding model name', default='Qwen/Qwen3-Embedding-4B')
    base_url = tui.prompt_text('Embedding API base URL', default='https://api.siliconflow.cn/v1/')
    api_key = tui.prompt_password('Embedding API key')
    return KnowledgeBaseConfig(
        EMBEDDING_MODEL=model,
        BASE_URL=base_url,
        API_KEY=api_key,
    )


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

    # pyrefly: ignore [bad-argument-type]
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


def apply_update_port(settings: 'Settings', update_port: UpdatePort) -> None:
    """Apply automatically assigned dependency ports to settings."""
    if host_port := update_port.api:
        settings.network.PORT = host_port

    if postgres_port := update_port.postgres:
        settings.persistent.database.DB_URL = PostgresDsn(
            f'postgresql+psycopg://bioflow_ai:123456@127.0.0.1:{postgres_port}/bioflow_ai'
        )

    if redis_port := update_port.redis:
        settings.persistent.cache.LLM_CACHE = RedisDsn(f'redis://localhost:{redis_port}/0')
        settings.persistent.cache.MONITOR_CACHE = RedisDsn(f'redis://localhost:{redis_port}/2')
        settings.persistent.cache.API_CACHE = RedisDsn(f'redis://localhost:{redis_port}/2')
        settings.persistent.cache.NOTIFICATION_CACHE = RedisDsn(f'redis://localhost:{redis_port}/2')
        settings.celery.BACKEND = RedisDsn(f'redis://localhost:{redis_port}/1')
        settings.develop.CHECKPOINT_URL = RedisDsn(f'redis://localhost:{redis_port}/0')

    if rabbitmq_port := update_port.rabbitmq:
        settings.celery.BROKER = AmqpDsn(f'amqp://bioflow_ai:123456@localhost:{rabbitmq_port}//')

    if rustfs_port := update_port.rustfs:
        settings.persistent.storage.s3.ENDPOINT = f'http://localhost:{rustfs_port}'

    if milvus_port := update_port.milvus:
        settings.knowledge_base.MILVUS_URL = f'http://localhost:{milvus_port}'


def create_settings(config_path: Path) -> tuple['Settings', UpdatePort]:
    """Interactively create settings, persist them, and return the settings object."""
    from bioflow_ai.core.config import Settings

    update_port = create_update_port()

    settings = Settings()
    settings.network = ask_network()
    settings.persistent.storage = ask_storage()
    settings.persistent.resource = ask_resource()
    settings.docker = ask_docker()
    settings.knowledge_base = ask_knowledge_base()
    settings.pdf_parser = ask_pdf_parser()

    render_port_preflight(update_port)
    apply_update_port(settings, update_port)

    write_settings(settings, config_path)
    tui.render_success(config_path)
    return settings, update_port
