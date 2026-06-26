from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w
from pydantic import AmqpDsn, PostgresDsn, RedisDsn, SecretStr

from app.utils.init import tui
from app.utils.init.ports import (
    SERVICE_DEFAULT_PORTS,
    UpdatePort,
    create_update_port,
    get_service_label,
)

if TYPE_CHECKING:
    from app.core.config import (
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
    return '请输入一个正整数'


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
    tui.render_section('端口预检查', '检查 API 与 compose 依赖服务宿主机端口。')
    changed_ports = update_port.model_dump(exclude_none=True)
    if not changed_ports:
        tui.render_notice('默认端口均可用。', style='#10b981')
        return

    for key, port in changed_ports.items():
        tui.render_notice(
            f'{get_service_label(key)} 端口 {SERVICE_DEFAULT_PORTS[key]} 已被占用, 已自动顺延为 {port}',
            style='#f59e0b',
        )


def ask_network() -> 'NetworkConfig':
    """Prompt for network settings."""
    from app.core.config import NetworkConfig

    tui.render_section('网络配置', '设置系统网络代理。')

    use_proxy = tui.prompt_confirm('是否使用代理?', default=False)
    if use_proxy:
        proxy_type = tui.prompt_select('代理类型', choices=['http', 'socks5'], default='http')
        proxy_host = tui.prompt_text('代理主机', default='localhost')
        proxy_port = int(tui.prompt_text('代理端口', default='7890', validate=validate_int))
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
    from app.core.config import StorageConfig

    tui.render_section('存储路径', '指定运行数据、生物信息数据库、基因组和测试数据目录。')

    storage_path = tui.prompt_path('数据存储路径 (storage_path)', default='./storage')
    bio_db_path = tui.prompt_path('生物信息数据库路径 (bio_db_path)', default='./storage/bio_dbs')
    genome_path = tui.prompt_path('基因组数据路径 (genome_path)', default='./storage/genome')
    test_data_path = tui.prompt_path('测试数据路径 (test_data_path)', default='./testdata')

    return StorageConfig(
        STORAGE_PATH=storage_path,
        BIO_DB_PATH=bio_db_path,
        GENOME_PATH=genome_path,
        TEST_DATA_PATH=test_data_path,
    )


def ask_resource() -> 'ResourceConfig':
    """Prompt for resource settings."""
    from app.core.config import ResourceConfig

    tui.render_section('资源限制', '控制任务默认线程数和内存上限。')

    max_thread = int(tui.prompt_text('最大线程数 (max_thread)', default='8', validate=validate_int))
    max_mem = tui.prompt_text('最大内存 (max_mem, 例如 4G)', default='4G')

    return ResourceConfig(MAX_THREAD=max_thread, MAX_MEM=max_mem)


def ask_docker() -> 'DockerConfig':
    """Prompt for container runtime settings."""
    from app.core.config import DockerConfig

    tui.render_section('容器运行时', '选择 Docker 或 Podman, 并配置镜像仓库登录信息。')

    container = tui.prompt_select('容器运行时', choices=['docker', 'podman'], default='docker')
    username = tui.prompt_text('docker.io 登录用户名 (可留空)', default='')
    password = tui.prompt_password('docker.io 登录密码 (可留空)')

    # pyrefly: ignore [bad-argument-type]
    return DockerConfig(CONTAINER=container, USERNAME=username, PASSWORD=password)


def ask_knowledge_base() -> 'KnowledgeBaseConfig':
    """Prompt for knowledge-base embedding settings."""
    from app.core.config import KnowledgeBaseConfig

    tui.render_section('知识库嵌入模型', '配置 embedding 模型。')

    model = tui.prompt_text('嵌入模型名称', default='Qwen/Qwen3-Embedding-4B')
    base_url = tui.prompt_text('模型 base_url', default='https://api.siliconflow.cn/v1/')
    api_key = tui.prompt_password('模型 api_key')
    return KnowledgeBaseConfig(
        EMBEDDING_MODEL=model,
        BASE_URL=base_url,
        API_KEY=api_key,
    )


def ask_pdf_parser() -> 'PDFParserConfig':
    """Prompt for PDF parser settings."""
    from app.core.config import MinerUConfig, OpenMinerUConfig, PDFParserConfig

    tui.render_section('PDF 解析器', '选择解析引擎并设置批处理和超时参数。')

    engine = tui.prompt_select(
        'PDF 解析引擎',
        choices=['local', 'open_mineru', 'mineru', 'doc2x', 'grobid'],
        default='local',
    )
    batch_size = int(tui.prompt_text('批处理大小 (batch_size)', default='5', validate=validate_int))

    # pyrefly: ignore [bad-argument-type]
    parser = PDFParserConfig(ENGINE=engine, BATCH_SIZE=batch_size)

    if engine == 'mineru':
        api_url = tui.prompt_text('MinerU API 地址', default='https://api.mineru.app')
        api_key = tui.prompt_password('MinerU API Key')
        timeout = int(tui.prompt_text('超时时间 (秒)', default='600', validate=validate_int))
        parser.mineru = MinerUConfig(API_URL=api_url, API_KEY=api_key, TIMEOUT=timeout)
    elif engine == 'open_mineru':
        api_url = tui.prompt_text('Open MinerU API 地址', default='http://localhost:8000')
        server_url = tui.prompt_text('Open MinerU Server 地址', default='http://localhost:30000')
        timeout = int(tui.prompt_text('超时时间 (秒)', default='600', validate=validate_int))
        parser.open_mineru = OpenMinerUConfig(API_URL=api_url, SERVER_URL=server_url, TIMEOUT=timeout)

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
    from app.core.config import Settings

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
