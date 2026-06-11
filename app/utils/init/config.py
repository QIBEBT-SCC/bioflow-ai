import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w
from pydantic import SecretStr

from app.utils.init import tui

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

DEFAULT_PORT = 8000
MAX_PORT = 65536


def validate_int(text: str) -> bool | str:
    """Validate that the input is a positive integer."""
    if text.strip().isdigit() and int(text) > 0:
        return True
    return '请输入一个正整数'


def is_port_available(port: int, host: str = '0.0.0.0') -> bool:  # noqa: S104
    """Check whether a host port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def find_available_port(start: int, host: str = '0.0.0.0', limit: int = 100) -> int:  # noqa: S104
    """Find the first available port from start, falling back to start."""
    for port in range(start, min(start + limit, MAX_PORT)):
        if is_port_available(port, host):
            return port
    return start


def validate_port(text: str) -> bool | str:
    """Validate that the input is a legal and currently available port."""
    if not (text.strip().isdigit() and 0 < int(text) < MAX_PORT):
        return '请输入一个有效的端口号 (1-65535)'
    if not is_port_available(int(text)):
        return f'端口 {text} 已被占用, 请换一个'
    return True


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


def ask_network() -> 'NetworkConfig':
    """Prompt for network settings."""
    from app.core.config import NetworkConfig

    tui.render_section('网络配置', '设置服务监听端口和可选代理。')

    default_port = find_available_port(DEFAULT_PORT)
    if default_port != DEFAULT_PORT:
        tui.render_notice(f'端口 {DEFAULT_PORT} 已被占用, 已自动顺延推荐 {default_port}')
    port = int(tui.prompt_text('服务监听端口', default=str(default_port), validate=validate_port))
    use_proxy = tui.prompt_confirm('是否使用代理?', default=False)

    if use_proxy:
        proxy_type = tui.prompt_select('代理类型', choices=['http', 'socks5'], default='http')
        proxy_host = tui.prompt_text('代理主机', default='localhost')
        proxy_port = int(tui.prompt_text('代理端口', default='7890', validate=validate_int))
        return NetworkConfig(
            PORT=port,
            USE_PROXY=True,
            # pyrefly: ignore [bad-argument-type]
            PROXY_TYPE=proxy_type,
            PROXY_HOST=proxy_host,
            PROXY_PORT=proxy_port,
        )

    return NetworkConfig(PORT=port, USE_PROXY=False)


def ask_storage() -> 'StorageConfig':
    """Prompt for storage settings."""
    from app.core.config import StorageConfig

    tui.render_section('存储路径', '指定运行数据、基因组和测试数据目录。')

    storage_path = tui.prompt_path('数据存储路径 (storage_path)', default='./storage')
    genome_path = tui.prompt_path('基因组数据路径 (genome_path)', default='./storage/genome')
    test_data_path = tui.prompt_path('测试数据路径 (test_data_path)', default='./testdata')

    return StorageConfig(
        STORAGE_PATH=storage_path,
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
    from app.core.config import KnowledgeBaseConfig, LocalEmbeddingModel, RemoteEmbeddingModel

    tui.render_section('知识库嵌入模型', '配置本地或远程 embedding 模型。')

    use_local = tui.prompt_confirm('是否使用本地嵌入模型?', default=False)

    if use_local:
        model = tui.prompt_text('本地模型名称', default='Qwen/Qwen3-Embedding-4B')
        use_modelscope = tui.prompt_confirm('是否使用 ModelScope 下载?', default=False)
        device = tui.prompt_text('运行设备 (例如 cuda:0)', default='cuda:0')
        return KnowledgeBaseConfig(
            USE_LOCAL_MODEL=True,
            local=LocalEmbeddingModel(MODEL=model, USE_MODELSCOPE=use_modelscope, DEVICE=device),
        )

    model = tui.prompt_text('远程模型名称', default='Qwen/Qwen3-Embedding-4B')
    base_url = tui.prompt_text('远程模型 base_url', default='https://api.siliconflow.cn/v1/')
    api_key = tui.prompt_password('远程模型 api_key')
    return KnowledgeBaseConfig(
        USE_LOCAL_MODEL=False,
        remote=RemoteEmbeddingModel(MODEL=model, BASE_URL=base_url, API_KEY=api_key),
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


def create_settings(config_path: Path) -> 'Settings':
    """Interactively create settings, persist them, and return the settings object."""
    from app.core.config import Settings

    settings = Settings()
    settings.network = ask_network()
    settings.persistent.storage = ask_storage()
    settings.persistent.resource = ask_resource()
    settings.docker = ask_docker()
    settings.knowledge_base = ask_knowledge_base()
    settings.pdf_parser = ask_pdf_parser()

    write_settings(settings, config_path)
    tui.render_success(config_path)
    return settings
