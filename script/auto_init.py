import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

import questionary
import tomli_w
from pydantic import SecretStr
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from app.core.config import (
        DockerConfig,
        KnowledgeBaseConfig,
        NetworkConfig,
        PDFParserConfig,
        ResourceConfig,
        StorageConfig,
    )

console = Console()

_DEFAULT_PORT = 8000
_MAX_PORT = 65536
# 回答完成后, 将上一行行首的问号 (?) 覆写为绿色对勾
_DONE_MARK = '\x1b[1A\r\x1b[38;2;16;185;129m✓\x1b[0m\x1b[1B\r'
_PROMPT_STYLE = questionary.Style(
    [
        ('qmark', 'fg:#22d3ee bold'),
        ('question', 'fg:#e5e7eb bold'),
        ('answer', 'fg:#10b981 bold'),
        ('pointer', 'fg:#f59e0b bold'),
        ('highlighted', 'fg:#22d3ee bold'),
        ('selected', 'fg:#10b981'),
        ('separator', 'fg:#64748b'),
        ('instruction', 'fg:#94a3b8'),
        ('text', 'fg:#e5e7eb'),
        ('disabled', 'fg:#64748b italic'),
    ]
)


def _finish(question: questionary.Question) -> Any:
    """提交问题并在完成后把行首问号替换为对勾."""
    answer = question.unsafe_ask()
    console.file.write(_DONE_MARK)
    console.file.flush()
    return answer


def _prompt_text(message: str, *, default: str = '', validate: Any = None) -> str:
    """使用统一主题询问文本输入."""
    return _finish(questionary.text(message, default=default, validate=validate, style=_PROMPT_STYLE))


def _prompt_password(message: str) -> str:
    """使用统一主题询问密码输入."""
    return _finish(questionary.password(message, style=_PROMPT_STYLE))


def _prompt_confirm(message: str, *, default: bool = False) -> bool:
    """使用统一主题询问确认项."""
    return _finish(questionary.confirm(message, default=default, style=_PROMPT_STYLE))


def _prompt_select(message: str, *, choices: list[str], default: str) -> str:
    """使用统一主题询问单选项."""
    return _finish(questionary.select(message, choices=choices, default=default, style=_PROMPT_STYLE))


def _prompt_path(message: str, *, default: str) -> str:
    """使用统一主题询问路径输入."""
    return _finish(questionary.path(message, default=default, style=_PROMPT_STYLE))


def _render_hero() -> None:
    """渲染初始化向导首页."""
    title = Text('BioFlowAI', style='bold #22d3ee')
    title.append(' 配置初始化向导', style='bold #f8fafc')

    body = Table.grid(padding=(0, 1))
    body.add_column(justify='center')
    body.add_row(title)
    body.add_row(Text('按步骤生成 config.toml, 默认值可直接回车确认。', style='#94a3b8'))

    console.print()
    console.print(
        Panel(
            Align.center(body),
            border_style='#22d3ee',
            box=box.ROUNDED,
            padding=(1, 4),
        )
    )
    console.print()


def _render_section(title: str, description: str) -> None:
    """渲染现代化分区标题."""
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text(title, style='bold #f8fafc'))
    body.add_row(Text(description, style='#94a3b8'))

    console.print()
    console.print(
        Panel(
            body,
            border_style='#334155',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _render_notice(message: str, *, style: str = '#f59e0b') -> None:
    """渲染提示信息."""
    console.print(Panel(Text(message, style=style), border_style=style, box=box.ROUNDED, padding=(0, 2)))


def _render_success(config_path: Path) -> None:
    """渲染完成信息."""
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text('配置初始化完成', style='bold #10b981'))
    body.add_row(Text(f'已写入 {config_path}', style='#cbd5e1'))

    console.print()
    console.print(
        Panel(
            body,
            title='[bold #10b981]Done[/bold #10b981]',
            border_style='#10b981',
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _validate_int(text: str) -> bool | str:
    """校验输入是否为正整数."""
    if text.strip().isdigit() and int(text) > 0:
        return True
    return '请输入一个正整数'


def _is_port_available(port: int, host: str = '0.0.0.0') -> bool:  # noqa: S104
    """检测指定端口是否可用 (未被占用)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def _find_available_port(start: int, host: str = '0.0.0.0', limit: int = 100) -> int:  # noqa: S104
    """从 start 开始向后查找首个可用端口, 找不到则返回 start."""
    for port in range(start, min(start + limit, _MAX_PORT)):
        if _is_port_available(port, host):
            return port
    return start


def _validate_port(text: str) -> bool | str:
    """校验端口号合法且未被占用."""
    if not (text.strip().isdigit() and 0 < int(text) < _MAX_PORT):
        return '请输入一个有效的端口号 (1-65535)'
    if not _is_port_available(int(text)):
        return f'端口 {text} 已被占用, 请换一个'
    return True


def _sanitize(value: Any) -> Any:
    """递归转换配置值, 还原 SecretStr 明文并将非基础类型转为字符串以便写入 TOML."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _ask_network() -> 'NetworkConfig':
    from app.core.config import NetworkConfig

    _render_section('网络配置', '设置服务监听端口和可选代理。')

    default_port = _find_available_port(_DEFAULT_PORT)
    if default_port != _DEFAULT_PORT:
        _render_notice(f'端口 {_DEFAULT_PORT} 已被占用, 已自动顺延推荐 {default_port}')
    port = int(_prompt_text('服务监听端口', default=str(default_port), validate=_validate_port))
    use_proxy = _prompt_confirm('是否使用代理?', default=False)

    if use_proxy:
        proxy_type = _prompt_select('代理类型', choices=['http', 'socks5'], default='http')
        proxy_host = _prompt_text('代理主机', default='localhost')
        proxy_port = int(_prompt_text('代理端口', default='7890', validate=_validate_int))
        return NetworkConfig(
            PORT=port,
            USE_PROXY=True,
            # pyrefly: ignore [bad-argument-type]
            PROXY_TYPE=proxy_type,
            PROXY_HOST=proxy_host,
            PROXY_PORT=proxy_port,
        )

    return NetworkConfig(PORT=port, USE_PROXY=False)


def _ask_storage() -> 'StorageConfig':
    from app.core.config import StorageConfig

    _render_section('存储路径', '指定运行数据、基因组和测试数据目录。')

    storage_path = _prompt_path('数据存储路径 (storage_path)', default='./storage')
    genome_path = _prompt_path('基因组数据路径 (genome_path)', default='./storage/genome')
    test_data_path = _prompt_path('测试数据路径 (test_data_path)', default='./testdata')

    return StorageConfig(
        STORAGE_PATH=storage_path,
        GENOME_PATH=genome_path,
        TEST_DATA_PATH=test_data_path,
    )


def _ask_resource() -> 'ResourceConfig':
    from app.core.config import ResourceConfig

    _render_section('资源限制', '控制任务默认线程数和内存上限。')

    max_thread = int(_prompt_text('最大线程数 (max_thread)', default='8', validate=_validate_int))
    max_mem = _prompt_text('最大内存 (max_mem, 例如 4G)', default='4G')

    return ResourceConfig(MAX_THREAD=max_thread, MAX_MEM=max_mem)


def _ask_docker() -> 'DockerConfig':
    from app.core.config import DockerConfig

    _render_section('容器运行时', '选择 Docker 或 Podman, 并配置镜像仓库登录信息。')

    container = _prompt_select('容器运行时', choices=['docker', 'podman'], default='docker')
    username = _prompt_text('docker.io 登录用户名 (可留空)', default='')
    password = _prompt_password('docker.io 登录密码 (可留空)')

    # pyrefly: ignore [bad-argument-type]
    return DockerConfig(CONTAINER=container, USERNAME=username, PASSWORD=password)


def _ask_knowledge_base() -> 'KnowledgeBaseConfig':
    from app.core.config import KnowledgeBaseConfig, LocalEmbeddingModel, RemoteEmbeddingModel

    _render_section('知识库嵌入模型', '配置本地或远程 embedding 模型。')

    use_local = _prompt_confirm('是否使用本地嵌入模型?', default=False)

    if use_local:
        model = _prompt_text('本地模型名称', default='Qwen/Qwen3-Embedding-4B')
        use_modelscope = _prompt_confirm('是否使用 ModelScope 下载?', default=False)
        device = _prompt_text('运行设备 (例如 cuda:0)', default='cuda:0')
        return KnowledgeBaseConfig(
            USE_LOCAL_MODEL=True,
            local=LocalEmbeddingModel(MODEL=model, USE_MODELSCOPE=use_modelscope, DEVICE=device),
        )

    model = _prompt_text('远程模型名称', default='Qwen/Qwen3-Embedding-4B')
    base_url = _prompt_text('远程模型 base_url', default='https://api.siliconflow.cn/v1/')
    api_key = _prompt_password('远程模型 api_key')
    return KnowledgeBaseConfig(
        USE_LOCAL_MODEL=False,
        remote=RemoteEmbeddingModel(MODEL=model, BASE_URL=base_url, API_KEY=api_key),
    )


def _ask_pdf_parser() -> 'PDFParserConfig':
    from app.core.config import MinerUConfig, OpenMinerUConfig, PDFParserConfig

    _render_section('PDF 解析器', '选择解析引擎并设置批处理和超时参数。')

    engine = _prompt_select(
        'PDF 解析引擎',
        choices=['local', 'open_mineru', 'mineru', 'doc2x', 'grobid'],
        default='local',
    )
    batch_size = int(_prompt_text('批处理大小 (batch_size)', default='5', validate=_validate_int))

    # pyrefly: ignore [bad-argument-type]
    parser = PDFParserConfig(ENGINE=engine, BATCH_SIZE=batch_size)

    if engine == 'mineru':
        api_url = _prompt_text('MinerU API 地址', default='https://api.mineru.app')
        api_key = _prompt_password('MinerU API Key')
        timeout = int(_prompt_text('超时时间 (秒)', default='600', validate=_validate_int))
        parser.mineru = MinerUConfig(API_URL=api_url, API_KEY=api_key, TIMEOUT=timeout)
    elif engine == 'open_mineru':
        api_url = _prompt_text('Open MinerU API 地址', default='http://localhost:8000')
        server_url = _prompt_text('Open MinerU Server 地址', default='http://localhost:30000')
        timeout = int(_prompt_text('超时时间 (秒)', default='600', validate=_validate_int))
        parser.open_mineru = OpenMinerUConfig(API_URL=api_url, SERVER_URL=server_url, TIMEOUT=timeout)

    return parser


def run_auto_init() -> None:
    """系统自动初始化主流程入口."""
    from app.core.config import Settings

    config_path = Path('config.toml')

    _render_hero()

    if config_path.exists():
        overwrite = _prompt_confirm(
            f'{config_path} 已存在, 是否覆盖?',
            default=False,
        )
        if not overwrite:
            _render_notice('已取消, 未修改任何配置。')
            return

    settings = Settings()
    settings.network = _ask_network()
    settings.persistent.storage = _ask_storage()
    settings.persistent.resource = _ask_resource()
    settings.docker = _ask_docker()
    settings.knowledge_base = _ask_knowledge_base()
    settings.pdf_parser = _ask_pdf_parser()

    config_dict = _sanitize(settings.model_dump(by_alias=True, mode='python', exclude_none=True))
    with config_path.open('wb') as f:
        tomli_w.dump(config_dict, f)

    _render_success(config_path)
