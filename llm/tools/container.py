import re
from typing import Any

from anyio import Path
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.session import engine
from app.schemas.tool_config import ImageConfig
from app.services import tool_catalog
from app.utils.container import ContainerMount, run_command
from llm.context import TestRunContext

MAX_CHAR = 3000
DEFAULT_LINUX_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-utils:1.2.0')


@tool('run_command', parse_docstring=True)
def run_command_tool(command: str, runtime: ToolRuntime[Any]) -> str:
    """Run command in a container and return the result.

    **This tool prohibits parallel calls! You are only allowed to run one command at a time!**

    Args:
        command: the command you need to run
    """
    image = runtime.state.get('image', DEFAULT_LINUX_IMAGE)
    _code, result = run_command(image, [], command)

    return f'[Container: {image.get_image()}]\n{result}'


def _validate_command(command: str, workdir: str | None, tool_call_id: str | None) -> Command | None:
    if re.search(r'\brm\b', command) and (workdir and '/data/output' not in workdir) and '/data/output' not in command:
        return _blocked_command('It is forbidden to use `rm` command outside of the /data/output path!', tool_call_id)

    if re.search(r'\bcd\b', command):
        return _blocked_command(
            'The `cd` command is prohibited. To specify the working directory, please use the `workdir` parameter.',
            tool_call_id,
        )

    return None


def _resolve_image_ref(
    image_ref: str | None,
    runtime: ToolRuntime[TestRunContext],
) -> tuple[ImageConfig, dict[str, str], Command | None]:
    """Resolve a short image reference to an ImageConfig and update cache mappings."""
    known_images = dict(runtime.state.get('known_images') or {})
    if not image_ref:
        return DEFAULT_LINUX_IMAGE, known_images, None

    cached_image = known_images.get(image_ref)
    if cached_image:
        return ImageConfig.from_str(cached_image), known_images, None

    try:
        with Session(engine) as session:
            tool_image = tool_catalog.resolve_image_reference(session, image_ref)
    except tool_catalog.AmbiguousToolImageError as error:
        return DEFAULT_LINUX_IMAGE, known_images, _blocked_command(str(error), runtime.tool_call_id)

    if tool_image is None:
        return (
            DEFAULT_LINUX_IMAGE,
            known_images,
            _blocked_command(
                f'No local image matches `{image_ref}`. Use `search_local_image` first, then pass a unique image reference from the results.',
                runtime.tool_call_id,
            ),
        )

    full_image = tool_image.image.get_image()
    known_images[image_ref] = full_image
    known_images[full_image] = full_image
    return tool_image.image, known_images, None


async def _validate_extra_mounts(
    mounts: list[ContainerMount] | None,
    tool_call_id: str | None,
) -> tuple[list[ContainerMount], Command | None]:
    if mounts is None:
        return [], None

    default_mounts = {'/data/testdata', '/data/output'}
    for mount in mounts:
        if mount.target in default_mounts:
            return [], _blocked_command(
                f'The container path `{mount.target}` is already managed and mounted by default, '
                'so manual configuration is not required. '
                'If additional data needs to be mounted, use a different path.',
                tool_call_id,
            )

        if mount.source and not await Path(mount.source).exists():
            return [], _blocked_command(
                f'The source path `{mount.source}` does not exist. `source` must be an actual directory or file path on the host machine.',
                tool_call_id,
            )

    return mounts, None


def _blocked_command(content: str, tool_call_id: str | None) -> Command:
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def _truncate_output(command: str, output: str) -> str:
    if any(k in command for k in ('about.md', 'DESCRIPTION.md')) or len(output) <= MAX_CHAR:
        return output
    return f'{output[:1500]}\n\n...(collapsed)...\n\n{output[-500:]}'


@tool('run_command', parse_docstring=True)
async def advance_run_command_tool(
    command: str,
    runtime: ToolRuntime[TestRunContext],
    image_ref: str | None = None,
    workdir: str | None = '/data/output',
    mounts: list[ContainerMount] | None = None,
) -> Command:
    """在容器环境中运行指定 bash 命令或生物信息学命令行工具并返回执行结果.

    Args:
        command: 要运行的Linux命令行指令
        image_ref: 用于执行命令的docker镜像, 必须包含镜像的repository, registry、namespace和tag可以视情况省略; 若不指定则使用默认的Linux基础镜像
        workdir: 指定命令运行的工作目录
        mounts: 除默认测试数据和输出目录外, **额外的**数据挂载
    """
    # 指令合法性检查
    if invalid_command := _validate_command(command, workdir, runtime.tool_call_id):
        return invalid_command

    # 镜像解析
    image, known_images, invalid_image = _resolve_image_ref(image_ref, runtime)
    if invalid_image is not None:
        return invalid_image

    # 固定挂载
    base_path = await (await Path(get_settings().persistent.storage.STORAGE_PATH).expanduser()).resolve()
    output_path = base_path / 'testrun' / runtime.context.test_run_id
    await output_path.mkdir(parents=True, exist_ok=True)

    testdata_path = await (await Path(get_settings().persistent.storage.TEST_DATA_PATH).expanduser()).resolve()

    extra_mounts, invalid_mount = await _validate_extra_mounts(mounts, runtime.tool_call_id)
    if invalid_mount is not None:
        return invalid_mount

    all_mounts = [
        ContainerMount(target='/data/testdata', source=str(testdata_path), read_only=True),
        ContainerMount(target='/data/output', source=str(output_path), read_only=False),
        *extra_mounts,
    ]

    code, output = run_command(image, all_mounts, command, workdir)
    status = 'successfully' if code == 0 else 'failed'

    image_name = image_ref or 'linux_default'
    if output.strip():
        return_msg = (
            f'[Image: {image_name}]\n{_truncate_output(command, output)}'
            if code == 0
            else f'[Image: {image_name}]\nCommand ran {status}.\n\n......\n{_truncate_output(command, output)}'
        )
    else:
        return_msg = f'[Image: {image_name}]\nCommand ran {status}. No other output.'

    update: dict[str, Any] = {
        'messages': [
            ToolMessage(
                content=return_msg,
                tool_call_id=runtime.tool_call_id,
            )
        ],
    }
    if image_ref:
        update['known_images'] = known_images

    return Command(update=update)
