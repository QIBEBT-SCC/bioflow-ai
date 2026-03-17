import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

from docker.types import Mount
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings
from llm.states import TestRunContext

if TYPE_CHECKING:
    from app.models.tools import ImageConfig

COMMAND_DESCRIPTION = """在容器环境中运行指定bash命令或生物信息学命令行工具并返回执行结果

## 如何使用此工具
- **此工具禁止并行调用**,当有多个命令想要执行时,必须依次调用
- 每次调用都应只执行一个核心任务。严禁使用 `&&` 或 `;` 连接多个独立的命令
- 当输出超过3000个字符的时候,输出将会被截断,并只返回给你头部和尾部的部分内容

### 运行系统命令
为了保证系统的安全性,你只允许使用下面列出来的系统命令和对应的参数:
<AvailableSystemCommands>
- `ls -lha`: 用于浏览目录结构
    - 不要在数据库目录内使用此命令。因为数据库可能包含数百个小文件,请改用 `find` 命令
- `find`: 用于按名称或模式在目录中查找文件
    - 可以与 `grep` 结合进行过滤 (例如,`find ... | grep ...` )
    - 禁止使用具有修改或执行效果的参数,例如 `-exec` 和 `-delete`
- `wc -l`: 用于获取文件行数
- `sed`: 用于按行范围读取文件内容
    - 仅允许使用 `sed -n 'start,endp' file` 形式
    - 禁止使用 `-i` 或任何修改文件的参数
    - 使用 sed 查看文件时,一次读取的行数不得超过 10 行
- `grep`: 用于在文件或其他命令的输出中搜索文本内容
    - 仅用于搜索,绝不能修改源文件
- `file`: 用于确定文件类型
- `cmp -s`: 用于快速比较两个文件是否相同; 无输出表示它们相同
- `whereis`: 用于查找可执行文件的路径
- `mkdir`: 用于在工作输出目录中创建新目录
    - 仅允许在 `/data/output` 及其子目录中创建目录
    - 可以使用 `-p` 参数进行递归创建
- `cp`: 仅用于将原始数据从 `/data/testdata` 复制到 `/data/output/00_raw`
    - 文件绝不能从 `/data/output` 复制回 `/data/testdata` 或任何其他敏感目录
    - 不允许使用删除或移动源文件的变体 (例如,`mv`命令,使用 `--remove-source-files` 等 )
</AvailableSystemCommands>

### 运行生物信息学命令行工具
当运行生物信息学命令行工具时,请遵循以下规则:
- 必须设置 `workdir` 以指明当前工作目录
- 当你需要运行的工具与Current Image不同时,应该使用 `run_other_tool` 而非此工具
"""

MAX_CHAR = 3000


class ContainerMount(BaseModel):
    target: str = Field(description='Target path in container, a suitably named subdirectory under /data/')
    source: str = Field(description='Source path on host')
    read_only: bool


@tool(parse_docstring=True)
def run_command(command: str, runtime: ToolRuntime) -> str:
    """Run command in a container and return the result.

    **This tool prohibits parallel calls! You are only allowed to run one command at a time!**

    Args:
        command: the command you need to run
    """
    image = cast('ImageConfig', runtime.state.get('image'))

    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import run_docker  # noqa: PLC0415

        code, result = run_docker(image, command, [])
    else:
        from app.utils.container.podman import run_podman  # noqa: PLC0415

        code, result = run_podman(image, command, [])

    return f'[Container: {image.get_image()}]\n{result}'


@tool('run_command', description=COMMAND_DESCRIPTION)
def run_command_in_container(command: str, runtime: ToolRuntime, workdir: Optional[str] = None) -> Command:
    """Run command, and return it`s output result."""
    if 'container_id' in runtime.state:
        container_id = runtime.state.get('container_id')
    else:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='You need to create and activate the container first before you can run commands within it.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    runtime.stream_writer({'run': command, 'dir': workdir})
    # 允许使用 cat 命令的文件白名单
    allowed_cat_files = ['DESCRIPTION.md', 'about.md', 'task_log.txt']
    if re.search(r'(?:^|[|&;])\s*cat\b', command) and not any(file in command for file in allowed_cat_files):
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='The cat command is only allowed to read the files "DESCRIPTION.md", "about.md" and "task_log.txt". To read other files, use the `sed` or `grep` commands; to write files, use the `echo` command.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    if re.search(r'\brm\b', command) and (workdir and '/data/output' not in workdir) and '/data/output' not in command:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='It is forbidden to use `rm` command outside of the /data/output path!',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    if re.search(r'\bcd\b', command):
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='The `cd` command is prohibited. To specify the working directory, please use the `workdir` parameter.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    if command.endswith('|| true'):
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Attaching `|| true` or any tricks that hide the true exit code is strictly prohibited.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import run_docker_by_id  # noqa: PLC0415

        code, out = run_docker_by_id(container_id, command, workdir)
    else:
        from app.utils.container.podman import run_podman_by_id  # noqa: PLC0415

        code, out = run_podman_by_id(container_id, command, workdir)

    run_status = 'successfully' if code == 0 else 'failed'
    if out.strip():
        return_msg = (
            _truncate_output(command, out)
            if code == 0
            else f'Command ran {run_status}.\n\n......\n{_truncate_output(command, out)}'
        )
    else:
        return_msg = f'Command ran {run_status}. No other output.'

    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=return_msg,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
def build_container(mounts: list[ContainerMount], runtime: ToolRuntime[TestRunContext]) -> Command:
    """Create a new container using tool image.

    The `/data/testdata` for storing test data and `/data/output` for tool outputs are already mounted by default, so no need to add them manually.

    Args:
        mounts: Other mounts besides testdata and output folder
    """
    output_path = (
        Path(get_settings().persistent.storage.STORAGE_PATH).expanduser().resolve()
        / 'testrun'
        / runtime.context.folder_name
    )
    output_path.mkdir(parents=True, exist_ok=True)

    image: ImageConfig = runtime.state.get('image')

    # 基础挂载配置

    test_data_path = Path(get_settings().persistent.storage.TEST_DATA_PATH).expanduser().resolve()
    for m in mounts:
        source_path = Path(m.source).expanduser().resolve()
        if source_path.is_relative_to(test_data_path):
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            content=f'你在试图挂载 `{m.source}`,但是测试数据{test_data_path}已经被默认挂载到容器的/data/testdata目录下,无需手动挂载。',
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

    mount_configs = [{'type': 'bind', 'target': m.target, 'source': m.source, 'read_only': m.read_only} for m in mounts]

    testdata_mount = {
        'type': 'bind',
        'target': '/data/testdata',
        'source': f'{test_data_path}',
        'read_only': True,
    }
    output_mount = {
        'type': 'bind',
        'target': '/data/output',
        'source': f'{output_path}',
    }

    container_id: Optional[str] = None
    match get_settings().docker.CONTAINER:
        case 'docker':
            from app.utils.container.docker import create_container  # noqa: PLC0415

            final_mounts = [
                Mount(
                    target=m['target'],
                    source=m['source'],
                    type='bind',
                    read_only=m.get('read_only', False),
                )
                for m in [*mount_configs, testdata_mount, output_mount]
            ]
            container_id = create_container(image, final_mounts)
        case 'podman':
            from app.utils.container.podman import create_container  # noqa: PLC0415

            # Podman specific relabeling
            for m in mount_configs:
                m['relabel'] = 'z'
            testdata_mount['relabel'] = 'z'

            final_mounts = [*mount_configs, testdata_mount, output_mount]
            container_id = create_container(image, final_mounts)
        case _:
            container_id = None
            logger.error('Incorrect container type')

    return Command(
        update={
            'container_id': container_id,
            'messages': [
                ToolMessage(
                    content=f'The container based on image `{image.get_image()}` has been created. Now you can use `run_command` to run command in it.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool('remove_container')
def stop_and_rm_container(runtime: ToolRuntime) -> Command:
    """Stop and remove container."""
    container_id = runtime.state.get('container_id')

    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import remove_container  # noqa: PLC0415

        remove_container(container_id)
    else:
        from app.utils.container.podman import remove_container  # noqa: PLC0415

        remove_container(container_id)

    return Command(
        update={
            'container_id': None,
            'messages': [
                ToolMessage(
                    content='The container has been removed',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _truncate_output(command: str, output: str) -> str:
    if any(k in command for k in ('find', 'grep', 'help', 'DESCRIPTION.md')) or len(output) <= MAX_CHAR:
        return output
    return f'{output[:1500]}\n...\n{output[-500:]}'
