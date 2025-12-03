import re
from pathlib import Path
from typing import Optional

from docker.types import Mount
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.tools import ImageConfig
from llm.states import TestRunContext


class ContainerMount(BaseModel):
    target: str = Field(description='Target path in container, a suitably named subdirectory under /data/')
    source: str = Field(description='Source path on host')
    read_only: bool


@tool(parse_docstring=True)
def run_command(command: str, runtime: ToolRuntime):
    """run command in a container and return the result
    **This tool prohibits parallel calls! You are only allowed to run one command at a time!**

    Args:
        command: the command you need to run
    """
    image = runtime.state.get('image')

    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import run_docker

        code, result = run_docker(image, command, [])
    else:
        from app.utils.container.podman import run_podman

        code, result = run_podman(image, command, [])

    return f'[Container: {image.get_image()}]\n{result}'


@tool(parse_docstring=True)
def build_container(mounts: list[ContainerMount], runtime: ToolRuntime[TestRunContext]):
    """Create a new container using tool image

    - The `/data/testdata` for storing test data and `/data/output` for tool outputs are already mounted by default, so you don't need to add them manually.

    Args:
        mounts: Other mounts besides testdata and output folder
    """
    output_path = Path(get_settings().persistent.STORAGE_PATH).absolute() / 'testrun' / runtime.context.folder_name
    output_path.mkdir(parents=True, exist_ok=True)

    image: ImageConfig = runtime.state.get('image')

    # 基础挂载配置
    mount_configs = [{'type': 'bind', 'target': m.target, 'source': m.source, 'read_only': m.read_only} for m in mounts]
    testdata_mount = {
        'type': 'bind',
        'target': '/data/testdata',
        'source': get_settings().persistent.TEST_DATA_PATH,
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
            from app.utils.container.docker import create_container

            final_mounts = []
            for m in mount_configs + [testdata_mount, output_mount]:
                final_mounts.append(
                    Mount(
                        target=m['target'],
                        source=m['source'],
                        type=m.get('type', 'bind'),
                        read_only=m.get('read_only', False),
                    )
                )
            container_id = create_container(image, final_mounts)
        case 'podman':
            from app.utils.container.podman import create_container

            # Podman specific relabeling
            for m in mount_configs:
                m['relabel'] = 'z'
            testdata_mount['relabel'] = 'z'

            final_mounts = mount_configs + [testdata_mount, output_mount]
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


@tool('run_command', parse_docstring=True)
def run_command_in_container(command: str, runtime: ToolRuntime, workdir: Optional[str] = None):
    """Run command, and return it`s output result.
    **This tool prohibits parallel calls! You are only allowed to run one command at a time!**

    When running system commands, follow these rules:
    <AvailableSystemCommands>
    1. `ls -lha`: Used to browse directory structures and file information.
        - Do not use this command inside the database directory. Because the database is likely to contain hundreds of small files, using the `find` command instead.
    2. `find`: Used to find files in directories by name or pattern.
        - Can be combined with `grep` for filtering (e.g., `find ... | grep ...`)
        - Parameters with modifying or executing effects such as `-exec` and `-delete` are prohibited.
        - Must not be used to execute any commands.
    3. `wc -l`: Used to get the number of lines in a file.
    4. `sed`: Used to read file content by line range (for safe viewing of partial content)
        - Only the form `sed -n 'start,endp' file` is allowed.
        - `-i` or any parameters that modify files are prohibited.
        - When using sed to view a file, the number of lines read at one time must not exceed 5.
    5. `grep`: Used to search for text content in files or find output.
        - Only for searching, must not execute or modify files.
        - Can be used to search files directly.
        - Can be combined with `find` pipes for filtering.
    6. `file`: Used to determine file type (text/binary).
    7. `cmp -s`: Used to quickly compare if two files are identical; no output means they are the same.
    8. `where`: Used to find the path of the executable file.
    9. `mkdir`: Used to create new directories in the working output directory.
        - Only allowed to create directories in `/data/output` and its subdirectories.
        - Recursive creation can be done using the `-p` parameter.
    10. `cp`: Only used to copy raw data from `/data/testdata` to `/data/output`.
        - Files must not be copied back from `/data/output` to `/data/testdata` or any other sensitive directory.
        - No variants that delete or move source files (e.g., `mv`, usage with `--remove-source-files`, etc.) are allowed.

    Except for the commands listed above, the use of any other system commands is prohibited.
    </AvailableSystemCommands>

    Args:
        command: The command you need to run
        workdir: Command execution working directory. Only pass this parameter when you need to enter a specific directory to execute a command.
    """
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
    if re.search(r'\bcat\b', command) and 'DESCRIPTION.md' not in command:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='The `cat` command is not allowed. Using the `sed` or `grep` command to view file.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import run_docker_by_id

        code, out = run_docker_by_id(container_id, command, workdir)
    else:
        from app.utils.container.podman import run_podman_by_id

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

    image: ImageConfig = runtime.state.get('image')
    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=f'[Container: {image.get_image()}]\n{return_msg}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool('remove_container')
def stop_and_rm_container(runtime: ToolRuntime):
    """Stop and remove container"""

    container_id = runtime.state.get('container_id')

    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import remove_container

        remove_container(container_id)
    else:
        from app.utils.container.podman import remove_container

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
    if any(k in command for k in ('find', 'grep', 'help')) or len(output) <= 3000:
        return output
    return f'{output[:1500]}\n...\n{output[-500:]}'
