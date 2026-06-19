from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Optional, cast

from langchain.agents import AgentState, create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.tools import ImageConfig
from app.utils.container import ContainerMount
from app.utils.container import run_command as run_command_once
from llm.agents.tool_agent.report_generator import CHECK_PROMPT
from llm.agents.tool_agent.searcher.image import search_image_agent
from llm.models import get_high_performance_model, get_long_context_model
from llm.states import TestRunContext
from llm.tools.database import search_biodb
from llm.tools.human import ask_human
from llm.tools.writter import write_report
from llm.utils.middleware import (
    CheckReportMiddleware,
    DisableParallelToolCallsMiddleware,
    SaveMessagesMiddleware,
    StatusBarMiddleware,
    TodoMiddleware,
)
from llm.utils.middleware.statistic import TokenStatisticMiddleware

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata


MAX_CHAR = 3000
MAX_THREAD = get_settings().persistent.resource.MAX_THREAD
MAX_MEM = get_settings().persistent.resource.MAX_MEM
DEFAULT_LINUX_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-utils:1.1.0')


DIRECT_TEST_RUN_SYSTEM = f"""您是一名专业的生物信息学工具分析师。
您的核心目标是根据用户需求运行生物信息学命令行工具, 并生成一份全面的工具试运行报告。

最终目标只是撰写关于这个工具如何运行、会生成什么文件的报告, **而不是进行实际的数据分析**。
因此, 当您执行整个任务时, 只需抱着“只要能运行”的心态, 而无需过多关注具体文件内容和质量。


## 新执行模型
您不再需要显式创建或维护容器。请直接调用 `run_command` 运行命令:
- 当运行目标生物信息学工具时, 传入对应的 `image_url`
- 当只需要执行文件浏览、复制、下采样、查看日志等通用 Linux 操作时, 省略 `image_url`, 系统会使用默认 Linux 工具镜像
- `/data/testdata` 和 `/data/output` 由程序自动固定挂载, 不要请求或传入这两个挂载
- 只有缺少数据库或外部资源时, 才通过 `mounts` 参数增挂数据库目录


## 任务流程
1. **[可选]搜索数据库**: 判断工具是否需要额外数据库文件
    - 如果需要参考基因组和注释文件, 它们已预先存储在 `/data/testdata` 中。无需搜索或询问
    - 对于普通数据库文件, 使用 `search_biodb` 搜索主机上是否已有数据库
    - 如果没有可用数据库, 调用 `ask_human` 告知人类需要什么数据库以及如何下载。**请勿自行下载数据库**
2. **搜索镜像**: 如果需要运行当前目标工具以外的工具, 调用 `search_tool_image` 查找镜像, 然后在 `run_command` 中传入返回的 `image_url`
3. **选择测试数据**:
    1. 使用默认 Linux 镜像阅读 `/data/testdata/DESCRIPTION.md`
    2. 创建 `/data/output/00_raw`, 并将选择的测试数据从 `/data/testdata` 复制到此处。后续运行必须始终使用此副本
    3. 计算开销大的任务必须下采样, 只保留足以跑通工具的最小数据
4. **构建计划**: 在正式运行分析命令前, 使用 `add_todos` 初始化任务列表
    - 只列出分析步骤本身, 不包含“数据准备”、“结果检查”等通用琐事
    - 如果需要上游文件, 先列出生成上游文件的任务
    - 最后列出运行当前工具核心命令的任务
5. **执行**:
    1. 查看 Todo, 将下一个 `pending` 任务标记为 `in_progress`
    2. 使用 `run_command` 运行当前步骤需要的命令
    3. 每个任务成功后, 在对应工作目录写入 `00_running_command.txt`
    4. 追加更新 `/data/output/task_log.txt`
    5. 将当前 Todo 标记为 `completed`
6. **生成报告**: 所有 Todo 完成后, 调用 `write_report` 输出英文试运行报告


## 工作路径设置
- 每个分析任务在 `/data/output` 下创建一个子目录, 通常使用 `序号+工具名`, 如 `/data/output/01_samtools_sort`
- `/data/output/00_raw` 专门存放原始测试数据副本


## 参数选择
- 命令只需要包含最基本、必要且能成功运行的参数
- 如果工具支持, 必须加入:
    - 线程参数, 设置为 {MAX_THREAD}
    - 内存参数, 设置为 {MAX_MEM}
    - 输出路径、输出前缀或输出文件名参数
    - 强制覆盖已有输出的参数
    - 工具内置日志参数
- 所有工具都应把日志写入 `11_run.log`
    - 普通情况使用 `> 11_run.log 2>&1`
    - 如果工具用 stdout 输出结果, 则使用 `> result_file 2> 11_run.log`


## 输出格式
最终报告必须是英文, 且符合以下结构:
## Test Run Report
working dir: [working_dir]
```bash
[running_command]
```

**Command Explain**
- [param1]: [description]

**Output Files**
- [path]: [file/folder][description]

[note(Optional)]
"""


class DirectTestRunState(AgentState, total=False):
    """State for the direct command based test-run agent."""

    image: ImageConfig
    known_images: dict[str, str]
    tool_report: NotRequired[str]
    updated: NotRequired[bool]
    checked: NotRequired[bool]


class SearchToolImageSchema(BaseModel):
    tool_name: str = Field(description='Name of the command-line tool whose image should be searched.')
    requirement: Optional[str] = Field(
        default=None,
        description='Optional short description of the task, used to disambiguate similarly named tools.',
    )


@tool(
    description='Search for a runnable container image for a command-line bioinformatics tool.',
    args_schema=SearchToolImageSchema,
)
async def search_tool_image(
    tool_name: str,
    runtime: ToolRuntime,
    requirement: Optional[str] = None,
) -> Command:
    """Search image and expose its URL to the main agent."""
    user_require = f'tool_name: {tool_name}'
    if requirement:
        user_require += f'\nrequirement: {requirement}'

    response = await search_image_agent.ainvoke(
        {'messages': [HumanMessage(content=f'<UserRequire>\n{user_require}\n</UserRequire>')]}
    )

    image = response.get('image')
    if image is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'No available image was found for "{tool_name}".',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    image = cast('ImageConfig', image)
    image_url = image.get_image()
    known_images = dict(runtime.state.get('known_images', {}))
    known_images[tool_name] = image_url

    return Command(
        update={
            'known_images': known_images,
            'messages': [
                ToolMessage(
                    content=f'Found image for {tool_name}: {image_url}',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class RunCommandSchema(BaseModel):
    command: str = Field(description='The bash command to run in the selected image.')
    image_url: Optional[str] = Field(
        default=None,
        description='Container image URL. Omit this to use the default Linux utility image.',
    )
    workdir: Optional[str] = Field(default=None, description='Working directory inside the container.')
    mounts: list[ContainerMount] = Field(
        default_factory=list,
        description='Extra database/resource mounts only. /data/testdata and /data/output are mounted automatically.',
    )


@tool('run_command', args_schema=RunCommandSchema)
def run_direct_command(
    command: str,
    image_url: Optional[str],
    workdir: Optional[str],
    mounts: list[ContainerMount],
    runtime: ToolRuntime[TestRunContext],
) -> str:
    """Run a command in a one-shot container with fixed testdata/output mounts."""
    image = ImageConfig.from_str(image_url) if image_url else DEFAULT_LINUX_IMAGE
    all_mounts = [*_fixed_mounts(runtime), *_validate_extra_mounts(mounts)]

    code, output = run_command_once(image, all_mounts, command, workdir)
    status = 'successfully' if code == 0 else 'failed'
    output = _truncate_output(command, output)

    if output.strip():
        return f'[Image: {image.get_image()}]\nCommand ran {status}.\n\n{output}'

    return f'[Image: {image.get_image()}]\nCommand ran {status}. No other output.'


def _fixed_mounts(runtime: ToolRuntime[TestRunContext]) -> list[ContainerMount]:
    output_path = (
        Path(get_settings().persistent.storage.STORAGE_PATH).expanduser().resolve()
        / 'testrun'
        / runtime.context.folder_name
    )
    output_path.mkdir(parents=True, exist_ok=True)

    testdata_path = Path(get_settings().persistent.storage.TEST_DATA_PATH).expanduser().resolve()

    return [
        ContainerMount(target='/data/testdata', source=str(testdata_path), read_only=True),
        ContainerMount(target='/data/output', source=str(output_path), read_only=False),
    ]


def _validate_extra_mounts(mounts: list[ContainerMount]) -> list[ContainerMount]:
    for mount in mounts:
        if mount.target in {'/data/testdata', '/data/output'}:
            err_msg = f'Extra mounts cannot override fixed mount target: {mount.target}'
            raise ValueError(err_msg)
        if not mount.target.startswith('/data/'):
            err_msg = f'Extra mount target must be under /data/: {mount.target}'
            raise ValueError(err_msg)

    return mounts


def _truncate_output(command: str, output: str) -> str:
    if any(k in command for k in ('find', 'grep', 'rg', 'help', 'DESCRIPTION.md')) or len(output) <= MAX_CHAR:
        return output
    return f'{output[:1500]}\n...\n{output[-500:]}'


DIRECT_TEST_RUN_MODEL = get_long_context_model()
DIRECT_CHECK_MODEL = get_high_performance_model()
metadata = cast('LLMMetadata', DIRECT_TEST_RUN_MODEL.metadata)


direct_test_run_agent = create_agent(
    model=DIRECT_TEST_RUN_MODEL,
    tools=[
        search_biodb,
        search_tool_image,
        run_direct_command,
        ask_human,
        write_report,
    ],
    system_prompt=DIRECT_TEST_RUN_SYSTEM,
    middleware=[
        DisableParallelToolCallsMiddleware(),
        StatusBarMiddleware(),
        SaveMessagesMiddleware('direct_test_run'),
        CheckReportMiddleware(DIRECT_CHECK_MODEL, CHECK_PROMPT),
        TodoMiddleware(),
        TokenStatisticMiddleware('direct_test_run', **metadata),
    ],
    state_schema=DirectTestRunState,
    # pyrefly: ignore [bad-argument-type]
    context_schema=TestRunContext,
)
