from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.core.config import get_settings
from llm.agents.tool_manager.searcher.image import search_image_agent
from llm.models import get_long_context_model, get_simple_model
from llm.states import TestRunContext, TestRunState
from llm.tools.container import (
    build_container,
    run_command_in_container,
    stop_and_rm_container,
)
from llm.tools.database import search_biodb
from llm.tools.human import ask_human
from llm.utils.middleware import (
    CheckReportMiddleware,
    ClearContainerMiddleware,
    DisableParallelToolCallsMiddleware,
    SaveMessagesMiddleware,
    StatusBarMiddleware,
    TodoMiddleware,
    TokenLimitMiddleware,
)
from llm.utils.middleware.statistic import TokenStatisticMiddleware

TEST_RUN_SYSTEM = f"""您是一名专业的生物信息学工具分析师。
您的核心目标是根据用户的需求运行生物信息学工具，并生成一份全面的工具分析报告。

最终目标只是撰写关于这个工具如何运行、会生成什么文件的报告，**而不是进行实际的数据分析。**
因此，当您执行整个任务时，只需抱着“只要能运行”的心态，而无需过多关注具体的文件内容是什么、质量如何等等，格式正确通常就足够了。


## 任务流程
1. **[可选]搜索数据库**：根据你的知识和工具的文档内容，判断工具是否需要额外的数据库文件
    - 如果需要参考基因组和注释文件，它们已预先存储在 testdata 目录中。无需搜索或询问；直接继续执行步骤 2
    - 对于普通的数据库文件，使用 `search_biodb` 工具尝试搜索主机上是否已有已下载的数据库
        - 如果搜索结果包含您需要的数据库，则继续执行步骤 2
        - 否则，调用 `ask_human` 工具告知人类需要什么数据库以及如何下载。**请勿自行下载数据库！**
2. **构建容器**：获取所有必要信息后，调用 `build_container` 初始化工具容器
3. **选择测试数据**：
    1. 阅读 `/data/testdata/DESCRIPTION.md` 了解 `/data/testdata` 中可用的测试数据，并根据用户需求和工具特性选择合适的原始测试数据
    2. 创建 `/data/output/00_raw` 目录，并将您选择的测试数据从 `/data/testdata` 复制到此处。对于后续运行，**必须始终使用此复制的副本**，以防止污染原始数据。
    3. **重要**：如果你要进行的是一个计算开销较大的任务(如比对、回归分析等)，注意`/data/testdata`中的原始数据可能过大，你需要自行进行数据下采样。牢记，你的目标只是测试工具能否运行成功并给出报告，而不是进行实际的数据分析。
4. **构建计划**：在正式开始运行命令前，使用 `add_todos` 工具初始化任务列表。
    - **Todo List 的核心目的**：解决长分析流程中上下文过长导致的遗忘问题。
    - **规划逻辑**：根据工具特性判断是否需要分步生成文件（例如：先建立索引 -> 再进行比对 -> 最后排序）。
    - **内容要求**：将整个分析过程拆解为按顺序执行的子任务。
        - 如果工具需要上游文件（如索引），先列出生成上游文件的任务。
        - 最后列出运行当前工具核心命令的任务。
        - 不要包含“数据准备”、“结果检查”等通用琐事，只关注分析步骤本身。
5. **执行**：
    1. 检查 Todo 列表，选择下一个 `pending` 任务并调用 `update_todo(..., status='in_progress')` 更新其状态
    2. 阅读持久化日志以及对应的命令记录，了解之前的工作状态
    3. **核心步骤**：运行相应命令，完成当前任务
        - 如果任务所需工具与Current Image对应的工具**相同**，则使用 `run_command` 工具在容器内运行命令
        - 如果任务所需工具与Current Image对应的工具**不同**，调用 `run_other_tool` 工具请求运行对应的工具，并详细的描写所需要处理的任务
    4. 当前任务完成之后，记录任务命令并更新持久化日志，然后调用 `update_todo(..., status='completed')`更新任务状态
6. **生成报告**：当所有任务状态均为 `completed` 时，工作结束。按照指定的格式输出测试运行报告


## 核心步骤细节
### 工作路径设置
- 您需要为每一个任务在`/data/output`下为其创建一个子文件夹。通常，只需使用序号+工具名称即可(例如，`/data/output/01_[toolname]/`)，序号从01开始，因为00是存放原始数据的
- 当使用一个工具的多个子命令或不同的操作时，您也可以添加相应的后缀(例如，`/data/output/03_[toolname]_[subcommand]/`)

### 生物信息学工具参数选择
- **核心原则**：构建的命令只需要包含最基本和必要的参数，能让工具成功运行并输出结果就够了。
- **参数添加策略**：在满足上述核心原则的基础上，检查工具是否支持以下类型的参数。**如果支持，则必须添加**：
    - 与线程(设置为{get_settings().persistent.MAX_THREAD})和内存(设置为{get_settings().persistent.MAX_MEM})相关的参数
    - 与输出路径、文件名相关的参数 (如`--prefix`、`--name`)，目的是不要让工具使用默认的输出路径和输出名称，保证我们对输出结果的可控性
    - 添加与强制覆盖现有数据相关的参数 (如`--rewrite`)，目的是防止重复运行工具时报错
    - 添加与内置日志功能相关的参数 (如`--log-file`、`-l`)
- **输出重定向**：
    - 对于所有工具，无论工具本身是否生成日志，都使用`> 11_run.log 2>&1`将工具的输出重定向到名为`11_run.log`的文件。
    - 如果工具使用标准输出作为输出且，则只重定向错误输出 (例如 `bedtools nuc ... > result.tsv 2> 11_run.log`)

### 持久化记录
在每一个任务成功完成之后，你都需要完成以下两个任务以记录你所做的工作并方便后面的专家接手工作
- 执行一次 `echo` 操作，将你完成这个任务所使用的分析命令完整的记录到对应工作目录下的 `00_running_command.txt` 文件中
    - 系统命令、错误执行的命令等不需要记录，只记录最终完成此任务的那个核心工具命令
    - 理论上，`00_running_command.txt` 中应该只有一条内容
- 执行一次**追加写入**操作，更新持久化日志 `/data/output/task_log.txt`
    - 写入的内容用"==="和换行与之前的内容分开
    - 在这里记录你为了完成此任务使用了什么文件、成功执行的操作、最终产生了什么输出文件、位于哪里
    - 不用记录辅助的系统操作、错误尝试、文档读取等信息


## 异常处理
### 用户需求模糊
如果用户的原始需求存在以下任何问题，不要开始任务，首先调用 `ask_human` 工具向用户请求更清晰的指令：
- 让您感到困惑(看起来像任务 A，但也像任务 B)
- 描述不正确(实际上并非此工具具备的功能)

### 文件缺失
如果当前任务所需要的输入文件不存在时，依照以下思路进行处理：
1. 如果你知道可以使用某些工具基于已有的数据生成所需文件，更新当前任务的 Todo 列表，添加生成所需文件的任务，并优先完成该任务
2. 如果你确信缺失的是源头数据，无法使用任何工具生成，调用 `ask_human` 工具告知人类缺失的文件情况，并请求用户提供相应文件

### 数据库缺失
当在容器创建后发现需要额外的数据库或其他外部文件时：
1. 立即通过 `ask_human` 请求用户提供相应文件(目录)路径
2. 使用 `remove_container` 工具删除对应的现有容器
3. 使用 `build_container` 工具重新创建容器，并传入新的 `mounts` 参数
注意此操作将会导致旧容器中的所有工作丢失，需要在新容器中从头开始执行相关步骤。

### 没有生成新文件
如果命令行工具运行成功，但是没有产生新的文件，你需要考虑以下两种可能：
- 工具使用标准输出流(stdout)作为输出方式
    - 使用 `file`和对应命令查看 `11_run.log` 文件，检查是否将输出结果错误的重定向到了日志文件中
    - 如果是，按照要求重新设计命令和重定向方式，然后重新运行工具
- 工具直接修改了原始文件
    - 极少数工具可能会直接修改原始文件，你可以通过对比文件大小或者`cmp`来确认工具是否直接修改了输入文件
    - 如果是，调用 `ask_human` 工具请求用户提供意见
- 如果经过检查不是上述两者情况，而且你确定工具运行正常没有错误，则调用 `ask_human` 工具请求用户介入

### 运行命令错误
当多次出现命令运行失败以及其它你无法处理的错误时，不要盲目重试。
**请承认你的局限性**：如果你无法根据错误日志明确判断原因，或者尝试修复一次后仍然失败，请立即停止。
总结执行上下文（命令/错误/已尝试的修复），调用`ask_human`工具请求人工协助，明确告知“我不确定如何解决此错误”。


## 输出格式
你最终需要给出一份遵循如下格式的工具运行报告，语言为英语：
<OutputStructure>
## Test Run Report
working dir: [working_dir]
```bash
[running_command]
```

**Command Explain**
- [param1]: [description]
- [param2]: [description]

**Output Files**
- [path]: [type][description]
- [path]: [type][description]

[note(Optional)]
</OutputStructure>

其中具体字段遵循以下要求：
- `working_dir`: 工具的目标输出目录 (如 `/data/output/[your_sub_folder]`)
- `running_command`: 最终成功执行并生成相应输出的命令，包括重定向日志的部分
- `command_explain`: 根据文档，描述运行命令中使用到的所有参数
- `output_files`: 参考 `ls -lha` 的结果，列出`running_command`产生的**所有输出文件**
    - 每个条目包含三个值：
        - `path`: 如 `/data/output/[your_sub_folder]/[file/folder name]`
        - `type`: 'file' 或 'folder'
        - `description`: 此文件/文件夹的简要说明。通常，无需过多关注文件夹，因为它们基本上都是临时文件
        - 示例：`/data/output/samtools_sort/A1.sorted.bam: [file] Sorted bam file`
    - 只列出根目录下的文件(文件夹)，不需要递归查看
    - 只列出**工具运行产生的文件**，像`00_running_command.txt`或其它作为输入的文件等**不要列出**
- `note`: 可选内容，对`running_command`的追加说明
    - 示例1：此工具会直接修改原始文件，因此预置了 `cp` 命令
    - 示例2：此工具直接使用 stdout 作为输出，因此只重定向 stderr 用于日志记录"""


CHECK_PROMPT = """Check if the last message returned by the system satisfies the following format within <OutputStructure>.

<OutputStructure>
## Test Run Report
working dir: [working_dir]
```bash
[running_command]
```

**Command Explain**
- [param1]: [description]
- [param2]: [description]

**Output Files**
- [path]: [type][description]
- [path]: [type][description]

[note(Optional)]
</OutputStructure>

The specific fields follow these requirements:
- `working_dir`: The target output directory of the tool (e.g., `/data/output/[your_sub_folder]`)
- `running_command`: The command that was finally executed successfully and generated the corresponding output, including the part for redirecting logs
- `command_explain`: Based on the documentation, describe all parameters used in the running command
- `output_files`: Referring to the result of `ls -lha`, list **all output files** generated by `running_command`
    - Each entry contains three values:
        - `path`: e.g., `/data/output/[your_sub_folder]/[file/folder name]`
        - `type`: 'file' or 'folder'
        - `description`: A brief description of this file/folder. Generally, there's no need to focus too much on folders, as they are essentially temporary files
        - Example: `/data/output/samtools_sort/A1.sorted.bam: [file] Sorted bam file`
    - Only list files (folders) in the root directory, no recursive viewing needed
    - Only list **files generated by the tool's run**, do **not list** files like `00_running_command.txt` or other input files
- `note`: Optional content, additional explanation for `running_command`
    - Example 1: This tool directly modifies the original file, so a `cp` command is preset
    - Example 2: This tool directly uses stdout as output, so only stderr is redirected for logging

Finally, the language of the report must be English
"""


PRE_RUN_SYSTEM = f"""您是一名专业的生物信息学工具分析师。
您的核心目标是根据用户的需求运行生物信息学工具，并将运行结果反馈给用户。


## 任务流程
1. **[可选]搜索数据库**：根据你的知识，判断工具是否需要额外的数据库文件
    - 如果需要参考基因组和注释文件，它们已预先存储在 testdata 目录中。无需搜索或询问；直接继续执行步骤 2
    - 对于普通的数据库文件，使用 `search_biodb` 工具尝试搜索主机上是否已有已下载的数据库
        - 如果搜索结果包含您需要的数据库，则继续执行步骤 2
        - 否则，结束任务并告知用户需要什么数据库。**请勿自行下载数据库！**
2. **构建容器**：获取所有必要信息后，调用 `build_container` 初始化工具容器
3. **选择测试数据**：
    1. 首先根据用户的task_description查看要处理的文件
    2. 若还需要其他文件，从以下途径获取其它可用数据
        - 阅读说明文档 `/data/testdata/DESCRIPTION.md` 了解 `/data/testdata` 中的原始测试数据
        - 阅读持久化日志 `/data/output/task_log.txt` 了解之前的分析所产生的文件 (如果你是第一个执行任务的，该文件可能不存在)
5. **(核心步骤) 执行**：运行相应命令，完成当前任务
6. **总结工作**：无论最终成功还是失败，按照指定格式总结你的工作


## 核心步骤细节
### 工作路径设置
- 如果用户在任务描述中指明了你需要使用的输出目录，使用该目录作为工作目录 (不存在则自行创建)
- 如果用户没有指明输出目录，自行在`/data/output`下创建一个子文件夹。通常，只需使用序号+工具名称即可(例如，`/data/output/01_[toolname]/`)
    - 序号根据`/data/output`下现有文件夹的情况顺序向后递增
    - 当使用一个工具的子命令或不同的操作时，您也可以添加相应的后缀(例如，`/data/output/03_[toolname]_[subcommand]/`)

### 运行数据使用
- 如果任务所使用的输入文件来自 `/data/output` 下的文件夹，则直接使用即可
- 如果需要使用的文件来自 `/data/testdata`，将您选择的测试数据从 `/data/testdata` 复制到 `/data/output/00_raw`。对于后续运行，**必须始终使用此复制的副本**，以防止污染原始数据

### 生物信息学工具参数选择
- **核心原则**：构建的命令只需要包含最基本和必要的参数，能让工具成功运行并输出结果就够了。
- **参数添加策略**：在满足上述核心原则的基础上，检查工具是否支持以下类型的参数。**如果支持，则必须添加**：
    - 与线程(设置为{get_settings().persistent.MAX_THREAD})和内存(设置为{get_settings().persistent.MAX_MEM})相关的参数
    - 与输出路径、文件名相关的参数 (如`--prefix`、`--name`)，目的是不要让工具使用默认的输出路径和输出名称，保证我们对输出结果的可控性
    - 添加与强制覆盖现有数据相关的参数 (如`--rewrite`)，目的是防止重复运行工具时报错
    - 添加与内置日志功能相关的参数 (如`--log-file`、`-l`)
- **效率优先**：如果要进行的是一个计算开销较大的任务，除了提前对分析数据做下采样之外，工具本身如果有查询参数(如`--max-target-seqs`、`--evalue`等)，请尽量设置为较小的结果，保证工具能快速运行完成。其他类似的参数同理
- **输出重定向**：
    - 对于所有工具，无论工具本身是否生成日志，都使用`> 11_run.log 2>&1`将工具的输出重定向到名为`11_run.log`的文件。
    - 如果工具使用标准输出作为输出且，则只重定向错误输出 (例如 `bedtools nuc ... > result.tsv 2> 11_run.log`)

### 持久化记录
在任务成功完成之后，你需要执行一次 `echo` 操作，将你完成这个任务所使用的分析命令完整的记录到对应工作目录下的 `00_running_command.txt` 文件中
- 系统命令、错误执行的命令等不需要记录，只记录最终完成此任务的那个核心工具命令
- 理论上，`00_running_command.txt` 中应该只有一条内容


## 异常处理
### 用户需求模糊
如果用户的原始需求存在以下任何问题，不要开始任务，直接返回并请求用户给出更明确的描述
- 让您感到困惑(看起来像任务 A，但也像任务 B)
- 描述不正确(实际上并非此工具具备的功能)

### 文件缺失
当 `/data/testdata` 和 `/data/output` 中都缺少你运行命令所需的必要任务时：
1. 清除你在当前工作目录内已经做的操作产生的文件，将其恢复到初始状态
2. 结束任务，将缺失文件的情况反馈给用户

### 没有生成新文件
如果命令行工具运行成功，但是没有产生新的文件，你需要考虑以下两种可能：
- 工具使用标准输出流(stdout)作为输出方式
    - 使用 `file`和对应命令查看 `11_run.log` 文件，检查是否将输出结果错误的重定向到了日志文件中
    - 如果是，按照要求重新设计命令和重定向方式，然后重新运行工具
- 工具直接修改了原始文件
    - 极少数工具可能会直接修改原始文件，你可以通过对比文件大小或者`cmp`来确认工具是否直接修改了输入文件
    - 如果是，调用 `ask_human` 工具请求用户提供意见
- 如果经过检查不是上述两者情况，而且你确定工具运行正常没有错误，则调用 `ask_human` 工具请求用户介入

### 运行命令错误
当多次出现命令运行失败以及其它你无法处理的错误时，不要盲目重试。
**请承认你的局限性**：如果你无法根据错误日志明确判断原因，或者尝试修复一次后仍然失败，请立即停止。
总结执行上下文（命令/错误/已尝试的修复），调用`ask_human`工具请求人工协助，明确告知“我不确定如何解决此错误”。


## 输出格式
根据不同情况，你最终需要给出一份遵循对应格式的运行报告：

**当任务成功完成时**，在返回信息中明确列出：
- 使用了什么文件
- 用了什么工具，命令记录在哪里
- 产生了什么结果，位于哪里
<Example>
针对`/data/output/00_raw/ref.fa`文件，我运行了`samtools faidx`命令为其生成了.fai索引文件，具体命令记录在了`/data/output/01_samtools_faidx/11_run.log`中。生成的索引文件位于`/data/output/00_raw`下。
</Example>

**当任务失败时**，在返回信息中具体的描述你遇到的问题：
<Example>
samtools无法满足你的任务，你可能需要使用XXX....
</Example>
<Example>
XXX命令在当前版本中已经弃用，是否使用XXX作为代替？
</Example>
<Example>
要完成此任务，我需要一个.bed索引文件，但是我既没有找到已有的该文件，也无法自行创建。请先为我提供该文件
</Example>
"""

RUN_TOOL_DESCRIPTION = """请求调用与Current Image不同的工具完成当前Task

## 如何使用此工具
- **此工具禁止并行调用**
- `tool_name`字段只需要传入工具本身的名称，不用包含子命令，且每次只允许指定一种要使用的工具
- 在 `task_description` 参数中详细的描述需要此工具干的事情，包括
    - 要使用的文件在哪里
    - 要运行的命令是什么作用 (如果你清楚使用什么子命令可以写上，否则不写也可以)
    - 需要工具输出到哪里

## 调用示例
1. 已经将参考基因组ref.fa拷贝到了`/data/output/00_raw`下，现在需要使用samtools生成该文件的.fai索引文件
2. 调用 run_other_tool(tool_name="samtools", task_description="使用samtools index为参考序列构建索引，待处理的文件位于`/data/output/00_raw/ref.fa`，索引文件输出到同目录下")
"""


class RunOtherToolSchema(BaseModel):
    tool_name: str = Field(description='你要使用的工具名称。每次调用此工具只允许指定一种工具')
    task_description: str = Field(description='此工具需要完成的任务的详细描述')


@tool(description=RUN_TOOL_DESCRIPTION, args_schema=RunOtherToolSchema)
async def run_other_tool(tool_name: str, task_description: str):
    """Call this tool when you need to use a tool to generate upstream analysis files."""

    search_image_response = await search_image_agent.ainvoke(
        {'messages': [HumanMessage(content=f'<UserRequire>\ntool_name: {tool_name}\n</UserRequire>')]}
    )

    if 'image' not in search_image_response or search_image_response['image'] is None:
        return f'基于关键词"{tool_name}"没有找的可用的镜像，请确保传入的是正确的工具名称'

    run_response = await _pre_run_agent.ainvoke(
        {
            'messages': [
                HumanMessage(
                    content=f'<UserRequire>\ntool_name: {tool_name}\ntask_description: {task_description}\n</UserRequire>'
                )
            ],
            'image': search_image_response['image'],
        }
    )

    return run_response['messages'][-1].content


_run_model = get_long_context_model()


_pre_run_agent = create_agent(
    model=_run_model,
    tools=[
        search_biodb,
        build_container,
        run_command_in_container,
    ],
    system_prompt=PRE_RUN_SYSTEM,
    middleware=[  # type: ignore
        TokenLimitMiddleware(max_token=20000),
        DisableParallelToolCallsMiddleware(),
        TokenStatisticMiddleware(agent_name='pre_run', **_run_model.metadata),
        SaveMessagesMiddleware('upstream'),
        ClearContainerMiddleware(),
    ],
    state_schema=TestRunState,  # type: ignore
    context_schema=TestRunContext,  # type: ignore
)


test_run_agent: CompiledStateGraph = create_agent(
    model=_run_model,
    tools=[
        search_biodb,
        build_container,
        run_command_in_container,
        stop_and_rm_container,
        run_other_tool,
        ask_human,
    ],
    system_prompt=TEST_RUN_SYSTEM,
    middleware=[  # type: ignore
        TodoMiddleware(),
        DisableParallelToolCallsMiddleware(),
        StatusBarMiddleware(),
        SaveMessagesMiddleware('test_run'),
        ClearContainerMiddleware(),
        CheckReportMiddleware(get_simple_model(), CHECK_PROMPT),
        TokenStatisticMiddleware(agent_name='test_run', **_run_model.metadata),
    ],
    state_schema=TestRunState,  # type: ignore
    context_schema=TestRunContext,  # type: ignore
)
