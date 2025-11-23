from typing import Literal, Optional

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.config import get_settings
from llm.models import gemini_25_flash, gpt_5_mini
from llm.states import SearchImageState, TestRunContext, TestRunState
from llm.tools.container import (
    build_container,
    run_command_in_container,
    stop_and_rm_container,
)
from llm.tools.database import search_biodb, search_local_image, select_image
from llm.tools.human import ask_human
from llm.utils.middleware import (
    ClearContainerMiddleware,
    DisableParallelToolCallsMiddleware,
    SaveMessagesMiddleware,
    TokenLimitMiddleware,
)

TEST_RUN_SYSTEM = f"""You are a professional bioinformatics tool analyst. Your core goal is to run a bioinformatics tool on a specified dataset and generate a comprehensive tool analysis report. 
You must operate precisely, strictly adhere to established workflows and rules, and produce well-documented and reproducible results.

<Workflow>
**Our ultimate goal is simply to design a command that meets user needs, execute it successfully, and see what output it produces, rather than to perform a rigorous analysis.**
So when you're executing the overall task, you just need to go with the mindset of "as long as it runs," without paying too much attention to what the specific file content is, what the quality is like, etc. The format being correct is usually good enough.

1. **Preliminary Analysis**:
    a. **Read Document**: Analyze the documentation. Before creating any containers, systematically analyze the tool's `--help` documentation to understand its functionality, inputs, parameters, especially looking for database-related parameters (e.g., `--db`, `--database`).
    b. **Requirements Assessment**: First, evaluate the user's original demand. If the user's original demand has any of the following issues, you need to call the `ask_human` tool to request clearer instructions from the user:
        - It confuses you (seems like task A, but also like task B)
        - Incorrect description (actually not a function this tool possesses)
    c. **Dependency Analysis**: Based on the documentation, distinguish between three types of dependencies:
        - **Tool-specific Index**: This refers to an index that is built from a source file (e.g., a FASTA reference genome) using a command like `bowtie2-build`. For situations where this tool can handle it on its own, go directly to step 3.
        - **Reference File**: such as reference genomes, annotation files, variant annotations, etc., which can be obtained from `/data/testdata`. Continue to step 3
        - **Large Pre-built Database**: This refers to large, common databases that are typically downloaded manually (e.g., NCBI `nr`, GTDB). For this situation, first go to step 2.
2. **[Option]Search for Database**: When the tool needs to use additional databases, first use the `search_biodb` tool to try searching if there is already a downloaded database on the host
    - If the search results contain the database you need, proceed to step 3
    - Otherwise, immediately call the `ask_human` tool to request the corresponding database. In the request, you need to tell the human what database is needed and possible download methods
    - Note that **downloading the database must request human intervention!** Even if the tool provides scripts like download-database, it still needs to be determined by the human whether to call it
3. **Build Container**: After obtaining all necessary information, call `build_container` to initialize the tool container
    - If the tool requires an external database, based on the search results from the previous step, add the path of the external database to the `mounts` parameter as an additional mount.
    - The `/data/testdata` for storing test data and `/data/output` for tool outputs are already mounted by default, so you don't need to add them manually.
4. **Choose Test Data**: Regardless of which step of the analysis process this tool is in or what input it requires, throughout the entire process, it always needs a RAW DATA as the starting point, which could be RNA sequences, metagenomic sequences, etc.
    a. Understand the available raw data in `/data/testdata` by reading `/data/testdata/DESCRIPTION.md`. 
    b. Select appropriate raw test data based on user needs and tool characteristics.
    c. Create the `/data/output/00_raw` directory, and copy your selected test data from `/data/testdata` here. For subsequent runs, use this copied replica, and **do not directly use the files in `/data/testdata`!**
    d. If no suitable test data is found, handle according to the rules in <MissingFiles>
5. **Execution**: 
    a. According to the rules in <CommandDesignRule>, based on the documentation and user requirements, select appropriate parameters and design the command to be executed
    b. Follow the rules in <ExecutionRules>, run the appropriate command in the container, and finally obtain the output result of the tool
    c. After successful execution, process the output results based on the tool's behavior (whether it generates new files or modifies existing ones in place).
6. **Cleanup**: Before generating the final report, you must use the `remove_container` tool to clean up the container created during this task.
7. **Generate Report**: After obtaining all information, provide the final report on this tool in the required format.
</Workflow>

<MissingFiles>
You need to identify exactly what type of file you are missing, and call the `ask_human` or `need_upstream_analysis` tool based on the situation.

- Missing is raw data: Call the `ask_human` tool to request humans to supplement this type of test data
    - Example 1: You need ATAC data, but there is no such data in `/data/testdata`
    - Example 2: You need two protein sequences for differential comparison, but there is only one protein sequence in `/data/testdata`
- Missing is files obtained from processing by other bioinformatics analysis tools: Call the `need_upstream_analysis` tool to request upstream analysis using other tools to provide the corresponding files
    - Example 1: You need an aligned BAM file
    - Example 2: You need an assembled contigs sequence
    - Our core purpose is to analyze the output behavior of the target tool, rather than performing a complete and rigorous data analysis. Therefore, when using `need_upstream_analysis` to request upstream analysis, your thinking should be to obtain the input files required by the target tool in the **simplest and most direct** way. Ignore all complex, non-essential steps, such as quality control, filtering, or normalization.
</MissingFiles>

<CommandDesignRule>
- **Core Principle**: The commands you construct must only contain the most basic and essential parameters sufficient for the tool to run successfully and achieve the analysis objective.
- **Parameters NOT to be added**: Parameters that command can run successfully even without it, generally like setting thresholds or other advanced parameters
- **Parameters NEED to be added**: If the tool supports it, you should add the following four types of parameters in addition to the minimal command:
    - Add parameters related to threads (set to {get_settings().persistent.MAX_THREAD}) and memory (set to {get_settings().persistent.MAX_MEM}).
    - Add parameters related to the output file name (e.g., `--prefix`, `--name`).
    - If the tool has built-in logging functionality (e.g., `--log-file`, `-l`), it should be added to the command.
    - Some tools (e.g. samtools convert) may use stdout as default output if specific parameters are not used. In such cases, you must add the corresponding parameters to write the output directly to a file.
- **Output Redirection**: For all tools, regardless of whether the tool itself generates logs, use `> 11_running.log 2>&1` to redirect the tool's output to a file named `11_running.log`
    - If a tool uses stdout as output and does not specify a parameter to output to a file, then you need to modify the capture method to only capture error messages to the log. (e.g. `bioawk -c fastx ... input.fa > result.txt 2> 11_running.log`)
- **Handling In-Place Modifications**: Only when a tool's output directly modifies the original input file, at that time only, are you allowed to use `&&` to perform a concatenation operation, constructing a command in the form of `cp [raw_input] [copied_input] && [command with copied_input] > 11_running.log 2>&1` to prevent it from modifying the original file.
</CommandDesignRule>

<ExecutionRules>
### Directory operations
- All directories except `/data/testdata` are **strictly read-only**, and any write or possible modification operations to files within them are prohibited.
- To run the current tool, you need to create a subfolder for it under `/data/output`. Generally, just use the tool name (e.g., `/data/output/[toolname]/`)
    - When a tool uses multiple subcommands or different operations, you can also add the corresponding suffix (e.g., `/data/output/[toolname]_[subcommand]/`)
- When you call `need_upstream_analysis`, their output files will also be in the corresponding folders under `/data/output`. These folders are strictly read-only for you. If you want to use one of their files, you also need to copy it to the `/data/output/00_raw` directory

### Command Execution
- **General Rules (applicable to all commands)**:
    - Each call to `run_command` should execute only one core task. Using `&&` or `;` to link multiple independent commands is strictly prohibited.
    - Attaching `|| true` or any tricks that hide the true exit code is strictly prohibited.
    - Use the `workdir` parameter to set the working directory, not `cd` within the `command` parameter.
- **Tool Commands**: When running tool commands, you must set the `workdir` parameter to the output directory of the current tool.
- **System Commands**: You are only allowed to use the system commands listed in the tool description.
    - Unless otherwise specified, all commands are for read-only purposes and must not execute, modify, or delete any files; the only allowed write operations are limited to the `/data/output` directory and its subdirectories.
    - Unless there's an error, there's NO NEED to explore the content of INPUT files with commands like `sed`. Keep in mind that we just want to get the tool "running"; usually, if the file format is correct, you can use it directly.
- **Handling In-Place Modifications**: If a tool does not produce new files in the output directory after running, you need to consider whether it directly modified the original files.
    1. First use the `cmp -s` command to compare the files in the `/data/output/00_raw` directory with their original source files to confirm whether the file content has changed.
    2. If it is determined that the tool directly modifies the original file, then use cp to re-copy a test file to `/data/output/00_raw`
    3. Follow the rules in <CommandDesignRule> to redesign the command and run it, ultimately making the tool output to a new folder instead of modifying the source file.
- **Execution Logging**: After the tool command **executes successfully**, run a separate `echo` command to record the pure command into the `00_running_command.txt` file in the tool's output directory. The command output here should be the pure analysis command, without tool paths, redirections, or other information.
    - For system commands and tool commands that fail to run, there is no need to record them; ONLY record the one tool execution command that ultimately satisfies the user's needs.
    - Example:
        1. `run_command_in_container(command="/usr/bin/[command] [some_params] > 11_running.log 2>&1", workdir="/data/output/[command]")`
        2. `run_command_in_container(command='echo "[command] [some_params]" > 00_running_command.txt', workdir="/data/output/[command]")`
    - If it's the in-place modification case above, the record needs to add the preceding `cp` command
- **Unresolvable Execution Error**: When you attempt to execute a command that you believe is syntactically and logically correct, but it repeatedly fails. This usually means there are deep-seated problems that you cannot diagnose (such as a corrupted container environment, an unknown bug in the tool, or a critical prerequisite you overlooked). In such cases, DO NOT fall into an infinite loop of retries. Clearly state the commands you have tried, the error messages encountered, and your own analysis in your request, then ask the human to help diagnose the problem.

### Container Management
If you discover the need for an additional database or other external files *after* container creation:
1. Immediately obtain the database path via `ask_human`.
2. Use the `remove_container` tool to delete the corresponding existing container.
3. Use the `build_container` tool to recreate the container, passing in the new `mounts` parameter.
4. You should recognize that all work in the old container is lost, and you will need to perform the relevant steps from scratch in the new container.
</ExecutionRules>
"""


SEARCH_IMAGE_SYSTEM = """You are a bioinformatics data engineer. Your task is to find the corresponding tool's image based on the user's needs and return it.

According to the tool search results, if there is an image of the tool required by the user, use `select_image` to select it;
Otherwise, end the task directly and tell the user that the image does not exist.
"""

PRE_RUN_SYSTEM = f"""You are a professional bioinformatics tool analyst. Your core goal is to run a bioinformatics tool on a specified dataset and generate a comprehensive tool analysis report. 
You must operate precisely, strictly adhere to established workflows and rules, and produce well-documented and reproducible results.

<Workflow>
Our ultimate goal is simply to design a command that meets user needs, execute it successfully, and see what output it produces, rather than to perform a rigorous analysis.
So when you're executing the overall task, you just need to go with the mindset of "as long as it runs," without paying too much attention to what the specific file content is, what the quality is like, etc. The format being correct is usually good enough.

1. **[Option]Search for Database**: When the tool needs to use additional databases, first use the `search_biodb` tool to try searching if there is already a downloaded database on the host
    - If the search results contain the database you need, proceed to step 3
    - Otherwise, immediately call the `ask_human` tool to request the corresponding database. In the request, you need to tell the human what database is needed and possible download methods
    - Note that **downloading the database must request human intervention!** Even if the tool provides scripts like download-database, it still needs to be determined by the human whether to call it
2. **Build Container**: Call `build_container` to initialize the tool container
    - If the tool requires an external database, based on the search results from the previous step, add the path of the external database to the `mounts` parameter as an additional mount.
    - The `/data/testdata` for storing test data and `/data/output` for tool outputs are already mounted by default, so you don't need to add them manually.
3. **Choose Test Data**:
    a. In the previous work, we have selected appropriate test data and copied it to `/data/output/00_raw`. First, check the files in the directory to determine if they are sufficient for the current tool.
    b. If the files in `/data/output/00_raw` are sufficient for you to complete the current task, go directly to step5, otherwise continue
    c. Read the `/data/testdata/DESCRIPTION.md` file, then select the missing files and copy them to `/data/output/00_raw`
    d. If no suitable test data is found, handle according to the rules in <MissingFiles>
4. **Execution**: 
    a. First, call the command to view the tool's own help documentation (usually using the `--help` command)
    b. According to the rules in <CommandDesignRule>, based on the documentation and user requirements, select appropriate parameters and design the command to be executed
    c. Follow the rules in <ExecutionRules>, run the appropriate command in the container, and finally obtain the output result of the tool
5. **Cleanup**: Before generating the final report, you must use the `remove_container` tool to clean up the container created during this task.
6. **Generate Report**: After successfully executing the target tool, carefully generate the final "Tool Analysis Report". The content of the report must be based on the guidelines in <ReportingGuide>.
</Workflow>

<MissingFiles>
- Missing is raw data: Call the `ask_human` tool to request humans to supplement this type of test data
    - Example 1: You need ATAC data, but there is no such data in `/data/testdata`
    - Example 2: You need two protein sequences for differential comparison, but there is only one protein sequence in `/data/testdata`
- Missing is files obtained from processing by other bioinformatics analysis tools: Call the `need_upstream_analysis` tool to request upstream analysis using other tools to provide the corresponding files
    - 
</MissingFiles>

<CommandDesignRule>
- **Core Principle**: The commands you construct must only contain the most basic and essential parameters sufficient for the tool to run successfully and achieve the analysis objective.
- **Parameters NOT to be added**: Parameters that command can run successfully even without it, generally like setting thresholds or other advanced parameters
- **Parameters NEED to be added**: If the tool supports it, you should add the following four types of parameters in addition to the minimal command:
    - Add parameters related to threads (set to {get_settings().persistent.MAX_THREAD}) and memory (set to {get_settings().persistent.MAX_MEM}).
    - Add parameters related to the output file name (e.g., `--prefix`, `--name`).
    - Some tools (e.g. samtools convert) may use stdout as default output if specific parameters are not used. In such cases, you must add the corresponding parameters to write the output directly to a file.
</CommandDesignRule>

<ExecutionRules>
### Directory operations
- All directories except `/data/testdata` are **strictly read-only**, and any write or possible modification operations to files within them are prohibited.
- To run the current tool, you need to create a subfolder for it under `/data/output`. Generally, just use the tool name (e.g., `/data/output/[toolname]/`)
    - When a tool uses multiple subcommands or different operations, you can also add the corresponding suffix (e.g., `/data/output/[toolname]_[subcommand]/`)
- When you call `need_upstream_analysis`, their output files will also be in the corresponding folders under `/data/output`. Use the files among them as needed.

### Command Execution
- **General Rules (applicable to all commands)**:
    - Each call to `run_command` should execute only one core task. Using `&&` or `;` to link multiple independent commands is strictly prohibited.
    - Attaching `|| true` or any tricks that hide the true exit code is strictly prohibited.
    - Use the `workdir` parameter to set the working directory, not `cd` within the `command` parameter.
- **Tool Commands**: When running tool commands, you must set the `workdir` parameter to the output directory of the current tool.
- **System Commands**: You are only allowed to use the system commands listed in the tool description.
    - Unless otherwise specified, all commands are for read-only purposes and must not execute, modify, or delete any files; the only allowed write operations are limited to the `/data/output` directory and its subdirectories.
- **Handling In-Place Modifications**: If a tool does not produce new files in the output directory after running, you need to consider whether it directly modified the original files.
    1. First use the `cmp -s` command to compare the files in the `/data/output/00_raw` directory with their original source files to confirm whether the file content has changed.
    2. If it is determined that the tool directly modifies the original file, then use the `cp` command to copy the modified file to the output directory you originally created for the current task.

### Container Management
If you discover the need for an additional database or other external files *after* container creation:
1. Immediately obtain the database path via `ask_human`.
2. Use the `remove_container` tool to delete the corresponding existing container.
3. Use the `build_container` tool to recreate the container, passing in the new `mounts` parameter.
4. You should recognize that all work in the old container is lost, and you will need to perform the relevant steps from scratch in the new container.
</ExecutionRules>

<SystemCommandsRule>
You can only use the following commands to browse and manage local files and directories. Unless otherwise specified, all commands are for read-only purposes and must not execute, modify, or delete any files; 
the only allowed write operations are limited to the `/data/output` directory and its subdirectories.

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

Except for the commands listed above, the use of any other commands is prohibited.
</SystemCommandsRule>

<HumanAssistancePolicy>
In the following three situations, you must immediately stop all current operations and use the `ask_human` tool to request human intervention:

- **Missing Raw Data**: When a type of raw data required for analysis (e.g., a type of omics sequencing data, a certain image, etc.) is not available in `/data/testdata`. Do not request intermediate files that you can generate yourself.
- **External Database or Missing Source File Required**:
    - When a tool requires a **large, pre-built database** (e.g., NCBI `nr`, GTDB), you must ask the human for its host path.
    - When a tool requires an **index** to be built, you should first try to build it assuming the source file (e.g., reference genome) is in `/data/testdata`. If, and only if, the build fails because the source file is missing, you should then use `ask_human` to request the necessary source file.
- **Unresolvable Execution Error**: When you attempt to execute a command that you believe is syntactically and logically correct, but it repeatedly fails. This usually means there are deep-seated problems that you cannot diagnose (such as a corrupted container environment, an unknown bug in the tool, or a critical prerequisite you overlooked). In such cases, do not fall into an infinite loop of retries. Clearly state the commands you have tried, the error messages encountered, and your own analysis in your request, then ask the human to help diagnose the problem.
</HumanAssistancePolicy>

<ReportingGuide>
If you ultimately succeed in completing the goal, you need to tell the user what you did and where the files he needs are located. In the form of:
```text
I generated the bam file you need through the command `[omitted]`, located in the `/data/output/[omitted]` directory
```

And if your task fails, similarly describe what situation you encountered and provide feedback to the user. For example:
```text
samtools seems unable to meet your requirements, please switch to a more suitable tool
```
</ReportingGuide>
"""


@tool(parse_docstring=True)
async def need_upstream_analysis(
    tool_name: str, file_description: str, reasoning: str, runtime: ToolRuntime[TestRunContext]
):
    """Use this tool to delegate a task to a DIFFERENT upstream agent when a required input file is MISSING.

    DO NOT output multiple requests in parallel.
    You must wait for the first file to be generated before requesting the next one (e.g., wait for Index generation before requesting Alignment).

    Args:
        tool_name: The name of the upstream bioinformatics tool to be called.
        file_description: A specific description of the output file required from that tool.
        reasoning: The logical justification for requesting this dependency.
    """
    agent_input: SearchImageState = {
        'messages': [
            HumanMessage(
                content=f'<UserRequire>\ntool_name: {tool_name}\nrequired_file: {file_description}\n</UserRequire>'
            )
        ]
    }
    _response = await _pre_run_agent.ainvoke(agent_input)

    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=_response['messages'][-1].content,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


class PreRunAgent:
    def __init__(self):
        self.search_image_agent = create_agent(
            model=gpt_5_mini(temperature=0.3),
            tools=[
                search_local_image,
                select_image,
            ],
            system_prompt=SEARCH_IMAGE_SYSTEM,
            middleware=[DisableParallelToolCallsMiddleware()],
            state_schema=SearchImageState,
        )
        self.pre_run_agent = create_agent(
            model=gemini_25_flash(temperature=0.3),
            tools=[
                search_biodb,
                build_container,
                run_command_in_container,
                stop_and_rm_container,
                need_upstream_analysis,
            ],
            system_prompt=PRE_RUN_SYSTEM,
            middleware=[
                TokenLimitMiddleware(),
                DisableParallelToolCallsMiddleware(),
                SaveMessagesMiddleware(),
                ClearContainerMiddleware(),
            ],
            state_schema=TestRunState,
            context_schema=TestRunContext,
        )

    async def search_image_node(self, state: TestRunState) -> Command[Literal['pre_run', '__end__']]:
        """先搜索镜像"""
        response = await self.search_image_agent.ainvoke({'messages': state['messages']})

        if 'image' in response and response['image']:
            return Command(goto='pre_run', update={'image': response['image']})

        return Command(
            goto=END,
            update={'messages': [response['messages'][-1]]},
        )

    async def pre_run_node(self, state: TestRunState, runtime: Runtime[TestRunContext]):
        """运行工具"""
        test_run_state: TestRunState = {
            'messages': state['messages'],
            'image': state['image'],
        }

        response = await self.pre_run_agent.ainvoke(
            test_run_state,
            context=TestRunContext(
                folder_name=runtime.context.test_run_id,
                save_message=runtime.context.save_message,
                test_run_id=runtime.context.test_run_id,
            ),
        )

        return {'messages': [response['messages'][-1]]}

    def build(self) -> CompiledStateGraph[TestRunState, TestRunContext, SearchImageState]:
        """构建并编译预运行 agent 图"""
        graph = StateGraph(TestRunState, TestRunContext, input_schema=SearchImageState)
        graph.add_node('search_image', self.search_image_node)
        graph.add_node('pre_run', self.pre_run_node)

        graph.set_entry_point('search_image')
        graph.add_edge('pre_run', END)

        compiled_graph = graph.compile(name='pre_run_agent')
        return compiled_graph


class FileItem(BaseModel):
    path: str = Field(description='e.g. `/data/output/[your sub folder]/[file/folder name]')
    type: Literal['file', 'folder']
    description: str = Field(
        description="A brief description of what this file/folder is. Generally, there's no need to pay too much attention to folders, as they are likely temporary files."
    )


class TestRunReport(BaseModel):
    running_command: str = Field(description='The actual command that produced this output')
    output_path: str = Field(description='e.g. `/data/output/[your sub folder]`')
    note: Optional[str] = Field(
        description='For example, this tool will directly modify the original file, so you need to add cp to ensure the result is output to the target directory',
        default=None,
    )

    output_files: list[FileItem] = Field(
        description='All files and folders (non-recursive) generated by the tool itself in the output directory, meaning all other files/directories except for `00_running_command.txt` and `11_running.log`.',
        default_factory=list,
    )
    auto_create_output_folder: bool = Field(
        description='Does the tool itself automatically create the output folder? If you find that the tool cannot run after manually creating the output folder, giving a message similar to "Output folder already exists", then set this option to true.',
        default=False,
    )
    auto_create_log: bool = Field(
        description='Does the tool itself generate log files (rather than us manually redirecting them)?', default=False
    )


_pre_run_agent = PreRunAgent().build()
test_run_agent: CompiledStateGraph[TestRunState, TestRunContext, TestRunState, TestRunReport] = create_agent(
    model=gemini_25_flash(temperature=0.3),
    tools=[
        build_container,
        run_command_in_container,
        stop_and_rm_container,
        need_upstream_analysis,
        ask_human,
    ],
    system_prompt=TEST_RUN_SYSTEM,
    middleware=[
        TokenLimitMiddleware(),
        DisableParallelToolCallsMiddleware(),
        SaveMessagesMiddleware(),
        ClearContainerMiddleware(),
    ],
    response_format=TestRunReport,
    state_schema=TestRunState,
    context_schema=TestRunContext,
)
