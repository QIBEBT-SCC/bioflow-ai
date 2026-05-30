from typing import TYPE_CHECKING, cast

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from app.models.tools import FileMount, ParamDefine
from llm.models import get_high_performance_model
from llm.states import ToolGeneratorState, UserContext
from llm.utils.middleware import DisableParallelToolCallsMiddleware, SkillMiddleware, TokenStatisticMiddleware
from llm.utils.middleware.final_check import CheckGenerateMiddleware

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata

SYSTEM_PROMPT = """You are a bioinformatics engineer and will see a real execution report of a bioinformatics tool, including test run commands and the list of generated output files.
Your task is to create a Docker execution template for the tool based on this information, so it can be used for subsequent automation.

<ThoughtProcess>
In the context information, the user will provide a test run command, and all you need to do is abstract based on this command and the output files it generates.
To achieve this goal, you can break down the task into the following steps:
1. **Read Skill**: Based on the operation report, select a skill corresponding to the relevant category to view detailed cases. Do not view multiple skills -- only examine the skill corresponding to the current tool's category.
2. **Mount Files**: Process the tool's input/output files, organizing them into a list with specific format (details are provided in the <FileMounts> section)
    - If the tool's output consists of only a few files, list all of them.
    - If the tool's output files are complex (e.g., including intermediate files generated during execution, temporary folders, etc.), only list files needed for subsequent analysis tasks based on your knowledge.
3. **Command Design**: Based on the tool documentation and the file mounts generated in the previous step, abstract the commands in the trial run report, splitting them into parameter variables and template strings.
    a. Extract the parameters involved in the command into three categories: `dynamic_params`, `immutable_static_params`, and `modifiable_static_params`.
    b. Generate the template string for the complete command.
4. **Information Generation**: Name the tool and generate a brief description

All tools must be called sequentially. Calling multiple tools simultaneously is **strictly prohibited**!
</ThoughtProcess>
"""

FILE_MOUNT_DESCRIPTION = """Set the file mounts for the tool.

Based on your knowledge and the user-provided information, define a list of the software's input/output files. Each file must include:
- **name** and **description**: The name field needs to be globally unique because it will serve as the keyword for the node's output interface.
- **file_type**: 'INPUT' or 'OUTPUT'.
- **file_path**: The path of the file on the host, a template string
    - For INPUT type, a variable value same as the **name** field (e.g. if the name is `r1`, then it is `{r1}`)
    - For OUTPUT type:
        - use `{output_dir}` and `{sample_name}` to construct a dynamic path template string.
        - It doesn't necessarily have to be a file or folder; depending on how it's used, it can also be a path prefix (e.g., a multi-file index built).
- **mount_path**: The output path of the file within the Docker container
    - For INPUT type:
        - If all input files are in one path, you can directly set the input to `/data/raw`.
        - If the input may come from different paths, you can design different mapping paths, with the root directory located under `/data` (e.g. /data/raw, /data/index ...).
    - For OUTPUT type, the default value is `/data/output`.
    - This value should be a directory, not a file:
        - Correct: `/data/output`
        - Incorrect: `/data/output/run.log`
- **is_report**:
    - If the file is a **read-only** report file (such as html, png, svg, etc.), set it to true.
    - For files like csv, json, or similar tsv that are functionally reports but can be used for subsequent analysis tasks, do not set it to true.
- **is_log**: There must be one and only one file named `log` with **is_log** set to true, representing the tool's execution log.
    - If the tool itself generates logs, use the tool's native log file
    - Otherwise, use `11_run.log` as the log file
"""

COMMAND_DESCRIPTION = """Set the command details for the tool.

## How to set parameters
The parameters involved in the command are divided into the following THREE categories:
- **dynamic_params**: Variables specifically related to **file paths and file names** (e.g., input files, database paths, output prefixes) that change with every run.
    - Each element in `dynamic_params` must follow this structure:
        - **description**: A description of the parameter, based on the help documentation and your knowledge.
        - **command**: The specific parameter composition string (e.g., `-i {input_file}`).
            - If the variable is a file already defined in **file_mounts**, you MUST use the corresponding defined `name` field in the template string (e.g., `{input_file}`).
            - Otherwise, you can directly use the `{output_dir}` and `{sample_name}` variables if applicable.
        - **is_position**: True if a parameter is a positional argument
        - **index**: Only useful if `is_position` is true. Represents the order (0-based) of this parameter among all positional parameters.
- **immutable_static_params**: Fixed parameters that must not be changed by the user to ensure the tool runs correctly (e.g., specific output formats required by the pipeline).
- **modifiable_static_params**: Default configuration parameters that users can override if needed (e.g., thread count, default thresholds).

**Package all parameters used in the test run report, DO NOT omit any, and DO NOT add new ones yourself.**


## How to set command_template
You must construct a `command_template` string using specific variables:
1. **Available Variables**:
   - `{static_params}`: Combines `immutable_static_params` and `modifiable_static_params`.
   - `{dynamic_params}`: Derived from `dynamic_params` where `is_position` is false.
   - `{position_params}`: Derived from `dynamic_params` where `is_position` is true.
   - `{log}`: The log file defined in <FileMounts> (where `is_log` is true), usually used for stderr redirection (e.g. `2> {log}`).
2. **Construction Rules**:
   - The template string should generally strictly contain only the variables mentioned above (plus the tool command itself).
   - **Exception (Direct File Reference)**:
        - For tools using **stdout** for output, or for **positional file arguments** (e.g., input files), you are allowed to directly use the **name** value from `<FileMounts>` (e.g., `{input_file}`, `> {output_file}`) in the template.
        - In these cases, these files do NOT need to be defined in `dynamic_params`.
   - Ensure the order of variables matches the tool's required syntax.
"""

INFO_DESCRIPTION = """Set the tool info for the tool.

## How to name a tool
- For general-purpose tools, use the original tool name directly.
- For specific cases, the name should reflect its particular use:
    - Subcommand: Format as "ToolName subcommand"
    - Specialized use related to input/output files or key parameters: Format as "ToolName(key_identifier)"
- Examples:
    - "fastp"
    - "samtools sort"
    - "diamond blastx (daa)"
    - "seqtk seq (fq2fa)"

## How to write description
In the description field, you need to briefly describe:
- What is the purpose of the current encapsulated tool command
- What usable files are produced after the command runs (briefly describe the main result files only)
- If the input files required by this command have special requirements (e.g., generated by a specific tool), these should also be described.
"""


@tool(description=FILE_MOUNT_DESCRIPTION)
async def set_file_mounts(
    file_mounts: list[FileMount],
    runtime: ToolRuntime[UserContext],
) -> Command:
    """Set the file mounts for the tool."""
    return Command(
        update={
            'file_mounts': file_mounts,
            'messages': [
                ToolMessage(
                    content='Okay, next, please design the parameters and command template.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(description=COMMAND_DESCRIPTION)
async def set_command_detail(
    dynamic_params: list[ParamDefine],
    immutable_static_params: str,
    modifiable_static_params: str,
    command_template: str,
    runtime: ToolRuntime[UserContext],
) -> Command:
    """Set the command details for the tool."""
    if 'file_mounts' not in runtime.state:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Error: file mounts not set yet. Call the `set_file_mounts` tool to set file mounts first.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    return Command(
        update={
            'command_template': command_template,
            'dynamic_params': dynamic_params,
            'immutable_static_params': immutable_static_params,
            'modifiable_static_params': modifiable_static_params,
            'messages': [
                ToolMessage(
                    content="Great, let's complete the remaining information.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(description=INFO_DESCRIPTION)
def set_tool_info(
    name: str,
    description: str,
    runtime: ToolRuntime[UserContext],
) -> Command:
    """Set the tool info for the tool."""
    if 'command_template' not in runtime.state:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='Error: command details not set yet. Call the `set_command_detail` tool to set command details first.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    return Command(
        update={
            'name': name,
            'description': description,
            'messages': [ToolMessage(content='Tool configuration completed.', tool_call_id=runtime.tool_call_id)],
        }
    )


GENERATOR_LLM = get_high_performance_model()
metadata = cast('LLMMetadata', GENERATOR_LLM.metadata)

config_generate_agent = create_agent(
    get_high_performance_model(),
    [set_file_mounts, set_command_detail, set_tool_info],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        DisableParallelToolCallsMiddleware(disabled_tools={'set_file_mounts', 'set_command_detail', 'set_tool_info'}),
        SkillMiddleware(['MODIFY_ORIGIN_TOOLS', 'SIMPLE_TOOLS', 'STDOUT_TOOLS']),
        CheckGenerateMiddleware(2),
        TokenStatisticMiddleware('tool_generator', **metadata),
    ],
    state_schema=ToolGeneratorState,
    # pyrefly: ignore [bad-argument-type]
    context_schema=UserContext,
)
