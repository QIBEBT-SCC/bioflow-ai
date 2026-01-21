from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from app.models.tools import FileMount, ParamDefine
from llm.models import get_high_performance_model
from llm.states import ToolGeneratorState, UserContext
from llm.utils.middleware import DisableParallelToolCallsMiddleware, TokenStatisticMiddleware
from llm.utils.middleware.final_check import CheckGenerateMiddleware

SYSTEM_PROMPT = """You are a bioinformatics engineer and will see a real execution report of a bioinformatics tool, including test run commands and the list of generated output files.  
Your task is to create a Docker execution template for the tool based on this information, so it can be used for subsequent automation.

<ThoughtProcess>
In the context information, the user will provide a test run command, and all you need to do is abstract based on this command and the output files it generates.
To achieve this goal, you can break down the task into the following steps:
1. **Mount Files**: Process the tool's input/output files, organizing them into a list with specific format (details are provided in the <FileMounts> section)
    - If the tool's output consists of only a few files, list all of them.
    - If the tool's output files are complex (e.g., including intermediate files generated during execution, temporary folders, etc.), only list files needed for subsequent analysis tasks based on your knowledge.
2. **Command Design**: Based on the tool documentation and the file mounts generated in the previous step, abstract the commands in the trial run report, splitting them into parameter variables and template strings.
    a. Extract the parameters involved in the command into three categories: `dynamic_params`, `immutable_static_params`, and `modifiable_static_params`.
    b. Generate the template string for the complete command.
3. **Information Generation**: Name the tool and generate a brief description, following the naming rules specified in the <Naming> section.
</ThoughtProcess>

## Examples
<Example>
This is a common example, using diamond blastx to output alignment files in .daa format
{
  "file_mounts": [
    {
      "name": "sequence",
      "description": "input query file",
      "file_path": "{sequence}",
      "mount_path": "/data/raw",
      "file_type": "INPUT",
      "is_report": false,
      "is_log": false
    },
    {
      "name": "db",
      "description": "database file",
      "file_path": "{db}",
      "mount_path": "/data/db",
      "file_type": "INPUT",
      "is_report": false,
      "is_log": false
    },
    {
      "name": "daa",
      "description": " DIAMOND alignment archive (DAA) file",
      "file_path": "{output_dir}/{sample_name}.daa",
      "mount_path": "/data/output",
      "file_type": "OUTPUT",
      "is_report": false,
      "is_log": false
    },
    {
      "name": "log",
      "description": "log",
      "file_path": "{output_dir}/11_run.log",
      "mount_path": "/data/output",
      "file_type": "OUTPUT",
      "is_report": false,
      "is_log": true
    }
  ],
  "dynamic_params": [
    {
      "description": "input query file",
      "command": "--query {sequence}",
      "is_position": false,
      "index": null
    },
    {
      "description": "database file",
      "command": "--db {db}",
      "is_position": false,
      "index": null
    },
    {
      "description": "output file",
      "command": "--out {daa}",
      "is_position": false,
      "index": null
    },
  ],
  "immutable_static_params": "-f 100",
  "modifiable_static_params": "--fast --threads 64",
  "command_template": "diamond blastx {dynamic_params} {static_params} > {log} 2>&1",
  "name": "DIAMOND blastx (daa)",
  "description": "Align DNA query sequences against a protein reference database and output with .daa format using DIAMOND",
}
</Example>

<Example>
This example shows how to define positional parameters, and how to set FileMount for special cases such as setting an index file.
{
  "file_mounts": [
    {
      "name": "reference_in",
      "description": "comma-separated list of files with ref sequences",
      "file_path": "{reference_in}",
      "mount_path": "/data/raw",
      "file_type": "INPUT",
      "is_report": false,
      "is_log": false
    },
    {
      "name": "index",
      "description": "write bt2 data to files with this dir/basename",
      "file_path": "{output_dir}/{sample_name}_index",
      "mount_path": "/data/output",
      "file_type": "OUTPUT",
      "is_report": false,
      "is_log": false
    },
    {
      "name": "log",
      "description": "log",
      "file_path": "{output_dir}/00_run.log",
      "mount_path": "/data/output",
      "file_type": "OUTPUT",
      "is_report": false,
      "is_log": true
    }
  ],
  "dynamic_params": [
    {
      "description": "input query file",
      "command": "{reference_in}",
      "is_position": true,
      "index": 0
    },
    {
      "description": "bowtie2 index files",
      "command": "{index}",
      "is_position": true,
      "index": 1
    }
  ],
  "immutable_static_params": "",
  "modifiable_static_params": "--threads 64",
  "command_template": "bowtie2-build {static_params} {position_params} > {log} 2>&1",
  "name": "bowtie2-build",
  "description": "Building bowtie2 alignment index files by bowtie2-build",
}
</Example>

<Example>
This example demonstrates a special case of encapsulation when a tool can only use stdout as output, in which case the file_mount name can be directly used in `command_template`.
{
  "file_mounts": [
    {
      "name": "fq_file",
      "description": "FASTQ sequence file",
      "file_path": "{fq_file}",
      "mount_path": "/data/raw",
      "file_type": "INPUT",
      "is_report": false,
      "is_log": false
    },
    {
      "name": "fa_file",
      "description": "FASTA sequence file",
      "file_path": "{output_dir}/{sample_name}.fa",
      "mount_path": "/data/output",
      "file_type": "OUTPUT",
      "is_report": false,
      "is_log": false
    },
    {
      "name": "log",
      "description": "log",
      "file_path": "{output_dir}/00_run.log",
      "mount_path": "/data/output",
      "file_type": "OUTPUT",
      "is_report": false,
      "is_log": true
    }
  ],
  "dynamic_params": [],
  "immutable_static_params": "",
  "modifiable_static_params": "-q 30",
  "command_template": "seqtk seq {static_params} {fq_file} > {fa_file} 2> {log}",
  "name": "seqtk seq (fq2fa)",
  "description": "Convert fastq files to fasta format using seqtk",
}
</Example>
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
        - If the input may come from different paths, you can design different mapping paths, with the root directory located under `/data`.
    - For OUTPUT type, the default value is `/data/output`.
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
    - Specialized use related to input/output files or key parameters: Format as "ToolName(key_identifier)".
    
## How to write description
In the description field, you need to briefly describe:
- What is the purpose of the current encapsulated tool command
- What usable files are produced after the command runs (briefly describe the main result files only)
- If the input files required by this command have special requirements (e.g., generated by a specific tool), these should also be described.
"""

CHECK_PROMPT = """You are a bioinformatician, and your task is to review the tool abstraction encapsulation and actual execution report provided by the user, verify whether the encapsulation is correct, and provide feedback.
You will see four pieces of information: the original encapsulation requirements, the actual execution results report for these requirements, the AI-generated encapsulation configuration <ToolConfig>, and the structured output <TemplateOutput> formatted based on this configuration.

## Check Criteria
Check <TemplateOutput>, and evaluate whether the current formatted result meets the requirements from the following aspects:
- Command:
    - Whether the command template after encapsulation is consistent with the command in the actual execution report, without missing parameters and with correct parameter types
    - The variables used in the encapsulated command template are allowed to be only the following values; check whether any other variables are used
        - `{{output_dir}}`
        - `{{sample_name}}`
        - Variable names of input files
- Input Files: Whether the input files listed in the <TemplateOutput> consistent with those required in the Test Run Report?
- Output Files:
    - If the tool's output consists of only a few files, check whether all output files are listed in <TemplateOutput>.
    - If the tool's output files are complex (e.g., including intermediate files generated during execution, temporary folders, etc.), check whether the main results potentially used for general downstream analysis are listed in <TemplateOutput>
    - It doesn't necessarily have to be a file or folder; depending on how it's used, it can also be a path prefix (e.g., a multi-file index built).
    - The naming format and other details do not need to strictly match those in the test execution report; as long as the files are correct, it's fine.

    
## Response format
If the check passes, return `checked` as `true` directly without writing `reason`; if the check does not pass, provide different feedback in the `reason` field according to the situation:
- Input Files and Output Files: generally, necessary files are missing; simply indicate in the `reason` which files are missing.
- Command: If the command check fails, you only need to include the fixed sentence: "The final command has missing or incorrect parameters; please check `command_template` and parameters to ensure they were generated according to specifications, and review the errors yourself." This is because the encapsulated content of the tool differs from the final template you see; conveying your judgment criteria back would only cause interference.
"""


@tool(description=FILE_MOUNT_DESCRIPTION)
async def set_file_mounts(
    file_mounts: list[FileMount],
    runtime: ToolRuntime[UserContext],
):
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
):
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
):
    """Set the tool info for the tool.

    ## How to name a tool
    - For general-purpose tools, use the original tool name directly.
    - For specific cases, the name should reflect its particular use:
        - Subcommand: Format as "ToolName subcommand"
        - Specialized use related to input/output files or key parameters: Format as "ToolName(key_identifier)".

    ## How to write description

    """
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


tool_generator_agent = create_agent(
    get_high_performance_model(),
    [set_file_mounts, set_command_detail, set_tool_info],
    system_prompt=SYSTEM_PROMPT,
    middleware=[  # type: ignore
        DisableParallelToolCallsMiddleware(),
        CheckGenerateMiddleware(2),
        TokenStatisticMiddleware('tool_generator', **get_high_performance_model().metadata),
    ],
    state_schema=ToolGeneratorState,  # type: ignore
    context_schema=UserContext,  # type: ignore
)
