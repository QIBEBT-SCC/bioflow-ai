import operator
from typing import Annotated, TypedDict

from langchain_core.messages import ToolMessage, BaseMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
from pydantic import BaseModel

from app.core.llm_chat import gemini_25_flash
from app.models.tools import ParamDefine, FileMount


SYSTEM_PROMPT = """We are designing a node-based bioinformatics analysis platform, to achieve this goal, it is necessary to encapsulate the analysis tools as nodes for invocation.  
As a bioinformatics engineer, your task is to complete the encapsulation settings of the tools according to actual needs.

<ThoughtProcess>
To achieve this goal, you can break down the task into the following steps:
- **Command Design**: If the user provides a complete command for reference, start designing the encapsulation directly based on that command; otherwise, design the command based on the help documentation yourself，and pay attention to the following details:
    - If there are set parameters related to the number of threads and memory, use 8 threads, 16G as default parameters  
    - Need to save the tool's running logs. If the tool itself generates (or is enabled via parameters), use the tool's own; otherwise, use `&>` to direct running information to a log file
- **Output Organization**: Process and format the tool's output files, organizing them into a list that adheres to a specific format (details are provided in the <OutputFiles> section). If the user provides a reference structure for the output files, organize based on it; otherwise, use your own knowledge. Note that you do not simply list all the tool's output files. In short:
    - If the tool's output consists of only a few files and all are potentially used in subsequent analysis tasks, retain all of them.
    - If the tool's output files are complex (e.g., including intermediate files generated during execution, temporary folders, etc.), extract only the files needed for subsequent analysis tasks based on your knowledge.
- **Command Splitting**: Combine the previously generated commands, summarized output files, and tool help documents to classify the instructions into `file_params` and `static_params`:
    - `file_params`: Parameters related to external file input and tool file output, i.e., the parameters corresponding to the input/output interfaces in the nodes. This is easy to understand, as node-based editors use connections to represent the flow of files (data). Each parameter is a specific data structure, detailed in the following <ParamDefine> section.
    - `static_params`: The remaining parameters, treated as a single string. Specifically, output-related parameters like setting prefixes and filenames, which do not correspond to specific files, should also be categorized under `static_params`, retaining the relevant variables within.
- **Template Definition**: After completing the above splitting, use the three filling variables {static_params}, `{dynamic_params}`, and `{position_params}` (the latter two are split from file_params based on the value of param_type) to construct a template string `command_template`, so that subsequent programs can use `command_template` to restore the original command based on the previously defined `output_files`, `file_params`, and `static_params`.
    - Except for the part of redirecting log output, the template string should only contain the three variables mentioned above (the redirection part can use `{output_dir}` and `{sample_name}`)
    - This step and the above **Command Splitting** are encapsulated in the tool `set_params`, because in fact, as the splitting is completed, the definition of the template string is also completed simultaneously.
- **Information Generation**: Complete the remaining node information
    - Name the tool and generate a brief description, following the naming rules specified in the <Naming> section.
    - Set the `use_temp_dir` value: If in the previous **Output Organization** step you decided to keep only some output files, set this value to true.  
    - Set the `mkdir_output` value: Based on your knowledge (or if the user provided relevant information), determine whether this tool needs a manually created output folder. When your knowledge is insufficient, one approach you can consider is to check if the tool's parameters include options like override or force; these usually indicate that the tool will create the output folder itself (if these parameters are used to force override the output path).

Each decomposed task has a corresponding tool. When you complete the design, invoke the corresponding tool, passing your design results to the next task, until all tasks are completed. In this process, each tool is called **at most once**.
</ThoughtProcess>

<OutputFiles>
Based on your knowledge and the user-provided information, define a list of the software's output files.
Each output file must include:
- `name` and `description`: The name field needs to be globally unique because it will serve as the keyword for the node's output interface.
- `file_path`: Use `{output_dir}` and `{sample_name}` to construct a dynamic path template string. `output_dir` represents the output directory, and `sample_name` represents the sample name.
- `mount_path`: The output path of the file within the Docker container. If not specified by the user, the default value is `/data/output`.
- `is_report`: If the file is a "read-only" report file (such as html, png, svg, etc.), set it to true. However, for files like csv, json, or similar tsv that are functionally reports but can be used for subsequent analysis tasks, do not set it to true.
- `There must be one and only one file named "log" with `is_log` set to true, representing the tool's execution log. This log might be generated by the tool itself, or if the tool lacks this feature, it will be output by the user using `...&> {log}`.
</OutputFiles>

<ParamDefine>
For each "file_param", the following fields must be included:  
- `param_type`: The type of parameter, where 0 represents INPUT, 1 represents INPUT_POSITION, and 2 represents OUTPUT.  
- `name`: The keyword name of the parameter, which must consist only of lowercase letters and underscores and be globally unique. If it is an output-type parameter and the corresponding file is already defined in `output_files`, the value here should be the same as the `name` value of that file in `output_files`.  
- `command`: The specific parameter composition, using the value of the `name` field defined above as a template variable.  
- `is_file`: Whether the corresponding input (output) value is a file or a folder (though this value is actually only used in the input section, as the final output is processed based on the definitions in `output_files`. However, it should still be set normally for the output section).  
- `index`: This parameter is only useful for INPUT_POSITION types and represents the order of this parameter in the positional parameters.  
- `required`: Whether this parameter is required (only applies to input parameters and determines whether the node must be connected).  
- `mount_path`: The mapping path of the file in the container. Since the final node execution depends on the container, the input and output directories need to be mapped into the container. The path settings here follow the following rules:  
    - First, this parameter only takes effect for input parameters, as the mapping of output files is already defined in `output_files`.  
    - If all input files are in one path, you can directly set the input to `/data/raw`.  
    - If the input may come from different paths, you can design different mapping paths, with the root directory located under `/data`.  
    - Specifically, if it is a database type (many bioinformatics software have such parameters), map it to `/data/db`.  
- `description`: A description of the parameter, based on the help documentation and your knowledge.
</ParamDefine>

<Naming>
- For general-purpose tools, use the original tool name directly (can be changed from all lowercase to mixed case).
- For specific cases, the name should reflect its particular use:
    - Subcommand: Format as "ToolName(subcommand)", e.g., "samtools(index)".
    - Specialized use related to input/output files or key parameters: Format as "ToolName(key_identifier)". For example, if the tool outputs DAA files, it would be "diamond(daa)"; if it processes single-cell data, it would be "SPAdes(sc)".
</Naming>

<Example>
For example, below is the definition of the assembly tool SPAdes for assembling single cells:
{
  "output_files": [
    {
      "name": "contigs",
      "file_path": "{output_dir}/contigs.fasta",
      "is_report": false,
      "is_log": false,
      "mount_path": "/data/output",
      "description": "resulting contigs"
    },
    {
      "name": "scaffolds",
      "file_path": "{output_dir}/scaffolds.fasta",
      "is_report": false,
      "is_log": false,
      "mount_path": "/data/output",
      "description": "resulting scaffolds (recommended for use as resulting sequences)"
    },
    {
      "name": "assembly_graph",
      "file_path": "{output_dir}/assembly_graph_with_scaffolds.gfa",
      "is_report": false,
      "is_log": false,
      "mount_path": "/data/output",
      "description": "Assembly graph file with scaffolds in GFA format"
    },
    {
      "name": "log",
      "file_path": "{output_dir}/spades.log",
      "is_report": false,
      "is_log": true,
      "mount_path": "/data/output",
      "description": "log"
    }
  ],
  "file_params": [
    {
      "name": "r1",
      "command": "-1 {r1}",
      "param_type": 0,
      "is_file": true,
      "index": null,
      "required": true,
      "mount_path": "/data/raw",
      "description": "file with forward paired-end reads"
    },
    {
      "name": "r2",
      "command": "-2 {r2}",
      "param_type": 0,
      "is_file": true,
      "index": null,
      "required": true,
      "mount_path": "/data/raw",
      "description": "file with reverse paired-end reads"
    },
    {
      "name": "output",
      "command": "-o {output_dir}",
      "param_type": 2,
      "is_file": true,
      "index": null,
      "required": true,
      "mount_path": "",
      "description": "directory to store all the resulting files (required)"
    }
  ],
  "static_params": "--sc --careful -t 128 -m 64",
  "command_template": "spades.py {dynamic_params} {static_params}",
  "name": "SPAdes(sc)",
  "description": "Single-cell genome assembly using SPAdes",
  "mkdir_output": true,
  "use_temp_dir": true
}
</Example>
"""


class InputType(TypedDict):
    messages: list[BaseMessage]


class GenerateState(AgentState):
    name: str
    description: str
    command_template: str
    output_files: Annotated[list[FileMount], operator.add]
    file_params: Annotated[list[ParamDefine], operator.add]
    static_params: str
    mkdir_output: bool
    use_temp_dir: bool


class GenerateOutput(BaseModel):
    name: str
    description: str
    command_template: str
    output_files: list[FileMount]
    file_params: list[ParamDefine]
    static_params: str
    mkdir_output: bool
    use_temp_dir: bool


@tool
def design_example_command(command: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Pass the command you generated to the subsequent task

    Args:
        command: Complete command generated based on task requirements and tool help documentation
    """
    return Command(
        update={
            'messages': [
                ToolMessage(
                    'To meet the requirements, a feasible command designed. Now, start executing the **Output Organization** task by calling tool `set_output_files`',
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool
def set_output_files(output_files: list[FileMount], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Set the expected output file for this tool node

    Args:
        output_files: List of output files
    """

    return Command(
        update={
            'output_files': output_files,
            'messages': [
                ToolMessage(
                    'The output files of the tool have been encapsulated. You can continue to start the work **Command Splitting** and **Template Definition** by calling tool `set_params`',
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def set_params(
    file_params: list[ParamDefine],
    static_params: str,
    command_template: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Decompose and reorganize the command parameters of the tool

    Args:
        file_params: Parameters related to external file input and tool file output
        static_params: Other parameters
        command_template: Template string of the command
    """
    return Command(
        update={
            'file_params': file_params,
            'static_params': static_params,
            'command_template': command_template,
            'messages': [
                ToolMessage(
                    'The params of the tool have been encapsulated. Based on the above work, complete **Information Generation** by calling tool `set_tool_info`',
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def set_tool_info(
    name: str,
    description: str,
    mkdir_output: bool,
    use_temp_dir: bool,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Set the basic information of the tool

    Args:
        name: Name of the tool node
        description: The description of the task encapsulated by this node
        mkdir_output: Do you need to manually create the output folder?
        use_temp_dir: Do you need to use a temporary folder?
    """
    return Command(
        update={
            'name': name,
            'description': description,
            'mkdir_output': mkdir_output,
            'use_temp_dir': use_temp_dir,
            'messages': [ToolMessage('All tasks have been completed.', tool_call_id=tool_call_id)],
        }
    )


generate_agent = create_react_agent(
    model=gemini_25_flash(),
    tools=[design_example_command, set_output_files, set_params, set_tool_info],
    prompt=SYSTEM_PROMPT,
    state_schema=GenerateState,
)
