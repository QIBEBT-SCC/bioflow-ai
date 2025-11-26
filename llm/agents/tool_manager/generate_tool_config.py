from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pydantic import BaseModel

from app.models.tools import FileMount, ParamDefine
from llm.states import ToolGeneratorState

SYSTEM_PROMPT = """You are a bioinformatics engineer

<ThoughtProcess>
In the context information, the user will provide a test run command, and all you need to do is abstract based on this command and the output files it generates.
To achieve this goal, you can break down the task into the following steps:
1. **Mount Files**: Process the tool's input/output files, organizing them into a list with specific format (details are provided in the <FileMounts> section)
    - If the tool's output consists of only a few files, list all of them.
    - If the tool's output files are complex (e.g., including intermediate files generated during execution, temporary folders, etc.), only list files needed for subsequent analysis tasks based on your knowledge.
2. **Command Design**: Based on the tool documentation and the file mounts generated in the previous step, abstract the commands in the trial run report, splitting them into parameter variables and template strings.
    a. Follow the rules in <ParamDefine> to extract the parameters involved in the command into three categories: `dynamic_params`, `immutable_static_params`, and `modifiable_static_params`.
    b. Follow the rules in <TemplateRule> to generate the template string for the complete command.
3. **Information Generation**: Name the tool and generate a brief description, following the naming rules specified in the <Naming> section.
</ThoughtProcess>

<FileMounts>
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
</FileMounts>

<ParamRules>
The parameters involved in the command are divided into the following three categories:
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
</ParamRules>

<TemplateRule>
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
</TemplateRule>

<Naming>
- For general-purpose tools, use the original tool name directly.
- For specific cases, the name should reflect its particular use:
    - Subcommand: Format as "ToolName subcommand"
    - Specialized use related to input/output files or key parameters: Format as "ToolName(key_identifier)".
</Naming>

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


class ToolConfigSchema(TypedDict):
    name: str
    description: str
    file_mounts: list[FileMount]
    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str
    modifiable_static_params: str


class FileMountsResult(BaseModel):
    file_mounts: list[FileMount]


class CommandDesignResult(BaseModel):
    dynamic_params: list[ParamDefine]
    immutable_static_params: str
    modifiable_static_params: str
    command_template: str


class ToolInfoResult(BaseModel):
    name: str
    description: str


class ToolGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def mount_file_node(self, state: ToolGeneratorState):
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name='history'),
                HumanMessage(content='First, please design the file mount list for the tool.'),
            ]
        )
        chain = prompt | self.llm.with_structured_output(
            FileMountsResult,
            method='json_schema',
        ).with_retry(retry_if_exception_type=(ValueError,), stop_after_attempt=3)

        response: FileMountsResult = await chain.ainvoke({'history': state['messages']})

        return {
            'file_mounts': response.file_mounts,
            'messages': [
                AIMessage(content=response.model_dump_json(ensure_ascii=True)),
                HumanMessage(content='Okay, next, please design the parameters and command template.'),
            ],
        }

    async def command_design_node(self, state: ToolGeneratorState):
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name='history'),
            ]
        )
        chain = prompt | self.llm.with_structured_output(
            CommandDesignResult,
            method='json_schema',
        ).with_retry(retry_if_exception_type=(ValueError,), stop_after_attempt=3)

        response: CommandDesignResult = await chain.ainvoke({'history': state['messages']})

        return {
            'command_template': response.command_template,
            'dynamic_params': response.dynamic_params,
            'immutable_static_params': response.immutable_static_params,
            'modifiable_static_params': response.modifiable_static_params,
            'messages': [
                AIMessage(content=response.model_dump_json(ensure_ascii=True)),
                ('human', "Great, let's complete the remaining information."),
            ],
        }

    async def info_generate_node(self, state: ToolGeneratorState):
        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=SYSTEM_PROMPT), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.llm.with_structured_output(
            ToolInfoResult,
            method='json_schema',
        ).with_retry(retry_if_exception_type=(ValueError,), stop_after_attempt=3)

        response: ToolInfoResult = await chain.ainvoke({'history': state['messages']})

        return {
            'name': response.name,
            'description': response.description,
        }

    def build(self, visualize: bool = False):
        graph = StateGraph(ToolGeneratorState, output_schema=ToolConfigSchema)
        graph.add_sequence(
            [
                ('mount_file', self.mount_file_node),
                ('command_design', self.command_design_node),
                ('info_generate', self.info_generate_node),
            ]
        )

        graph.set_entry_point('mount_file')
        graph.set_finish_point('info_generate')

        compiled_graph = graph.compile()
        if visualize:
            mermaid_code = compiled_graph.get_graph().draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with open('images/tool_generator.md', 'w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')
        return compiled_graph
