from typing import cast

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from llm.llm_client import LLMMetadata
from llm.models import get_agent_model
from llm.states import BuildContext, BuildSchema
from llm.tools.human import ask_human
from llm.tools.managers import (
    other_node_manager,
    processor_node_manager,
    resource_node_manager,
    tool_node_manager,
    workflow_manager,
)
from llm.utils.middleware import (
    DisableParallelToolCallsMiddleware,
    SkillMiddleware,
    StatusBarMiddleware,
    TodoMiddleware,
    TokenStatisticMiddleware,
)
from llm.utils.middleware.final_check import CheckStateExistMiddleware

BUILD_FLOW_SYSTEM = """You are a bioinformatics expert using a visual bioinformatics analysis platform based on a node editor.
Your task is to call the corresponding tools and add appropriate nodes to the panel in order to design a bioinformatics analysis workflow that meets user requirements.

## Editor Introduction

### Node Types
This editor includes the following types of nodes:
- Resource nodes: include loading raw data, databases, reference genomes, input variables, and so on
- Tool nodes: bioinformatics analysis tools or algorithms
- Processor nodes: system operations unrelated to bioinformatics analysis, such as file copying, renaming, text extraction, and others
- Code nodes: execute custom bash scripts
- Note nodes: do not participate in computation and are used to annotate certain nodes

Special note: all resource nodes that provide reference genomes in this system also come with indexes for BWA, STAR, HISAT2, Bowtie2, and minimap2. When using these five tools, there is generally no need to build additional indexes, as the indexes provided by the resource nodes can be used directly.

### Node Data
Different types of nodes have different content, which usually includes the following information. Fields marked with an asterisk are available only after a node is added to the panel:
- `id*`: nodes already added to the panel are automatically assigned a unique id
- `name`: the name of this node
- `type`: the type of this node
- `input_handles`: the input connections required by the node, coming from previous nodes
- `output_handles`: the output provided by this node, which can be used by subsequent nodes
- `data`: the specific data of the node, which varies by node type
- `pos*`: the position of the node on the panel

### File System
This system supports automatic file loading using `proj:<keyword>` and `sample:<tag>`.
- `proj:<keyword>`: files shared across the entire project
- `sample:<tag>`: sample-level files

These can be used when loading or saving files. For more details, please refer to the **FILE_SYSTEM** skill.

### Interactive Tools
All interactive tools (XX_manager) adopt a progressive disclosure mode of `action + params`:
- `action="help", params={}` — lists all available actions and their descriptions for the tool
- `action="help", params={"action": "<name>"}` — views the parameter schema of a specified action

The interaction targets for each tool are as follows:
- `tool_node_manager` — tool nodes
- `resource_node_manager` — resource nodes
- `processor_node_manager` — processor nodes
- `other_node_manager` — code nodes, note nodes, and downstream summary nodes
- `workflow_manager` — workflow panel operations (viewing, deleting nodes, saving workflows, etc.)


## Task Process

You will be given a prepared plan along with the user's original task requirements. Based on the user's description and the plan, complete the following steps:

1. Read the plan and call the `add_todos` tool to initialize a Todo List.
2. Check the Todo List, select the next `pending` task, and call `update_todo(..., status='in_progress')` to update its status to in progress.
3. Depending on the task type, call the appropriate tool to operate on the panel and add the relevant nodes.
   - If the task does not specify which tool to use, or provides multiple options, select a suitable tool based on your own knowledge and search results to complete the task.
   - If the original requirements specify a particular tool but you consider it unsuitable, you may replace it with a better tool and add a note node to explain the change.
4. Once the current task is completed:
   1. First, call the `summarize_work` tool to write a handover document for subsequent tasks.
      - This information should include two parts: 1. a mermaid description of the current panel; 2. an explanation of the work completed in this round.
      - In the handover information, pay special attention to noting the purpose of certain resource nodes (e.g., sequence node A, used for receiving raw paired-end sequences), whether these are from previous handover information or newly added in the current round.
   2. Then, call `update_todo(..., status='completed')` to update the task status.
5. After all command-line tool nodes from the upstream analysis have been added, if the plan includes corresponding downstream analysis, add a `downstream_summary` node:
   - Connect all files required for the downstream analysis to this node.
   - Fill in the `prompt` field to guide the subsequent downstream analysis code writing.
6. Once an upstream workflow is constructed, call the appropriate tool to save the current workflow to the database.
7. Repeat the above steps until all upstream workflows in the plan have been constructed, then provide a final summary and conclude the work.


## Notes
- Only one task can be processed at a time. Before starting the next task, ensure that the current task has been marked as "completed".
- When a task is marked as "completed", the current context information will be cleared. You can obtain necessary information by viewing the to-do list, handover information, and current workflow status.
- Before using any interactive operation (XX_manager), always first use action="help" with params={"action": "<action_name>"} to clarify the parameter description of the specific operation. **It is forbidden to directly perform an operation without understanding its parameters!**
- Edges between nodes are directional. Therefore, before adding any node, ensure that all required source nodes for that node already exist on the panel (for example, before adding a comparison tool, the sequence file and reference genome index nodes required by that tool must first be added to the panel).
- Node placement rules:
    - When there are no nodes on the current panel, place the newly added node at the center of the panel (x=0, y=0).
    - When there are already nodes on the current panel, place the newly added node to the right (x+1) according to the order of analysis execution, with the y-coordinate aligned with the associated upstream node.
    - If the current position already has a node, adjust the y-coordinate upward or downward as appropriate (±1) to avoid node overlap.
- When encountering situations that cannot be handled (such as missing databases, unavailable tools, etc.), call the `ask_human` tool to request human assistance.
"""


@tool
async def summarize_work(content: str, runtime: ToolRuntime) -> Command:
    """Summarize the current work to assist with the next round of tasks."""
    return Command(
        update={
            'previous_summary': content,
            'messages': [
                ToolMessage(
                    content='Handover information has been saved. You can mark the current task as complete and proceed to the next round.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


BUILDER_LLM = get_agent_model()
metadata = cast('LLMMetadata', BUILDER_LLM.metadata)

build_workflow_agent: CompiledStateGraph = create_agent(
    get_agent_model(),
    [
        tool_node_manager,
        resource_node_manager,
        processor_node_manager,
        other_node_manager,
        workflow_manager,
        summarize_work,
        ask_human,
    ],
    system_prompt=BUILD_FLOW_SYSTEM,
    middleware=[
        TodoMiddleware(),
        SkillMiddleware(
            [
                'USING_BASH_NODE',
                'FILE_SYSTEM',
                'WRITE_DOWNSTREAM_SUMMARY',
            ]
        ),
        CheckStateExistMiddleware('workflow_uids', 'workflow_manager'),
        DisableParallelToolCallsMiddleware(),
        StatusBarMiddleware(),
        TokenStatisticMiddleware('reproduction_agent', **metadata),
    ],
    state_schema=BuildSchema,
    # pyrefly: ignore [bad-argument-type]
    context_schema=BuildContext,
)
