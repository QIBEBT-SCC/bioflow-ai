from typing import TYPE_CHECKING, Optional, cast

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from llm.context import ProjectContext
from llm.models import get_agent_model
from llm.states import BuildSchema
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

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata

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
- `workflow_manager` — workflow draft and panel operations (creating/switching workflow drafts, viewing, deleting nodes, saving all workflows, etc.)


## Task Process

You will be given a prepared plan along with the user's original task requirements. Based on the user's description and the plan, complete the following steps:

1. Read the plan and call the `add_todos` tool to initialize a Todo List.
2. Check the Todo List, select the next `pending` task, and call `update_todo(..., status='in_progress')` to update its status to in progress.
3. Before constructing a workflow, call `workflow_manager` to create or switch to the correct workflow draft.
   - Use one workflow draft for each workflow described in the plan.
   - The status bar shows the current active workflow draft. Ensure the active workflow matches the task being processed before adding nodes.
   - When moving from one workflow to another, call `switch_workflow_draft`; it snapshots the current panel and loads the selected workflow draft.
4. Depending on the task type, call the appropriate tool to operate on the panel and add the relevant nodes.
   - If the task does not specify which tool to use, or provides multiple options, select a suitable tool based on your own knowledge and search results to complete the task.
   - If the original requirements specify a particular tool but you consider it unsuitable, you may replace it with a better tool and add a note node to explain the change.
5. Once the current task is completed:
   1. First, call the `summarize_work` tool to write a handover document for subsequent tasks.
      - This information should include two parts: 1. a mermaid description of the current panel; 2. an explanation of the work completed in this round.
      - In the handover information, pay special attention to noting the purpose of certain resource nodes (e.g., sequence node A, used for receiving raw paired-end sequences), whether these are from previous handover information or newly added in the current round.
   2. Then, call `update_todo(..., status='completed')` to update the task status.
6. After all command-line tool nodes from the upstream analysis have been added, if the plan includes corresponding downstream analysis, add a `downstream_summary` node:
   - Connect all files required for the downstream analysis to this node.
   - Fill in the `prompt` field to guide the subsequent downstream analysis code writing.
7. Repeat the above steps until all upstream workflow drafts in the plan have been constructed.
8. After every workflow draft is complete, call `workflow_manager` with `save_all_workflows` to save all workflow drafts to the database in one operation, then provide a final summary and conclude the work.


## Notes
- Only one task can be processed at a time. Before starting the next task, ensure that the current task has been marked as "completed".
- When a task is marked as "completed", the current context information will be cleared. You can obtain necessary information by viewing the to-do list, handover information, and current workflow status.
- Do not save an individual workflow immediately after it is constructed. Keep it as a workflow draft and save all drafts together only after every workflow in the plan is complete.
- If the plan contains multiple workflows, use `create_workflow_draft` for each workflow and `switch_workflow_draft` whenever you need to inspect or continue editing a previous draft.
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


def build_workflow_builder(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Compile the workflow-builder agent.

    Pass a checkpointer to run it as a top-level graph with native interrupt/resume; pass
    ``None`` when embedding it as a subgraph so it inherits the parent graph's checkpointer.

    Args:
        checkpointer: Checkpointer to bind, or ``None`` to inherit from a parent graph.

    Returns:
        The compiled workflow-builder agent.
    """
    if checkpointer:
        # pyrefly: ignore [no-matching-overload]
        return create_agent(
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
                        'FILE_SYSTEM',
                        'USING_BASH_NODE',
                        'WRITE_DOWNSTREAM_SUMMARY',
                    ]
                ),
                CheckStateExistMiddleware('workflow_uids', 'workflow_manager'),
                DisableParallelToolCallsMiddleware(),
                StatusBarMiddleware(),
                TokenStatisticMiddleware('workflow_builder', **metadata),
            ],
            state_schema=BuildSchema,
            context_schema=ProjectContext,
            checkpointer=checkpointer,
        )
    # pyrefly: ignore [no-matching-overload]
    return create_agent(
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
        context_schema=ProjectContext,
    )


build_workflow_agent: CompiledStateGraph = build_workflow_builder()
