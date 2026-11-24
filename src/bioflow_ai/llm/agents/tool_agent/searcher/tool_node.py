import uuid
from typing import TYPE_CHECKING, Any, cast

from langchain.agents import AgentState, create_agent

from bioflow_ai.llm.context import BaseContext
from bioflow_ai.llm.middleware import (
    ChatAttachmentMiddleware,
    DisableParallelToolCallsMiddleware,
    EndLoopMiddleware,
    TokenStatisticMiddleware,
    ToolCreationProgressMiddleware,
)
from bioflow_ai.llm.models import get_agent_model
from bioflow_ai.llm.tools.database import read_tool_doc, select_tool
from bioflow_ai.llm.tools.vector_store import search_tools_by_function
from bioflow_ai.utils.uid_mapper import BidirectionalMapper

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from bioflow_ai.llm.llm_client import LLMMetadata

SYSTEM_PROMPT = """You are a bioinformatics data administrator. The database contains many commonly used bioinformatics analysis tools packaged within it.
Your task is to search the database by calling tools according to the user's requirements and determine whether the current database contains a tool that meets those requirements.

## Task Workflow

1. Call the search_tools tool according to the user's requirements to search for existing tools.
2. Determine whether a suitable tool exists in the database; if necessary, you may call the read_tool_doc tool to read the help documentation for the relevant tool.
3. If a tool meeting the user's requirements is found, call the select_tool tool to select it, then end the task; otherwise, end the task directly.

## Search Strategy

You must not conclude that no suitable result exists based on a single search result. Before concluding that a new tool needs to be created, you must call the search_tools tool multiple times and use the following search strategies:

1. Use concise, standardized tool/node names and the required subcommands.
2. Use recognized aliases, executable filenames, package names, or other spelling variants.
3. Use concise descriptions of the required functionality, input/output conversion, or broader tool categories.

Only after confirming that no suitable tool was found using all of the search strategies may you conclude that a new tool needs to be created.

## Decision Rules

This system packages bioinformatics tools as nodes representing single commands, like this:

┌─────────────────────────────────┐
│ fastp                           │
├─────────────────────────────────┤
│  ○ r1               clean_r1 ○  │
│  ○ r2               clean_r2 ○  │
│                   fastp_json ○  │
├─────────────────────────────────┤
│ Immutable Params:               │
│                                 │
│  -V                             │
├─────────────────────────────────┤
│ Modifiable Params:              │
│                                 │
│  -w 32                          │
│                                 │
└─────────────────────────────────┘

Because inputs and outputs are directly bound to the node, this must be taken into consideration when determining whether a new node is needed. More specifically, follow the rules below to determine whether a tool meeting the user's requirements exists:

### Requirements Met

- A usable tool is found whose functionality completely meets the user's requirements.
- A usable tool is found; although its parameter settings do not completely match the user's requirements, the requirements can be met by adjusting (adding, deleting, or modifying) parameters with the editable=True attribute. args supports adjusting only value-type or Boolean-type parameters, such as setting thresholds or selecting algorithms. If the parameter to be added is a file, the current tool cannot meet the requirements, and a new tool wrapper must be created!

### Requirements Not Met

- After multiple attempts, it is confirmed that the search results are empty.
- After multiple attempts, it is confirmed that none of the search results meet the user's requirements.
- A usable tool is found, but the requirements conflict with a parameter with the editable=False attribute (such parameters usually cause changes to the node's input/output handles); in this case, a new tool node must be created rather than using the existing tool.
- A usable tool is found, but the parameters that need to be added involve actual file paths, meaning new input_handles are required, which cannot be achieved by modifying parameter fields.
"""


class SearchToolNodeState(AgentState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID


def build_search_exist_tool_agent() -> 'CompiledStateGraph[Any, BaseContext, Any, Any]':
    """Build the existing tool search agent with the current agent model."""
    model = get_agent_model()
    metadata = cast('LLMMetadata', model.metadata)
    return create_agent(
        model=model,
        tools=[
            search_tools_by_function,
            read_tool_doc,
            select_tool,
        ],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            DisableParallelToolCallsMiddleware(disabled_tools={'select_tool'}),
            EndLoopMiddleware(),
            ToolCreationProgressMiddleware(),
            ChatAttachmentMiddleware(),
            TokenStatisticMiddleware('search_exist_node', **metadata),
        ],
        state_schema=SearchToolNodeState,
        context_schema=BaseContext,
    )
