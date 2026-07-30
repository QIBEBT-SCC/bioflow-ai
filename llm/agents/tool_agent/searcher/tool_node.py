import uuid
from typing import TYPE_CHECKING, cast

from langchain.agents import create_agent
from langgraph.graph import MessagesState

from llm.context import BaseContext
from llm.models import get_agent_model
from llm.tools.database import read_tool_doc, select_tool
from llm.tools.vector_store import search_tools_by_function
from llm.utils.middleware import TokenStatisticMiddleware
from llm.utils.uid_mapper import BidirectionalMapper

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata

SEARCH_LLM = get_agent_model()
metadata = cast('LLMMetadata', SEARCH_LLM.metadata)

SYSTEM_PROMPT = """You are a professional bioinformatics data administrator. In this database, many common bioinformatics analysis tools are encapsulated.
Your task is to call tools to search the database based on user requirements and determine whether there is a tool in the current database that meets these requirements.

## Task Flow
1. Call the `search_tools_by_function` tool based on user requirements to search for existing tools. Follow the
   mandatory search policy below.
2. Determine whether there is a suitable tool in the database based on the following rules; if necessary, you can call the `read_tool_doc` tool to read the help documentation of the corresponding tool.
3. If a tool that meets the user's requirements is found, call the `select_tool` tool to select that tool, and then end the task; otherwise, end the task directly.

## Mandatory Search Policy
One empty or irrelevant result is not enough to conclude that a matching node is absent. Before ending without selecting
a tool, you MUST make at least three sequential `search_tools_by_function` calls using materially different strategies:
1. The concise canonical tool/node name and required subcommand.
2. A recognized alias, executable, package name, or alternative spelling.
3. A concise description of the required function, input/output transformation, or broader tool category.

Do not count cosmetic rewrites or repeated queries as separate strategies. If results are returned but none appears
suitable, refine the next query and continue. Stop early only when a suitable node is found. You may conclude that the
database has no suitable node only after all three strategies fail.

## Judgment Rules
Follow the rules below to determine if there is a tool that meets the user's requirements.

### Requirements Met
- A usable tool is found, and its functionality perfectly matches the user's requirements.
- A usable tool is found; although the parameter settings do not fully match the user's requirements, they can be achieved by adjusting (adding, deleting, or modifying) parameters with the "editable=True" attribute.

### Requirements Not Met
- All three required search strategies have been attempted and returned no results.
- All three required search strategies have been attempted and the returned tools do not meet the user's requirements.
- A usable tool is found, but the requirements conflict with parameters having the "editable=False" attribute; in this case, the user must create a new tool encapsulation instead of using the existing one.
- A usable tool is found, but the parameters to be added are related to actual file paths, requiring the addition of input_handles, which cannot be achieved by modifying parameter fields.

**Note**: "args" only support adjustments for value or boolean type parameters, such as setting thresholds or selecting algorithms. If the parameter to be added is a file, the current tool cannot meet the requirement, and a new tool must be created!
"""


class SearchToolNodeState(MessagesState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    tool_uid: str


# pyrefly: ignore [no-matching-overload]
search_exist_tool_agent = create_agent(
    model=get_agent_model(),
    tools=[
        search_tools_by_function,
        read_tool_doc,
        select_tool,
    ],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        TokenStatisticMiddleware('search_exist_node', **metadata),
    ],
    state_schema=SearchToolNodeState,
    context_schema=BaseContext,
)
