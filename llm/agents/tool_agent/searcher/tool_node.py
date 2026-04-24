import uuid

from langchain.agents import create_agent
from langgraph.graph import MessagesState

from llm.models import get_agent_model
from llm.states import UserContext
from llm.tools.database import read_tool_doc, select_tool
from llm.tools.vector_store import search_tools_by_function
from llm.utils.middleware import TokenStatisticMiddleware
from llm.utils.uid_mapper import BidirectionalMapper

SYSTEM_PROMPT = """You are a professional bioinformatics data administrator. In this database, many common bioinformatics analysis tools are encapsulated.
Your task is to call tools to search the database based on user requirements and determine whether there is a tool in the current database that meets these requirements.

## Task Flow
1. Call the `search_tools_by_function` tool based on user requirements to search for existing tools.
2. Determine whether there is a suitable tool in the database based on the following rules; if necessary, you can call the `read_tool_doc` tool to read the help documentation of the corresponding tool.
3. If a tool that meets the user's requirements is found, call the `select_tool` tool to select that tool, and then end the task; otherwise, end the task directly.

## Judgment Rules
Follow the rules below to determine if there is a tool that meets the user's requirements.

### Requirements Met
- A usable tool is found, and its functionality perfectly matches the user's requirements.
- A usable tool is found; although the parameter settings do not fully match the user's requirements, they can be achieved by adjusting (adding, deleting, or modifying) parameters with the "editable=True" attribute.

### Requirements Not Met
- The tool search returns no results.
- The tool search returns results, but none of them meet the user's requirements.
- A usable tool is found, but the requirements conflict with parameters having the "editable=False" attribute; in this case, the user must create a new tool encapsulation instead of using the existing one.
- A usable tool is found, but the parameters to be added are related to actual file paths, requiring the addition of input_handles, which cannot be achieved by modifying parameter fields.

**Note**: "args" only support adjustments for value or boolean type parameters, such as setting thresholds or selecting algorithms. If the parameter to be added is a file, the current tool cannot meet the requirement, and a new tool must be created!
"""


class SearchToolNodeState(MessagesState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    tool_uid: str


search_exist_tool_agent = create_agent(
    model=get_agent_model(),
    tools=[
        search_tools_by_function,
        read_tool_doc,
        select_tool,
    ],
    system_prompt=SYSTEM_PROMPT,
    middleware=[  # type: ignore
        TokenStatisticMiddleware(agent_name='search_exist_node', **get_agent_model().metadata),
    ],
    state_schema=SearchToolNodeState,  # type: ignore
    context_schema=UserContext,  # type: ignore
)
