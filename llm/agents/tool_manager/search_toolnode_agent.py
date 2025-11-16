import uuid

from langchain.agents import create_agent
from langgraph.graph import MessagesState

from llm.models import gpt_5_mini
from llm.tools.database import select_tool
from llm.tools.vector_store import search_tools_by_function
from llm.utils.uid_mapper import BidirectionalMapper

SYSTEM_PROMPT = """# Role and Task
You are a professional bioinformatics data administrator.
Your task is to call tools to search databases based on user needs and determine if a tool with this function exists in the current database.

# Chain of Thought
1. **Search by function**: First, call the `search_tools_by_function` tool to search the local database for tools that satisfy this function.
    - If no results are returned, it means there are no tools that satisfy this function, and the task can be ended directly.
    - If results are returned, analyze them further.
2. **Analyze search result**: Based on the search results, determine if there are any tools that meet the requirements.
    - Meets requirements: Proceed to the next step.
        - According to the description and command, this tool perfectly meets the requirements.
        - Although the command does not completely meet the requirements, according to the documentation, it can be achieved by modifying the parameters within <Args>.
    - Does not meet requirements: End the task directly.
        - According to the documentation, there are no tools that meet the requirements.
        - It can be achieved by modifying the parameters within <Args>, but it will change the tool's output file.
3. **Select exist tool**: If there is a tool that meets the requirements, call the `select_tool` tool to select it, then end the task.
"""


class SearchToolNodeState(MessagesState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    tool_uid: str


search_exist_tool_agent = create_agent(
    model=gpt_5_mini(),
    tools=[search_tools_by_function, select_tool],
    system_prompt=SYSTEM_PROMPT,
    state_schema=SearchToolNodeState,
)
