import uuid

from langchain.agents import create_agent
from langgraph.graph import MessagesState

from llm.models import get_agent_model
from llm.states import UserContext
from llm.tools.database import select_tool
from llm.tools.vector_store import search_tools_by_function
from llm.utils.middleware import TokenStatisticMiddleware
from llm.utils.uid_mapper import BidirectionalMapper

SYSTEM_PROMPT = """您是一位专业的生物信息学数据管理员，在这个数据库中，封装了很多常见生物信息分析工具。
您的任务是根据用户需求调用工具搜索数据库，并确定当前数据库中是否存在满足此需求的工具。

## 判断规则
在调用搜索工具后，遵守以下规则，判断是否存在满足用户需求的工具
若找到了符合用户需求的工具，调用 `select_tool` 工具选择该工具，然后结束任务；
否则，直接结束任务

### 满足需求
- 找到了可用的工具，且其功能完美符合用户需求
- 找到了可用工具，虽然参数设置不完全符合用户需求，但是可以通过调整`ModifiableParams`来实现

### 不满足需求
- 搜索工具返回结果为空
- 搜索工具返回了结果，但是其中完全没有符合用户要求的工具
- 找到了可用工具，但是需要实现的需求与工具的`ImmutableParams`冲突，此时用户必须新建一个节点而不能使用数据库内的
"""


class SearchToolNodeState(MessagesState, total=False):
    id_mapper: BidirectionalMapper

    image_uid: uuid.UUID
    tool_uid: str


search_exist_tool_agent = create_agent(
    model=get_agent_model(),
    tools=[
        search_tools_by_function,
        select_tool,
    ],
    system_prompt=SYSTEM_PROMPT,
    middleware=[  # type: ignore
        TokenStatisticMiddleware(agent_name='search_exist_node', **get_agent_model().metadata),
    ],
    state_schema=SearchToolNodeState,  # type: ignore
    context_schema=UserContext,  # type: ignore
)
