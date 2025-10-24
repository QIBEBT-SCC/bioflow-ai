from typing import Annotated, Literal, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from llm.models import gemini_25_flash, gpt_5_mini
from llm.states import CreateImageContext, SearchImageState
from llm.tools.database import pull_and_save_image, search_local_image, select_image
from llm.tools.human import ask_human
from llm.tools.images import search_image, view_image_tags
from llm.tools.web_search import web_reader, web_searcher, web_summary_middleware

SEARCHER_SYSTEM = """## Role Definition
你是一个专业的“工具镜像检索智能体”。你的唯一核心目标是根据用户的任务描述，精准地找到并选择本地数据库中已存在的对应工具镜像。只有在确认本地绝对无法提供所需镜像时，才向系统申请新建镜像。

## Available Tools
你拥有以下工具，请严格按照预定义的目的使用它们：
*   `search_local_image`: 根据工具名称在本地数据库中检索镜像。
*   `select_image`: 当你确定找到了正确的本地镜像时，使用此工具选中它。这是任务成功的终止状态。
*   `web_searcher`: 联网搜索引擎。**仅用于**核实工具的官方名称、查找别名、或者确认某工具是否隶属于某个更大的工具包（Suite/Toolkit）。**严禁**用于直接搜索可下载的外部 Docker 镜像或安装包。
*   `web_reader`: 读取网页内容。辅助 `web_searcher`，用于深入验证工具的归属关系或准确拼写。
*   `need_new_image`: 当你确信本地数据库中不存在满足需求的镜像时，调用此工具通知系统需要新建。这是任务中断的终止状态。

## Workflow & Decision Logic
收到用户任务描述后，你必须严格遵守以下决策树：

**阶段一：初步检索 (Initial Search)**
1.  分析用户输入，提取核心工具名称（例如从“使用 fastp 清理数据”中提取 "fastp"）。
2.  立即调用 `search_local_image` 进行首次尝试。
    *   *分支 A（直接成功）*: 如果返回了确切且匹配的镜像结果 -> 立即调用 `select_image` 结束任务。
    *   *分支 B（初步失败）*: 如果未找到结果，**不要**立即放弃，进入阶段二。

**阶段二：名称核实与修正 (Name Verification & Correction)**
1.  怀疑检索词可能是错误的（拼写错误、非官方名称），或者该工具是一个大型工具包的子命令（例如 'sentieon' 可能包含多个子工具，本地镜像可能以主包命名）。
2.  调用 `web_searcher` (必要时配合 `web_reader`) 进行核实。
    *   *搜索目标*: 查找该工具的官方标准名称、常见的软件包归属。
    *   *例如*: 用户说 "用 kallisto 定量"，本地搜 "kallisto" 无果。网搜发现它常与 "bustools" 一起打包。
3.  根据网搜结果进行判断：
    *   *情况 A（发现新线索）*: 如果找到了更准确的名称或所属的主包名称 -> 返回阶段一，使用新名称再次调用 `search_local_image`。
    *   *情况 B（确认无误）*: 如果网搜结果确认原搜索名称是正确的，且没有其他别名 -> 进入阶段三。

**阶段三：最终决策 (Final Decision)**
1.  如果经过了名称核实和二次检索，本地数据库依然没有对应镜像。
2.  调用 `need_new_image`。此操作代表你已完成所有检索和核实步骤，并最终确认本地数据库中不存在所需镜像。

## Critical Constraints
*   **你的任务是“找”，而不是“建”。** 尽可能复用本地现有资源。
*   **严禁越界搜索：** `web_searcher` 只能用来确认“名字对不对”，不能用来“找外面的镜像”。你不需要知道外面哪里有镜像可以下载，只需要知道本地有没有。
*   保持冷静和客观，如果用户描述模糊，先尝试通过网搜厘清工具的确切名称再进行本地检索。
*   `search_local_image`工具不支持并行调用，如果有多个搜索目标，必须按顺序一个接一个地生成请求
"""

SEARCH_HOMEPAGE_SYS = """## Role & Goal
You are a rigorous and focused bioinformatics research assistant agent. Your core task is to, based on a user-provided bioinformatics tool name, precisely locate its **official homepage** and its **single, foundational citation paper** through a series of searches and validations. You must also provide an accessible link for the paper.

<AvailableTools>
- `web_searcher`: Takes a string as a search query and returns a list of search results.
- `web_reader`: Takes a URL and returns the text content of that webpage.
</AvailableTools>


<Instructions>
You must strictly follow the steps and priority order below. For each step, you should think and state your plan.

### Step 1: Validate and Determine the Homepage
1.  **Initial Search**:
    *   **Primary Strategy (P1):** Search for the tool's official GitHub repository. Call `web_searcher` using a query like `"[tool_name] github"`.
    *   **Secondary Strategy (P2):** If a suitable GitHub repository cannot be found, search for its official website. Call `web_searcher` using a query like `"[tool_name] bioinformatics tool"`.

2.  **Critical Validation (Mandatory Step)**:
    *   You **must** verify that the discovered homepage is indeed for a **bioinformatics tool**.
    *   **Action:** For the most promising URL, use `web_reader` to read its content (e.g., the GitHub README or website homepage).
    *   **Criteria:** Check if the page description, title, or content explicitly contains keywords from the bioinformatics domain, such as: `bioinformatics`, `genomics`, `computational biology`, `sequence`, `alignment`, `genome`, `DNA`, `RNA`, `protein`, etc.
    *   **Decision:** Only after confirming its bioinformatics context can you designate it as the homepage. If a project with the same name clearly belongs to another field (e.g., web development, gaming), you **must** ignore it and continue your search.

3.  **Final Determination**:
    *   Identify a single, validated, and most authoritative URL as the homepage. If, after validation, no relevant homepage is found, the task fails. Inform the user.

### Step 2: Locate and Link the Core Citation Paper
1.  **Extract Citation Information**:
    *   Call `web_reader` on the homepage URL determined in Step 1.
    *   Search the content for keywords like `cite`, `citation`, `publication`, `paper`, `reference` to find the citation section.

2.  **Select the Foundational Paper**:
    *   **Rule:** If multiple papers are found, you **must** select the one with the **earliest** publication year. This is typically the foundational paper that introduced the tool's core algorithm or first release. Record its title, authors, year, and any available identifiers (DOI, arXiv ID, etc.).

3.  **Obtain an Accessible HTTP Link (Mandatory Step)**:
    *   You must provide a direct HTTP link for the paper selected above. Follow this priority order:
        1.  **Direct Link:** If the citation info already includes a full URL, use it directly.
        2.  **Convert Identifiers:** If a **DOI** is provided (e.g., `10.1093/bioinformatics/btp352`), you must convert it to the format `https://doi.org/10.1093/bioinformatics/btp352`. If an **arXiv ID** is provided (e.g., `arXiv:1303.3997`), you must convert it to `https://arxiv.org/abs/1303.3997`.
        3.  **Obtain via Search:** If only text information (like the title) is available, you **must** perform an additional search (e.g., `web_searcher(query='"Paper Title"')`) to find an authoritative link (from the journal's website, PubMed, etc.).

4.  **Fallback Plan**:
    *   If no citation information can be found on the homepage, search directly online using `"[tool_name] paper"`. After finding results, follow the rules in **Step 2.2** and **2.3**.

5.  **Final Handling**:
    *   If all methods fail to find any paper, you must clearly state "Not found" in the "Foundational Paper" section.
</Instructions>


<Example>
**User Input:** `samtools`

**Agent's Ideal Internal Thought and Action Process:**
1.  **Thought:** My task is to find the homepage and the original paper for `samtools`. I will start by searching GitHub.
2.  **Action:** `web_searcher(query="samtools github")`
3.  **Thought:** The first result, `github.com/samtools/samtools`, looks official. I need to validate it.
4.  **Action:** `web_reader(url="https://github.com/samtools/samtools")`
5.  **Thought:** Reading the README, the description includes "Tools for ... SAM/BAM/CRAM format," which is clearly bioinformatics-related. Validation passed. The homepage is confirmed.
6.  **Thought:** Now I will look for a "Citing samtools" section in the README. I found it.
7.  **Thought:** The section lists two papers, one from 2009 and one from 2021. According to the rules, I must choose the earliest one: the 2009 paper, "The Sequence Alignment/Map format and SAMtools".
8.  **Thought:** The paper provides a DOI: `10.1093/bioinformatics/btp352`. I must convert this into an accessible link: `https://doi.org/10.1093/bioinformatics/btp352`.
9.  **Thought:** I have all the necessary information. I will now output it in the required format.

**Final Result:**
Homepage: https://github.com/samtools/samtools
Foundational Paper: https://doi.org/10.1093/bioinformatics/btp352
</Example>
"""

SEARCH_IMAGE_SYS = """# Role and Goal
You are an expert AI assistant responsible for automatically finding, selecting, and saving the most suitable Docker image for a given tool. You must strictly follow a predefined workflow and set of rules to complete this task efficiently and accurately. Your final objective is to call the `pull_and_save_image` tool with all a-required, correctly formatted parameters.

# Available Tools
- `search_image`: Searches for public Docker images based on a tool's name.
- `view_image_tags`: Lists all available tags for a specific Docker image.
- `web_reader`: Reads and understands the content of a webpage, primarily used for accessing the tool's official homepage.
- `pull_and_save_image`: Pulls a Docker image and saves its metadata to a database.
- `ask_human`: Requests human intervention when you cannot resolve the problem independently.

# Core Workflow
You must execute your tasks following this logical sequence:

1.  **Initial Search**: Use the `search_image` tool based on the user-provided tool name.
2.  **Analyze and Filter Images**:
    *   **If `search_image` returns multiple results**: **You must first filter them to select the single most credible image** before proceeding. Follow the **[Image Selection Rules]** below to make your decision.
    *   **If `search_image` returns only one credible result**: Proceed directly to the next step.
    *   **If `search_image` returns no results**: Switch strategy. Use `web_reader` on the tool's homepage to find an official image URL. If found, treat it as the sole candidate and proceed.
3.  **View and Select Tag**: Once a single image is identified, use `view_image_tags` to inspect all its available tags. Then, select the most appropriate tag according to the **[Image Tag Selection Rules]**.
4.  **Gather Information**: After determining the final image and tag, use `web_reader` on the tool's homepage to find the official description for the `description` parameter.
5.  **Final Execution**:
    *   **If successful**: Assemble all parameters (`name`, `image_url`, `version`, `description`) and call `pull_and_save_image`.
    *   **If all methods fail**: Call `ask_human`, clearly explaining the steps you've taken.

# Detailed Instructions and Constraints
You must strictly adhere to the following rules:

**[Image Selection Rules] (For filtering multiple search results)**
1.  **Prioritize Official or Trusted Sources**: Give precedence to images marked as "Official Image" or from well-known, trusted organizations (e.g., `biocontainers`).
2.  **Evaluate Name Match and Pull Count**: Among credible candidates, select the one whose name most closely matches the tool's name and has a **significantly higher pull count**. A high pull count is a strong indicator of community trust and widespread use.
3.  **Make a Final Decision**: Your goal is to narrow down the candidates to **only one** single, most appropriate image before calling `view_image_tags`.

**[Image Tag Selection Rules] (For filtering multiple tags of a single image)**
1.  **Latest Version First**: Filter for tags corresponding to the highest version number.
2.  **Priorities for Same Version**: If the latest version has multiple tags, choose in this order:
    *   a. **Prefer Base Image**: Select the base image without extra services (e.g., databases).
    *   b. **Prefer Newer Build Environment**: Choose the newer environment if specified (e.g., `py312` is better than `py39`).
    *   c. **Prefer Later Publication**: If still tied, choose the most recently published one.

**[`pull_and_save_image` Parameter Details]**
*   `name`: The tool's name, as provided by the user.
*   `image_url`: The full image URL (`[registry]/[namespace]/[repository]:tag`), including the specific tag you selected.
*   `version`: The **clean version number** extracted from the selected image tag (e.g., extract `1.10.3` from `1.10.3--h45fbf2d_5`). This version **must** come from the tag, not the homepage.
*   `description`: The tool's official description.
    *   **Must** be obtained using `web_reader` from the tool's homepage.
    *   **You must quote verbatim the core introductory text about the tool's function. Do not summarize, abbreviate, or rephrase it.**
    *   This description is about the **tool's purpose**, not your current task.
"""


class ToolInfoSchema(BaseModel):
    homepage: str = Field(description="Tool's homepage, prioritizing the GitHub source code repository")
    paper_url: Optional[str] = Field(description="Link to the tool's publication, leave blank if not applicable")


class GetImageAgent:
    def __init__(self):
        self.llm = gpt_5_mini(temperature=0.3)

        self.tools = [search_local_image, select_image, web_searcher, web_reader, need_new_image]
        self.search_tool_node = ToolNode(self.tools)

        # subagents
        self.search_info_agent = create_agent(
            model=self.llm,
            tools=[web_searcher, web_reader],
            system_prompt=SEARCH_HOMEPAGE_SYS,
            middleware=[web_summary_middleware],
            response_format=ToolInfoSchema,
        )
        self.search_image_agent = create_agent(
            model=self.llm,
            tools=[search_image, view_image_tags, web_reader, pull_and_save_image, ask_human],
            system_prompt=SEARCH_IMAGE_SYS,
            state_schema=SearchImageState,
            context_schema=CreateImageContext,
            middleware=[
                SummarizationMiddleware(
                    model=gemini_25_flash(),
                    max_tokens_before_summary=6000,
                    messages_to_keep=6,
                )
            ],
        )

    async def searcher_node(self, state: SearchImageState):
        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=SEARCHER_SYSTEM), MessagesPlaceholder(variable_name='history')]
        )
        llm = self.llm.bind_tools(self.tools)
        chain = prompt | llm
        response = await chain.ainvoke({'history': state['messages']})
        return {'messages': [response]}

    @staticmethod
    async def fix_tool_call_node(state):
        return {'messages': [HumanMessage(content='Continue to call tools to complete the task')]}

    @staticmethod
    def searcher_router(state: SearchImageState) -> Literal['search_tool', 'continue_search', '__end__']:
        last_message = state['messages'][-1]
        assert isinstance(last_message, AIMessage)

        if last_message.tool_calls:
            return 'search_tool'

        if not state['image_uid']:
            return 'continue_search'

        return END

    @staticmethod
    def search_tool_router(state: SearchImageState) -> Literal['searcher', 'homepage_searcher']:
        last_message = state['messages'][-1]

        if isinstance(last_message, HumanMessage):
            return 'homepage_searcher'

        return 'searcher'

    async def homepage_searcher_node(self, state: SearchImageState):
        response = await self.search_info_agent.ainvoke({'messages': state['messages']})
        return {
            'messages': [
                HumanMessage(content=f'The homepage of the tool is {response["structured_response"].homepage}')
            ],
            'homepage': response['structured_response'].homepage,
            'paper_link': response['structured_response'].paper_url,
        }

    async def image_searcher_node(self, state: SearchImageState):
        response = await self.search_image_agent.ainvoke(
            {'messages': state['messages']},
            context=CreateImageContext(homepage=state['homepage'], paper_link=state['paper_link']),
        )
        return {
            'messages': [response['messages'][-1]],
            'image_uid': response['image_uid'],
            'image': response['image'],
        }

    def build(self, visualize: bool = False):
        graph = StateGraph(state_schema=SearchImageState)
        graph.add_node('searcher', self.searcher_node)
        graph.add_node('search_tool', self.search_tool_node)
        graph.add_node('continue_search', self.fix_tool_call_node)
        graph.add_node('homepage_searcher', self.homepage_searcher_node)
        graph.add_node('image_searcher', self.image_searcher_node)

        graph.set_entry_point('searcher')
        graph.add_conditional_edges('searcher', self.searcher_router)
        graph.add_conditional_edges('search_tool', self.search_tool_router)
        graph.add_edge('continue_search', 'searcher')
        graph.add_edge('homepage_searcher', 'image_searcher')
        graph.add_edge('image_searcher', END)

        compiled_graph = graph.compile(name='get_image_agent')

        if visualize:
            mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with open('images/create_image_agent.md', 'w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        return compiled_graph


@tool
def need_new_image(name: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Initiate a request to create a new image

    Args:
        name: The name of the tool for which a new image needs to be created. Must be a verified correct name.
    """

    return Command(
        update={
            'messages': [
                ToolMessage(content='', tool_call_id=tool_call_id),
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage(content=f'Need to create a new image for tool: {name}'),
            ]
        },
    )
