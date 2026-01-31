from typing import Literal, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langchain_core.tools import tool
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlmodel import Session

import app.crud.tools.image as image_crud
from app.core.redis_client import llm_cache
from app.core.session import engine
from app.models.tools import ImageConfig, ToolImage
from llm.models import get_agent_model, get_long_context_model
from llm.states import SearchImageState
from llm.tools.database import search_local_image, select_image
from llm.tools.human import ask_human
from llm.tools.images import search_image, view_image_tags
from llm.tools.web_search import web_reader, web_searcher, web_summary_middleware
from llm.utils.middleware import DisableParallelToolCallsMiddleware, TokenStatisticMiddleware

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

SEARCH_INFO_SYS = """You are an AI assistant specializing in the field of bioinformatics. 
Your task is to identify the bioinformatics command-line tools mentioned by the user, call the tool for an internet search, and finally output the details of this tool in the specified format.

<AvailableTools>
- `web_searcher`: Takes a string as a search query and returns a list of search results.
- `web_reader`: Takes a URL and returns the text content of that webpage.

To save tokens, all tools are forbidden from parallel invocation, meaning **you can only call one tool at a time**.
</AvailableTools>

<ToolInfoSchema>
Your final output needs to be a JSON string containing the following fields:

-   `name`: The name of the tool
    -   If the input is a subcommand (e.g., `samtools view`), you must return the parent tool's name (`samtools`).
    -   If an official abbreviation exists (e.g., `GATK` for `Genome Analysis Toolkit`), you **must** use it.
    -   Use the exact name from the official source, respecting its case and spelling.
-   `description`: The description of this tool
    -   You **must not** summarize or rephrase. Find the official, one-paragraph description from the tool's homepage or the top of its `README.md` and quote it exactly.
-   `document_url`: Tool documentation page URL
    -   If a GitHub repository exists, begin your search there. It's the best central point.
    -   From the repository, check the "About" section, links in the `README.md`, and the github wiki tab to find all potential documentation sites.
    -   Choose the single best URL. A dedicated documentation site (e.g., `docs.tool.com` or `tool.github.io`) is usually preferable to a `README`, unless the dedicated site is clearly outdated.
-   `paper_url`: Link to the corresponding paper of the tool
    -   Look for a "How to Cite," "Citation," or "Publication" section on the tool's official website or GitHub `README.md`.
    -   If multiple papers are listed, choose the one that originally introduced the tool (usually the oldest).
    -   Provide a direct URL to the paper, preferably a DOI link (e.g., `https://doi.org/...`).
    -   You should **not** use `web_reader` on the paper's URL. Verifying the paper's relevance from the search result title and snippet is sufficient.
    -   **Stop if Not Provided**: If the authors do not provide a citation link on their official sites, or after several searches no usable links were found, leave the field empty. Do not perform a separate web search to find a paper.
</ToolInfoSchema>

<Example>
**Example 1: User input is a command-line tool (with an official abbreviation)**

**User Input:**
"Search about `Genome Analysis Toolkit (GATK)`, and here is used for performing base recalibration on sequencing data"

**Expected Output:**
```json
{{
  "name": "GATK",
  "description": "The Genome Analysis Toolkit or GATK is a bioinformatics software package for analyzing next-generation DNA sequencing data. The toolkit offers a wide variety of tools with a primary focus on variant discovery and genotyping.",
  "document_url": "https://gatk.broadinstitute.org/hc/en-us",
  "paper_url": "http://biorxiv.org/lookup/doi/10.1101/201178"
}}
```
</Example>

<Example>
**Example 2: User input is a subcommand**

**User Input:**
"Search about `samtools view`, and here is used for viewing and converting SAM/BAM files"

**Expected Output:**
```json
{{
  "name": "samtools",
  "description": "Samtools is a suite of programs for interacting with high-throughput sequencing data. It consists of three separate repositories: samtools, bcftools, htslib.",
  "document_url": "https://www.htslib.org/doc/samtools.html",
  "paper_url": "https://doi.org/10.1093/gigascience/giab008"
}}
```
</Example>
"""

SEARCH_IMAGE_SYS = """You are an expert agent designed to find the most suitable and available container image for a given software tool.
Your primary task is to process information about a tool, search for its container image, and return the result in a structured format.


<AvailableTools>
You have access to the following specialized tools to accomplish your task:
*   `search_image`: Use this tool to search for images within the Biocontainers repository based on the tool's name.
*   `view_image_tags`: Use this tool to retrieve all available tags for a specific image found in Biocontainers.
*   `web_reader`: Use this tool to read the content of a given URL, specifically for finding official image information in the tool's documentation.
*   `ask_human`: When you have used multiple query methods and still haven't found an available mirror, call this tool to request further information from a human.

To save tokens, all tools are forbidden from parallel invocation, meaning **you can only call one tool at a time**.
</AvailableTools>


<ChainOfThought>
You should follow a structured reasoning process to ensure accuracy and efficiency:
1.  **Analyze**: Carefully examine the input `<ToolInformation>` (name, description, document_url).
2.  **Prioritize Biocontainers**: Your first action is always to use `search_image` with the tool's name.
3.  **Evaluate Search Results**:
    *   If no results are found, proceed directly to step 6 (Consult Documentation).
    *   If one result is found, verify its relevance and proceed to step 5 (Get Tags).
    *   If multiple results are found, apply the **[Image Selection Rules]** to identify the single best match. If no match is satisfactory, proceed to step 6.
4.  **Get Tags**: Once a suitable image is identified, use `view_image_tags` to fetch all its tags.
5.  **Select Best Tag**: Apply the **[Image Tag Selection Rules]** to the list of tags to choose the most appropriate one. Once selected, proceed to step 7 (Format Output).
6.  **Consult Documentation**: This step is a fallback. Use `web_reader` with the `document_url` only if the Biocontainers search was unsuccessful. Look for an officially provided or recommended Docker image. 
    *   If you find one, attempt to determine its version and best tag, then proceed to step 7.
    *   If still no suitable result is found, call the `ask_human` tool and make further decisions based on the human feedback.
7.  **Format Output**: Based on your findings, construct the final JSON object according to the **[Output Structure]**. If at any point you conclude no suitable image exists from either source, ensure the output fields are empty.
</ChainOfThought>


<DetailInstructions>
*   **Biocontainers First**: Always start by searching Biocontainers. Do not use the `web_reader` unless the Biocontainers search yields no relevant results.
*   **No Image Found**: If you have exhausted both the Biocontainers search and the official documentation via `web_reader` without finding a usable image, you **must** return an empty result. Do not fabricate or guess an image URL.
*   **Be Decisive**: Your goal is to select one single best image and tag, not to provide a list of options.

### **[Image Selection Rules]** (For filtering multiple search results)
When `search_image` returns multiple potential images, apply these rules in order to select the single most appropriate one:
1.  **Name & Description Match**: The image's name and description must have the highest relevance to the tool's name and description from the input.
2.  **Pull Count**: Prefer the image with a significantly higher number of pulls, as this indicates it is more widely used and trusted.
3.  **Relevance**: If pull counts are similar or low, re-evaluate the description to ensure it perfectly matches the intended tool.

### **[Image Tag Selection Rules]** (For filtering multiple tags of a single image)
After selecting an image and retrieving its tags with `view_image_tags`, apply these rules in order to select the single best tag:
1.  **Latest Tool Version**: Prioritize the tag corresponding to the newest stable version of the tool (e.g., choose `v2.5.1` over `v2.4.0`).
2.  **Base Image First**: If a single version has multiple builds, prefer the base image. Base images typically have simpler tags and smaller file sizes, avoiding extra dependencies like databases (e.g., prefer `...-py3` over `...-py3-db12`).
3.  **Newer Build Environment**: If multiple base images exist for the same tool version, choose the one with a more recent build environment (e.g., prefer `...-py12` over `...-py9`).
4.  **Latest Publication Date**: If all other factors are equal, select the tag that was published most recently.
</DetailInstructions>


<OutputStructure>
Your final output must be a JSON object that strictly conforms to the `ToolImageOuput` model.
*   `image_url`: The full Docker image URL for the tool, which must follow the `[registry]/[namespace]/[repository]:tag` format
    *   Note that images from `docker.io` sometimes omit the registry; you must remember to add it.
    *   **Example**: `quay.io/biocontainers/samtools:1.22.1--h96c455f_0`
*   `version`: The tool's own version number(string).
    *   This version should be extracted from the image tag and must be distinguished from the full tag, which may contain additional build information (like environment, build number, etc.).
    *   **Example**: For the tag `1.22.1--h96c455f_0`, the corresponding version should be `1.22.1`.
</OutputStructure>
"""

SEARCH_IMAGE_SYSTEM = """You are a bioinformatics data engineer. Your task is to find the corresponding tool's image based on the user's needs and return it.

According to the tool search results, if there is an image of the tool required by the user, use `select_image` to select it;
Otherwise, end the task directly and tell the user that the image does not exist.
"""


class ToolInfoOutput(BaseModel):
    name: str = Field(description='The name of the tool')
    description: str = Field(description='The description of this tool')
    document_url: Optional[str] = Field(description='Tool documentation page URL', default=None)
    paper_url: Optional[str] = Field(description='Link to the corresponding paper of the tool', default=None)


class ToolImageOutput(BaseModel):
    image_url: str = Field(
        description='The Docker image of the tool, in the format `[registry]/[namespace]/[repository]:tag`',
        default=None,
    )
    version: str = Field(description='The version of the tool', default=None)


class GetImageAgent:
    def __init__(self):
        self.llm = get_agent_model()

        self.tools = [search_local_image, select_image, web_searcher, web_reader, need_new_image]
        self.search_tool_node = ToolNode(self.tools)

        # subagents
        self.search_info_agent = create_agent(
            model=self.llm,
            tools=[web_searcher, web_reader],
            system_prompt=SEARCH_INFO_SYS,
            middleware=[
                web_summary_middleware,
                TokenStatisticMiddleware(agent_name='search_info', **self.llm.metadata),
            ],
            response_format=ToolInfoOutput,
        )
        self.search_image_agent = create_agent(
            model=self.llm,
            tools=[search_image, view_image_tags, web_reader, ask_human],
            system_prompt=SEARCH_IMAGE_SYS,
            middleware=[  # type: ignore
                SummarizationMiddleware(
                    model=get_long_context_model(),
                    max_tokens_before_summary=10000,
                    messages_to_keep=6,
                )
            ],
            response_format=ToolImageOutput,
        )

    async def local_searcher_node(self, state: SearchImageState):
        """尝试搜索本地数据库，是否已经有保存的镜像"""
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

    async def info_searcher_node(self, state: SearchImageState):
        response = await self.search_info_agent.ainvoke({'messages': state['messages']})  # type: ignore
        tool_info: ToolInfoOutput = response['structured_response']
        msg_content = (
            f'<ToolInformation>\n'
            f'name: {tool_info.name}\n'
            f'description: {tool_info.description}\n'
            f'document_url: {tool_info.document_url}\n'
            f'</ToolInformation>'
        )
        return {
            'messages': [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage(content=msg_content),
            ],
            'name': tool_info.name,
            'description': tool_info.description,
            'homepage': tool_info.document_url,
            'paper_link': tool_info.paper_url,
        }

    async def image_searcher_node(self, state: SearchImageState):
        response = await self.search_image_agent.ainvoke({'messages': state['messages']})
        tool_image: ToolImageOutput = response['structured_response']

        with Session(engine) as session:
            new_image = ToolImage(
                name=state['name'],
                image=ImageConfig.from_str(tool_image.image_url),
                version=tool_image.version,
                description=state['description'],
                homepage=state['homepage'],
                paper_link=state['paper_link'],
            )
            new_image = image_crud.insert(session, new_image)
            image_uid = new_image.uid

        return {
            'image_uid': image_uid,
            'image': new_image.image,
        }

    @staticmethod
    def searcher_router(state: SearchImageState) -> Literal['local_search_tool', 'continue_search', '__end__']:
        last_message = state['messages'][-1]
        assert isinstance(last_message, AIMessage)

        if last_message.tool_calls:
            return 'local_search_tool'

        if 'image_uid' not in state or not state['image_uid']:
            return 'continue_search'

        return END

    @staticmethod
    def search_tool_router(state: SearchImageState) -> Literal['local_searcher', 'info_searcher']:
        last_message = state['messages'][-1]

        if isinstance(last_message, HumanMessage):
            return 'info_searcher'

        return 'local_searcher'

    def build(self, visualize: bool = False):
        graph = StateGraph(state_schema=SearchImageState)  # type: ignore
        graph.add_node('local_searcher', self.local_searcher_node)  # type: ignore
        graph.add_node('local_search_tool', self.search_tool_node)
        graph.add_node('continue_search', self.fix_tool_call_node)
        graph.add_node('info_searcher', self.info_searcher_node)  # type: ignore
        graph.add_node('image_searcher', self.image_searcher_node)  # type: ignore

        graph.set_entry_point('local_searcher')
        graph.add_conditional_edges('local_searcher', self.searcher_router)
        graph.add_conditional_edges('local_search_tool', self.search_tool_router)
        graph.add_edge('continue_search', 'local_searcher')
        graph.add_edge('info_searcher', 'image_searcher')
        graph.add_edge('image_searcher', END)

        compiled_graph = graph.compile(name='get_image_agent')

        if visualize:
            mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with open('images/create_image_agent.md', 'w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        return compiled_graph


@tool(parse_docstring=True)
def need_new_image(name: str, runtime: ToolRuntime):
    """Initiate a request to create a new image

    Args:
        name: The name of the tool for which a new image needs to be created. Must be a verified correct name.
    """

    return Command(
        update={
            'messages': [
                ToolMessage(content='', tool_call_id=runtime.tool_call_id),
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage(content=f'Need to create a new image for tool: {name}'),
            ]
        },
    )


search_image_agent = create_agent(
    model=get_agent_model(),
    tools=[
        search_local_image,
        select_image,
    ],
    system_prompt=SEARCH_IMAGE_SYSTEM,
    middleware=[  # type: ignore
        DisableParallelToolCallsMiddleware(),
        TokenStatisticMiddleware(agent_name='search_image', **get_agent_model().metadata),
    ],
    state_schema=SearchImageState,  # type: ignore
    cache=llm_cache,  # type: ignore
)
