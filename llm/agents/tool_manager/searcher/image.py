from pathlib import Path
from typing import Any, Literal, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables.graph import CurveStyle
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlmodel import Session

import app.crud.tools.image as image_crud
from app.core.session import engine
from app.models.tools import ImageConfig, ToolImage
from llm.models import get_agent_model, get_long_context_model
from llm.states import SearchImageState
from llm.tools.database import search_local_image, select_image
from llm.tools.human import ask_human
from llm.tools.images import search_image, view_image_tags
from llm.tools.web_search import web_reader, web_searcher, web_summary_middleware
from llm.utils.middleware import DisableParallelToolCallsMiddleware, TokenStatisticMiddleware

SEARCH_INFO_SYS = """# Role and Task
You are an AI assistant specializing in the field of bioinformatics.
Your primary mission is to identify bioinformatics **command-line tools** mentioned by the user and to gather accurate, core information about them from the internet.

# Output Schemas
You must strictly adhere to one of the following two JSON formats for your output:

1.  **[ToolInfoSchema]**: When the input is a command-line tool
    ```json
    {{
      "tool_info": {{
        "name": "The official tool name",
        "description": "A concise description from the official source.",
        "document_url": "The URL for the tool's documentation.",
        "paper_url": "The URL for the tool's primary publication."
      }}
    }}
    ```

2.  **[NotToolSchema]**: When the input is not a command-line tool
    ```json
    {{
      "tool_info": {{
        "correct_type": "r_package | python_package | software | other",
        "reason": "A brief explanation for your classification."
      }}
    }}
    ```

# Workflow
1.  **Initial Search & Classification**: Use `web_searcher` to perform a preliminary search. Based on the results (e.g., presence of a Bioconductor/CRAN page, a PyPI page, or descriptions of a GUI), determine if the input refers to a command-line tool.
    - Note that due to data extraction errors, the input here might not be an analytical tool at all, but rather a testing platform, reagents, species names, and so on.
2.  **Route to Schema**:
    - If it is **not** a command-line tool, determine its `correct_type`, formulate the `reason`, and generate the output using `NotToolSchema`.
    - If it **is** a command-line tool, proceed to the next step.
3.  **Gather Detailed Information**: Sequentially use your tools to find the information for each field (`name`, `description`, `document_url`, `paper_url`), strictly following the **`Information Gathering Rules`** outlined below.
4.  **Format and Output**: Assemble the collected information into the `ToolInfoSchema` and return the final JSON.

# Information Gathering Rules
-   **For `name`**:
    -   **Identify Parent Tool**: If the input is a subcommand (e.g., `samtools view`), you must return the parent tool's name (`samtools`).
    -   **Prioritize Official Abbreviation**: If an official abbreviation exists (e.g., `GATK` for `Genome Analysis Toolkit`), you **must** use it.
    -   **Ensure Accuracy**: Use the exact name from the official source, respecting its case and spelling.
-   **For `description`**:
    -   **Quote Verbatim**: You **must not** summarize or rephrase. Find the official, one-paragraph description from the tool's homepage or the top of its `README.md` and quote it exactly.
-   **For `document_url`**:
    -   **Start with GitHub**: If a GitHub repository exists, begin your search there. It's the best central point.
    -   **Find the Best Source**: From the repository, check the "About" section, links in the `README.md`, and the github wiki tab to find all potential documentation sites.
    -   **Select the Most Comprehensive & Current URL**: Choose the single best URL. A dedicated documentation site (e.g., `docs.tool.com` or `tool.github.io`) is usually preferable to a `README`, unless the dedicated site is clearly outdated.
-   **For `paper_url`**:
    -   **Find Official Citation**: Look for a "How to Cite," "Citation," or "Publication" section on the tool's official website or GitHub `README.md`.
    -   **Select the Foundational Paper**: If multiple papers are listed, choose the one that originally introduced the tool (usually the oldest).
    -   **Must Be a Direct Link**: Provide a direct URL to the paper, preferably a DOI link (e.g., `https://doi.org/...`).
    -   **Stop if Not Provided**: If the authors do not provide a citation link on their official sites, leave the field empty or do not include it. Do not perform a separate web search to find a paper.
    -   **Do Not Read Paywalled Content**: You should **not** use `web_reader` on the paper's URL. Verifying the paper's relevance from the search result title and snippet is sufficient.

<Example>
**Example 1: User input is a command-line tool (with an official abbreviation)**

**User Input:**
"Search about `Genome Analysis Toolkit (GATK)`, and here is used for performing base recalibration on sequencing data"

**Expected Output:**
```json
{{
  "tool_info": {{
    "name": "GATK",
    "description": "The Genome Analysis Toolkit or GATK is a bioinformatics software package for analyzing next-generation DNA sequencing data. The toolkit offers a wide variety of tools with a primary focus on variant discovery and genotyping.",
    "document_url": "https://gatk.broadinstitute.org/hc/en-us",
    "paper_url": "http://biorxiv.org/lookup/doi/10.1101/201178"
  }}
}}
```</Example>

<Example>
**Example 2: User input is a subcommand**

**User Input:**
"Search about `samtools view`, and here is used for viewing and converting SAM/BAM files"

**Expected Output:**
```json
{{
  "tool_info": {{
    "name": "samtools",
    "description": "Samtools is a suite of programs for interacting with high-throughput sequencing data. It consists of three separate repositories: samtools, bcftools, htslib.",
    "document_url": "https://www.htslib.org/doc/samtools.html",
    "paper_url": "https://doi.org/10.1093/gigascience/giab008"
  }}
}}
```
</Example>

<Example>
**Example 3: User input is an R package**

**User Input:**
"Search about `DESeq2`, and here is used for differential gene expression analysis using a pseudobulk approach"

**Expected Output:**```json
{{
  "tool_info": {{
    "correct_type": "r_package",
    "reason": "DESeq2 is a package for the R programming environment, available through Bioconductor. It is not a standalone command-line tool but is executed within an R session."
  }}
}}
```
</Example>


<Example>
**Example 4: User entered other unrelated things**

**User Input:**
"Search about `Illumina NovaSeq 6000`, and here is used for performing high-throughput sequencing of single-nucleus RNA-seq libraries"

**Expected Output:**```json
{{
  "tool_info": {{
    "correct_type": "other",
    "reason": "The Illumina NovaSeq 6000 is a high-throughput gene sequencer, not an analysis tool."
  }}
}}
```
</Example>
"""

SEARCH_IMAGE_SYS = """# Role and Task
You are an expert agent designed to find the most suitable and available container image for a given software tool. Your primary task is to process information about a tool, search for its container image, and return the result in a structured format.

# Available Tools
You have access to the following specialized tools to accomplish your task:
*   `search_image`: Use this tool to search for images within the Biocontainers repository based on the tool's name.
*   `view_image_tags`: Use this tool to retrieve all available tags for a specific image found in Biocontainers.
*   `web_reader`: Use this tool to read the content of a given URL, specifically for finding official image information in the tool's documentation.

To save tokens, all tools are forbidden from parallel invocation, meaning **you can only call one tool at a time**.

# Chain of Thought
You should follow a structured reasoning process to ensure accuracy and efficiency:
1.  **Analyze**: Carefully examine the input `<ToolInformation>` (name, description, document_url).
2.  **Prioritize Biocontainers**: Your first action is always to use `search_image` with the tool's name.
3.  **Evaluate Search Results**:
    *   If no results are found, proceed directly to step 6 (Consult Documentation).
    *   If one result is found, verify its relevance and proceed to step 5 (Get Tags).
    *   If multiple results are found, apply the **[Image Selection Rules]** to identify the single best match. If no match is satisfactory, proceed to step 6.
4.  **Get Tags**: Once a suitable image is identified, use `view_image_tags` to fetch all its tags.
5.  **Select Best Tag**: Apply the **[Image Tag Selection Rules]** to the list of tags to choose the most appropriate one. Once selected, proceed to step 7 (Format Output).
6.  **Consult Documentation**: This step is a fallback. Use `web_reader` with the `document_url` only if the Biocontainers search was unsuccessful. Look for an officially provided or recommended Docker image. If you find one, attempt to determine its version and best tag, then proceed to step 7.
7.  **Format Output**: Based on your findings, construct the final JSON object according to the **[Output Structure]**. If at any point you conclude no suitable image exists from either source, ensure the output fields are empty.

# Detailed Instructions
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

# Output Structure
Your final output must be a JSON object that strictly conforms to the `ToolImageOuput` model.
*   `image_url`
    *   **Description**: The full Docker image URL for the tool, which must follow the `[registry]/[namespace]/[repository]:tag` format. Note that images from `docker.io` sometimes omit the registry; you must remember to add it.
    *   **Example**: `quay.io/biocontainers/samtools:1.22.1--h96c455f_0`
*   `version`
    *   **Description**: The tool's own version number. This version should be extracted from the image tag and must be distinguished from the full tag, which may contain additional build information (like environment, build number, etc.).
    *   **Example**: For the tag `1.22.1--h96c455f_0`, the corresponding version should be `1.22.1`.
"""

SEARCH_IMAGE_SYSTEM = """You are a bioinformatics data engineer. Your task is to find the corresponding tool's image based on the user's needs and return it.

According to the tool search results, if there is an image of the tool required by the user, use `select_image` to select it;
Otherwise, end the task directly and tell the user that the image does not exist.
"""

SELF_SEARCH_IMAGE_SYSTEM = """You are a bioinformatics data engineer. Your task is to find the corresponding tool's image based on the user's needs and return it.

According to the tool search results, if there is an image of the tool required by the user, use `select_image` to select it;
Otherwise, call the `ask_human` tool to request the user to add an available image.
"""


class ToolInfoSchema(BaseModel):
    name: str = Field(description="The name of the tool, uniformly use the official name found on the tool's homepage")
    description: str = Field(
        description="For a description of this tool, you can refer to the official website's description of it (usually found at the very beginning of the README or homepage)."
    )
    document_url: Optional[str] = Field(description='Tool documentation page URL', default=None)
    paper_url: Optional[str] = Field(
        description='Link to the corresponding paper of the tool. Prefer the published version over the preprint. If there are multiple papers, prefer the earliest published one.',
        default=None,
    )


class NotToolSchema(BaseModel):
    correct_type: Literal['r_package', 'python_package', 'software', 'other']
    reason: str = Field(description='The reason for your incorrect type judgment')


class ToolInfoOutput(BaseModel):
    info: ToolInfoSchema | NotToolSchema = Field(
        description='Return information in different formats depending on different situations.'
    )


class ToolImageOutput(BaseModel):
    image_url: Optional[str] = Field(
        description='The Docker image of the tool, in the format `[registry]/[namespace]/[repository]:tag`. Note that docker.io images sometimes omit the registry, remember to add it.',
        default=None,
    )
    version: Optional[str] = Field(
        description='The version of the tool, note that it should be distinguished from the image tag (the latter may include some additional characters)',
        default=None,
    )


class GetImageAgent:
    def __init__(self) -> None:
        self.llm = get_agent_model()

        # subagents
        self.search_info_agent = create_agent(
            model=self.llm,
            tools=[web_searcher, web_reader],
            system_prompt=SEARCH_INFO_SYS,
            middleware=[
                web_summary_middleware,
                TokenStatisticMiddleware(agent_name='search_info', **self.llm.metadata),
            ],
            state_schema=SearchImageState,
            response_format=ToolInfoOutput,
        )
        self.search_remote_image_agent = create_agent(
            model=self.llm,
            tools=[search_image, view_image_tags, web_reader, ask_human],
            system_prompt=SEARCH_IMAGE_SYS,
            middleware=[
                SummarizationMiddleware(
                    model=get_long_context_model(),
                    max_tokens_before_summary=10000,
                    messages_to_keep=6,
                )
            ],
            state_schema=SearchImageState,
            response_format=ToolImageOutput,
        )

    async def local_searcher_node(self, state: SearchImageState) -> Command[Literal['info_searcher', '__end__']]:
        """尝试搜索本地数据库, 是否已经有保存的镜像."""
        response = await search_image_agent.ainvoke({'history': state['messages']})
        if response.get('image_uid'):
            return Command(
                goto=END,
                update={
                    'image_uid': response['image_uid'],
                    'image': response['image'],
                },
            )
        return Command(goto='info_searcher')

    async def info_searcher_node(self, state: SearchImageState) -> Command[Literal['image_searcher', '__end__']]:
        """搜索工具基础信息."""
        response = await self.search_info_agent.ainvoke({'messages': state['messages']})  # type: ignore
        tool_info: ToolInfoOutput = response['structured_response']
        if isinstance(tool_info.info, ToolInfoSchema):
            msg_content = (
                f'<ToolInformation>\n'
                f'name: {tool_info.info.name}\n'
                f'description: {tool_info.info.description}\n'
                f'document_url: {tool_info.info.document_url}\n'
                f'</ToolInformation>'
            )
            return Command(
                goto='image_searcher',
                update={
                    'messages': [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES),
                        HumanMessage(content=msg_content),
                    ],
                    'name': tool_info.info.name,
                    'description': tool_info.info.description,
                    'homepage': tool_info.info.document_url,
                    'paper_link': tool_info.info.paper_url,
                },
            )

        return Command(goto=END, update={'messages': [AIMessage(content=tool_info.info.reason)]})

    async def image_searcher_node(self, state: SearchImageState) -> dict[str, Any]:
        """搜索工具镜像."""
        response = await self.search_remote_image_agent.ainvoke({'messages': state['messages']})
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

    def build(self, *, visualize: bool = False) -> CompiledStateGraph[SearchImageState]:  # type: ignore
        """搜索新工具镜像."""
        graph = StateGraph(state_schema=SearchImageState)  # type: ignore
        graph.add_node('local_searcher_node', self.local_searcher_node)  # type: ignore
        graph.add_node('info_searcher', self.info_searcher_node)  # type: ignore
        graph.add_node('image_searcher', self.image_searcher_node)  # type: ignore

        graph.set_entry_point('local_searcher_node')
        graph.add_edge('image_searcher', END)

        compiled_graph = graph.compile(name='get_image_agent')

        if visualize:
            mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with Path('images/create_image_agent.md').open('w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        return compiled_graph


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
)

# TODO: 目前是手动补充
self_search_image_agent = create_agent(
    model=get_agent_model(),
    tools=[
        search_local_image,
        select_image,
        ask_human,
    ],
    system_prompt=SELF_SEARCH_IMAGE_SYSTEM,
    middleware=[  # type: ignore
        DisableParallelToolCallsMiddleware(),
        TokenStatisticMiddleware(agent_name='search_image', **get_agent_model().metadata),
    ],
    state_schema=SearchImageState,  # type: ignore
)
