from llm.agents.tool_agent.report_generator_v2 import direct_test_run_agent
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import CurveStyle
from langchain_milvus import BM25BuiltInFunction, Milvus
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field
from sqlmodel import Session

import app.crud.tools.image as image_crud
import app.crud.tools.tool as tool_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool_tag as tag_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.tools import ToolDefine, ToolType
from app.schemas.tools import ToolNodePublic
from llm.agents.tool_agent.config_generator import config_generate_agent
from llm.agents.tool_agent.report_generator import test_run_agent
from llm.agents.tool_agent.searcher.doc import doc_searcher
from llm.agents.tool_agent.searcher.image import self_search_image_agent
from llm.agents.tool_agent.searcher.tool_node import search_exist_tool_agent
from llm.models import get_agent_model, qwen3_embedding
from llm.states import (
    CreateToolContext,
    CreateToolOutput,
    CreateToolState,
    SearchImageState,
    SelectGroupState,
    TestRunContext,
    TestRunState,
    UserContext,
)
from llm.tools.database import (
    select_group,
    view_groups,
)
from llm.utils.formatters import handle_formatter, search_nodes_formatter
from llm.utils.middleware import TokenStatisticMiddleware
from llm.utils.uid_mapper import BidirectionalMapper

if TYPE_CHECKING:
    from llm.llm_client import LLMMetadata

SUMMARY_SYS = """You are an expert summarizer creating a concise log of previously completed web research to prevent redundant work.
Analyze the provided context of `web_searcher` and `web_reader` tool calls. Your task is to create a historical record of what has been searched and what was learned from the pages that were actually **read**.
Group the results under the specific search query that led to them.

**Strictly adhere to the following nested format for your output:**

*   **Query:** "[Insert the exact search query text here]"
    *   **Link:** [The URL that was read]
    *   **Extracted Information:** [A summary of the key information from that page. If the content was irrelevant or unhelpful, state "No useful information found."]

*If multiple pages were read for a single query, list each one as a separate "Link" entry under that query.*
"""

SELECT_IMAGE_SYSTEM = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, analysis data, etc., into individual nodes. Now, what we need to do is encapsulate a new node.
To achieve this, your first task is to **find available Docker images** based on the user's description, which will serve as the foundation for encapsulation.

<ThinkingProcess>
To find them, you just need to follow these steps:
First, use the `search_local_image` tool to search the local database for existing tool images.
    - If an image for the tool exists, directly use `select_image` with the exact image URL shown in the `<url>` field to select the corresponding image.
    - Otherwise, call `search_image` to search for images on Docker Hub, and based on the results, use the `pull_and_save_image` tool to pull and save the image.

If you encounter any unresolvable issues at any point, you can call the `ask_human` tool to ask for human assistance.
</ThinkingProcess>
"""

GET_DOC_SYSTEM = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, and analysis data into individual nodes, and what we need to do now is encapsulate a new node.
To achieve this, we have already found the corresponding image for the tool in previous work. Your task now is to **obtain the help documentation for the corresponding tool** based on the user's description, as a basis for encapsulation.

<ThinkingProcess>
To achieve your goal, you can follow these steps:
1. First, call the `list_docs` tool to view the tool documentation already recorded under the corresponding image.
2. If there are records in the returned results that you think might be useful, you can call the `view_doc` tool to view the specific content of that record.
    - If this record is suitable, call the `select_doc` tool to select this document and complete your task.
    - Otherwise, view other records as appropriate or proceed to the next step.
3. If the result of step 1 is empty or there are no suitable records, then you need to call the `run_command` tool to try to run commands within the corresponding container to get help documentation (see the detailed explanation in the <RunCommand> section).
4. `save_doc` will save the output from the last `run_command` execution to the database. After you get the desired help documentation, remember to call this tool to save the document and complete your task.

When you think a document can be selected or saved, **directly** call the tool to execute the operation.
When `run_command` fails multiple times or other unresolvable errors occur, you can call the `ask_human` tool to seek human intervention.
</ThinkingProcess>

<RunCommand>
When there is no corresponding help document in the database, you need to try running commands to get the tool's help document. In this step, you need to pay attention to:
- The tool is already running in a Docker container. You only need to provide the bare command-line instructions — do **not** include prefixes like `docker run`.
- If you receive an error like `command not found`, it indicates the tool name is incorrect. Try a new, correct tool name based on your knowledge.
    - If you can't find the name of the tool in your knowledge, try to remember if you have to add a suffix such as `.py`, `.sh`, etc.
</RunCommand>
"""


SELECT_GROUP_SYSTEM = """You are a bioinformatics analysis engineer using a node-based bioinformatics analysis platform.
This platform encapsulates analysis tools, databases, and analysis data into individual nodes, and what we need to do now is to encapsulate a new node.
Now your task is to **assign an appropriate group** to this tool based on the information available about it.

<ThinkingProcess>
To complete this task, you can follow these steps:
1. Call the `view_groups` tool to see the existing groups.
2. If there is a suitable group, directly call the `select_group` tool to select the corresponding group.
</ThinkingProcess>
"""


class ToolInfoSchema(BaseModel):
    homepage: str = Field(description="Tool's homepage, prioritizing the GitHub source code repository")
    paper_url: Optional[str] = Field(description="Link to the tool's publication, leave blank if not applicable")


class TempIDClass(BaseModel):
    uid: uuid.UUID


class CreateToolAgent:
    def __init__(self) -> None:
        self.think_llm = get_agent_model()

        # sub agents
        metadata = cast('LLMMetadata', self.think_llm.metadata)
        self.select_group_agent = create_agent(
            model=self.think_llm,
            tools=[view_groups, select_group],
            system_prompt=SELECT_GROUP_SYSTEM,
            middleware=[TokenStatisticMiddleware('select_group', **metadata)],
            state_schema=SelectGroupState,
            # pyrefly: ignore [bad-argument-type]
            context_schema=UserContext,
        )

    async def select_image_node(self, state: CreateToolState) -> dict[str, Any]:
        """首先搜索得到此工具的对应镜像."""
        if state.get('image_uid'):
            # 我们认为数据库中的镜像信息就是完全的
            with Session(engine) as session:
                tool_image = image_crud.get(session, str(state['image_uid']))
                return {'image': tool_image.image}
        else:
            # TODO: 重写镜像搜索
            writer = get_stream_writer()
            writer({'info': 'Searching for docker image...'})

            search_input: SearchImageState = {'messages': state['messages']}
            response = await self_search_image_agent.ainvoke(search_input)

            logger.success(f'Select image {response["image"]}')
            return {
                'image_uid': response['image_uid'],
                'image': response['image'],
            }

    async def get_doc_node(self, state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any] | None:
        """然后尝试获取指令对应的帮助文档."""
        if state.get('doc_uid'):
            return None
        writer = get_stream_writer()
        writer({'info': "Searching for tool's help document..."})

        # pyrefly: ignore [no-matching-overload]
        response = await doc_searcher.ainvoke(
            {
                'messages': state['messages'],
                'image_uid': state['image_uid'],
                'image': state['image'],
            },
            context=UserContext(user_id=runtime.context.user_id),
        )
        doc_uid = response['doc_uid']

        return {
            'doc_uid': doc_uid,
        }

    async def search_exist_tool_node(
        self, state: CreateToolState, runtime: Runtime[CreateToolContext]
    ) -> Command[Literal['get_report', '__end__']]:
        """检查是否已经有此功能的节点."""
        writer = get_stream_writer()
        writer({'info': 'Searching if an available node already exists locally...'})

        with Session(engine) as session:
            tools = tool_crud.get_all_by_image(session, str(state['image_uid']))
            if not tools:
                return Command(goto='get_report')

        # pyrefly: ignore [no-matching-overload]
        pre_search_result = await search_exist_tool_agent.ainvoke(
            {
                'messages': state['messages'],
                'image_uid': state['image_uid'],
            },
            context=UserContext(user_id=runtime.context.user_id),
        )
        if pre_search_result.get('tool_uid'):
            with Session(engine) as session:
                tool_data = tool_crud.get(session, pre_search_result['tool_uid'])
                if tool_data is None:
                    err_msg = f'Tool with uid {pre_search_result["tool_uid"]} not found'
                    raise LookupError(err_msg)

                mapper = BidirectionalMapper(datas=[TempIDClass(uid=tool_data.uid)])
                tool_node = ToolNodePublic.model_validate(tool_data.model_dump())
                tool_info = (
                    f'<tool>\n'
                    f'  <name>{tool_node.name}</name>\n'
                    f'  <description>{tool_node.description}</description>\n'
                    f'  <input_handles>\n'
                    f'{handle_formatter(tool_node.input_handles, 4)}\n'
                    f'  </input_handles>\n'
                    f'  <output_handles>\n'
                    f'{handle_formatter(tool_node.output_handles, 4)}\n'
                    f'  </output_handles>\n'
                    f'  <data>\n'
                    f'    <tool_id>{mapper.get_index_by_uid(str(tool_data.uid))}</tool_id>\n'
                    f'    <args>{tool_node.modifiable_static_params}<args>\n'
                    f'  </data>\n'
                    f'  <command>{tool_node.complete_command}</command>\n'
                    f'</tool>'
                )
            return Command(
                goto=END,
                update={
                    'id_mapper': mapper,
                    'tool_uid': str(tool_data.uid),
                    'messages': [
                        (
                            'human',
                            f'Found existing tools: \n\n {tool_info}',
                        )
                    ],
                },
            )

        return Command(goto='get_report')

    async def get_report_node(self, state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """获取工具的输出结果."""
        writer = get_stream_writer()
        writer({'info': 'Getting more information through test run...'})

        testrun_input = {
            'messages': state['messages']
        }
        # pyrefly: ignore [no-matching-overload]
        response = await direct_test_run_agent.ainvoke(
            testrun_input,
            context=TestRunContext(
                user_id=runtime.context.user_id,
                # pyrefly: ignore [bad-argument-type]
                folder_name=runtime.context.test_run_id,
                save_message=runtime.context.save_message,
                test_run_id=runtime.context.test_run_id,
            ),
        )
        run_report = response['tool_report']

        return {'messages': [HumanMessage(content=run_report)]}

    async def choose_group_node(self, state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """最后一步, 为工具选择分组."""
        writer = get_stream_writer()
        writer({'info': 'Assigning groups to tools...'})

        # pyrefly: ignore [no-matching-overload]
        response = await self.select_group_agent.ainvoke(
            {'messages': state['messages']},
            context=UserContext(user_id=runtime.context.user_id),
        )

        # 不更新消息
        return {'group_id': response['group_id']}

    async def generate_tool_node(self, state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """生成工具配置."""
        writer = get_stream_writer()
        writer({'info': 'Generating tool config...'})

        # pyrefly: ignore [no-matching-overload]
        result = await config_generate_agent.ainvoke(
            {'messages': state['messages']},
            context=UserContext(user_id=runtime.context.user_id),
        )

        with Session(engine) as session:
            group = group_crud.get(session, state['group_id'])
            if group is None:
                err_msg = 'Empty Group ID'
                raise ValueError(err_msg)

            tag = tag_crud.get_by_name(session, 'AI Unchecked')

            tool_data = ToolDefine(
                tool_type=ToolType.DOCKER,
                image_uid=state['image_uid'],
                help_doc_uid=state['doc_uid'],
                group_id=group.id,
                tags=[group.tag, tag],
                **result,
            )
            if not runtime.context.auto_save:
                return {'json_response': tool_data.model_dump()}

            tool_data = tool_crud.insert(session, tool_data)

            # 添加描述的向量索引
            embedding_model = qwen3_embedding()
            vectorstore = Milvus(
                collection_name='ToolCollection',
                connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
                embedding_function=embedding_model,
                builtin_function=BM25BuiltInFunction(),
                vector_field=['dense', 'sparse'],
                index_params=[{'index_type': 'HNSW', 'metric_type': 'L2'}, {}],
                consistency_level='Strong',
                drop_old=False,
                auto_id=True,
            )

            docs = [
                Document(
                    page_content=tool_data.description,
                    metadata={
                        'image_uid': str(tool_data.image_uid),
                        'tool_uid': str(tool_data.uid),
                        'name': tool_data.name,
                    },
                ),
            ]
            await vectorstore.aadd_documents(docs)

            mapper = BidirectionalMapper(datas=[TempIDClass(uid=tool_data.uid)])

            tool_node = ToolNodePublic.model_validate(tool_data.model_dump())
            tool_info = search_nodes_formatter([tool_node], mapper)

        return {
            'id_mapper': mapper,
            'tool_uid': str(tool_data.uid),
            'messages': [
                (
                    'human',
                    f'New tool node has been saved to the database for addition to the workflow: \n\n {tool_info}',
                )
            ],
        }

    @staticmethod
    def whether_local_search(
        state: CreateToolState,
        runtime: Runtime[CreateToolContext],
    ) -> Literal['search_exist_tool', 'get_report']:
        """如果是以智能体模式运行, 需要检查冗余."""
        if runtime.context.auto_save:
            return 'search_exist_tool'

        return 'get_report'

    def build(self, memory: Optional[BaseCheckpointSaver] = None, *, visualize: bool = False) -> CompiledStateGraph:
        """Create Tool Agent."""
        # pyrefly: ignore [bad-specialization]
        graph = StateGraph(
            state_schema=CreateToolState,
            context_schema=CreateToolContext,
            output_schema=CreateToolOutput,
        )

        graph.add_node('select_image', self.select_image_node)
        graph.add_node('get_doc', self.get_doc_node)
        graph.add_node('search_exist_tool', self.search_exist_tool_node)
        graph.add_node('get_report', self.get_report_node)
        graph.add_node('choose_group', self.choose_group_node)
        graph.add_node('generate_tool_config', self.generate_tool_node)

        graph.set_entry_point('select_image')
        graph.add_edge('select_image', 'get_doc')
        graph.add_conditional_edges('get_doc', self.whether_local_search)
        graph.add_edge('get_report', 'choose_group')
        graph.add_edge('choose_group', 'generate_tool_config')
        graph.add_edge('generate_tool_config', END)

        compiled_graph = graph.compile(checkpointer=memory) if memory is not None else graph.compile()

        if visualize:
            mermaid_code = compiled_graph.get_graph().draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with Path('images/create_tool_agent.md').open('w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        # pyrefly: ignore [bad-return]
        return compiled_graph
