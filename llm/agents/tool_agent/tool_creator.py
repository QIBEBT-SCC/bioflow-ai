import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Optional, cast

from langchain.agents import AgentState, create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import CurveStyle
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import app.crud.tools.image as image_crud
import app.crud.tools.tool as tool_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool_tag as tag_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.tools import ToolDefine, ToolHelpDoc
from app.schemas.tool_config import ToolType
from app.schemas.tools import ToolNodePublic
from app.utils.container import run_command_async
from llm.agents.tool_agent.config_generator import config_generate_agent
from llm.agents.tool_agent.report_generator_v2 import test_run_agent
from llm.agents.tool_agent.searcher import search_image_agent
from llm.agents.tool_agent.searcher.tool_node import search_exist_tool_agent
from llm.context import BaseContext, CreateToolContext, TestRunContext, ToolGeneratorContext
from llm.models import get_agent_model, qwen3_embedding
from llm.states import (
    CreateToolOutput,
    CreateToolState,
)
from llm.tools.database import (
    select_group,
    view_groups,
)
from llm.tools.test_run_report import TestRunReport
from llm.utils.formatters import handle_formatter, search_nodes_formatter
from llm.utils.middleware import TokenStatisticMiddleware
from llm.utils.tool_creation_routes import route_after_image_selection, route_after_tool_report
from llm.utils.uid_mapper import BidirectionalMapper
from llm.vectorstores import MilvusVectorStore

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
    - If an image for the tool exists, directly use `select_image(image_ref=...)` with the exact `<image_ref>` value to select the corresponding image.
    - Otherwise, call `search_image` to search for images on Docker Hub, and based on the results, use the `pull_and_save_image` tool to pull and save the image.

If you encounter any unresolvable issues at any point, you can call the `ask_human` tool to ask for human assistance.
</ThinkingProcess>
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


class SelectGroupState(AgentState):
    group_id: NotRequired[int]


class CreateToolAgent:
    def __init__(self) -> None:
        self.think_llm = get_agent_model()

        # sub agents
        metadata = cast('LLMMetadata', self.think_llm.metadata)
        # pyrefly: ignore [no-matching-overload]
        self.select_group_agent = create_agent(
            model=self.think_llm,
            tools=[view_groups, select_group],
            system_prompt=SELECT_GROUP_SYSTEM,
            middleware=[TokenStatisticMiddleware('select_group', **metadata)],
            state_schema=SelectGroupState,
            context_schema=BaseContext,
        )

    @staticmethod
    def _user_message(language: str | None, *, zh: str, en: str) -> str:
        return zh if language == 'zh' else en

    @staticmethod
    async def select_image_node(state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """首先搜索得到此工具的对应镜像."""
        if state.get('image_uid'):
            # 我们认为数据库中的镜像信息就是完全的
            with Session(engine) as session:
                tool_image = image_crud.get(session, str(state['image_uid']))
                if tool_image is None:
                    err_msg = f'Image with uid {state["image_uid"]} not found'
                    raise LookupError(err_msg)
                return {'image': tool_image.image}
        else:
            # TODO: 重写镜像搜索
            writer = get_stream_writer()
            writer({'info': 'Searching for docker image...'})

            search_input = {'messages': state['messages']}
            response = await search_image_agent.ainvoke(
                search_input,
                context=BaseContext(
                    user_id=runtime.context.user_id,
                    language=runtime.context.language,
                    delegate_human_requests=runtime.context.delegate_human_requests,
                ),
            )

            if response.get('return_to_parent'):
                return {
                    'return_to_parent': True,
                    'parent_request': response['parent_request'],
                }

            logger.success(f'Select image {response["image"]}')
            return {
                'image_uid': response['image_uid'],
                'image': response['image'],
            }

    @staticmethod
    async def search_exist_tool_node(
        state: CreateToolState, runtime: Runtime[CreateToolContext]
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
            context=BaseContext(user_id=runtime.context.user_id),
        )
        if pre_search_result.get('tool_uid'):
            with Session(engine) as session:
                tool_data = tool_crud.get(session, pre_search_result['tool_uid'])
                if tool_data is None:
                    err_msg = f'Tool with uid {pre_search_result["tool_uid"]} not found'
                    raise LookupError(err_msg)

                mapper = BidirectionalMapper(datas=[TempIDClass(uid=tool_data.uid)])
                tool_node = ToolNodePublic.model_validate(tool_data)
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
                    f'    <args>{tool_node.modifiable_static_params}</args>\n'
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
                            CreateToolAgent._user_message(
                                runtime.context.language,
                                zh=f'找到已有工具节点:\n\n{tool_info}',
                                en=f'Found an existing tool node:\n\n{tool_info}',
                            ),
                        )
                    ],
                },
            )

        return Command(goto='get_report')

    @staticmethod
    async def get_report_node(state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """获取工具的输出结果."""
        writer = get_stream_writer()
        writer({'info': 'Getting more information through test run...'})

        testrun_input = {'messages': state['messages']}
        response = await test_run_agent.ainvoke(
            testrun_input,
            context=TestRunContext(
                user_id=runtime.context.user_id,
                language=runtime.context.language,
                test_run_id=runtime.context.test_run_id,
                project_id=runtime.context.project_id,
                save_message=runtime.context.save_message,
                delegate_human_requests=runtime.context.delegate_human_requests,
            ),
        )
        if response.get('return_to_parent'):
            return {
                'return_to_parent': True,
                'parent_request': response['parent_request'],
            }
        structured_report = TestRunReport.model_validate(response['test_run_report'])

        return {
            'messages': [HumanMessage(content=structured_report.to_markdown())],
            'test_run_report': structured_report,
            'help_command': structured_report.help_command,
        }

    async def choose_group_node(self, state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """最后一步, 为工具选择分组."""
        writer = get_stream_writer()
        writer({'info': 'Assigning groups to tools...'})

        # pyrefly: ignore [no-matching-overload]
        response = await self.select_group_agent.ainvoke(
            {'messages': state['messages']},
            context=BaseContext(user_id=runtime.context.user_id),
        )

        # 不更新消息
        return {'group_id': response['group_id']}

    @staticmethod
    async def generate_tool_node(state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """生成工具配置."""
        writer = get_stream_writer()
        writer({'info': 'Generating tool config...'})

        # pyrefly: ignore [no-matching-overload]
        result = await config_generate_agent.ainvoke(
            {
                'messages': state['messages'],
                'image': state['image'],
                'test_run_report': state['test_run_report'],
            },
            context=ToolGeneratorContext(
                user_id=runtime.context.user_id,
                language=runtime.context.language,
                test_run_id=runtime.context.test_run_id,
                project_id=runtime.context.project_id,
                save_message=runtime.context.save_message,
            ),
        )
        return {
            key: result[key]
            for key in (
                'name',
                'description',
                'command_template',
                'dynamic_params',
                'immutable_static_params',
                'modifiable_static_params',
                'file_mounts',
            )
        }

    async def process_help_doc_node(self, state: CreateToolState) -> dict[str, Any]:
        """处理帮助文档表, 并在状态中更新 doc_uid."""
        writer = get_stream_writer()
        writer({'info': 'Preparing tool help document...'})

        if state.get('doc_uid'):
            return {'doc_uid': state['doc_uid']}

        help_command = state['help_command'].strip()
        if not help_command:
            err_msg = 'Empty help command'
            raise ValueError(err_msg)

        with Session(engine) as session:
            tool_doc = doc_crud.get_by_command(session, help_command, f'{state["image_uid"]}')
            if tool_doc is not None:
                return {'doc_uid': tool_doc.uid}

        _code, content = await run_command_async(state['image'], [], help_command)

        with Session(engine) as session:
            tool_doc = ToolHelpDoc(
                help_command=help_command,
                description='',
                content=content,
                image_uid=state.get('image_uid'),
            )
            tool_doc = doc_crud.insert(session, tool_doc)
            session.commit()
            return {'doc_uid': tool_doc.uid}

    async def save_tool_node(self, state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """保存工具节点, 或在非自动保存模式下返回结构化结果."""
        writer = get_stream_writer()
        writer({'info': 'Saving tool config...'})

        tool_config = {
            key: state[key]
            for key in (
                'name',
                'description',
                'command_template',
                'dynamic_params',
                'immutable_static_params',
                'modifiable_static_params',
                'file_mounts',
            )
        }

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
                **tool_config,
            )
            if not runtime.context.auto_save:
                return {
                    'json_response': {
                        'name': tool_data.name,
                        'image_uid': tool_data.image_uid,
                        'description': tool_data.description,
                        'tool_type': tool_data.tool_type,
                        'group_id': tool_data.group_id,
                        'help_doc_uid': tool_data.help_doc_uid,
                        'command_template': tool_data.command_template,
                        'dynamic_params': tool_data.dynamic_params,
                        'immutable_static_params': tool_data.immutable_static_params,
                        'modifiable_static_params': tool_data.modifiable_static_params,
                        'file_mounts': tool_data.file_mounts,
                    }
                }

            tool_data = tool_crud.insert(session, tool_data)
            session.commit()

            # 添加描述的向量索引
            embedding_model = qwen3_embedding()
            vectorstore = MilvusVectorStore(
                collection_name='ToolCollection',
                embedding_function=embedding_model,
                uri=get_settings().knowledge_base.MILVUS_URL,
                hybrid=True,
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
            async with vectorstore:
                await vectorstore.aadd_documents(docs)

            mapper = BidirectionalMapper(datas=[TempIDClass(uid=tool_data.uid)])

            tool_node = ToolNodePublic.model_validate(tool_data)
            tool_info = search_nodes_formatter([tool_node], mapper)

        return {
            'id_mapper': mapper,
            'tool_uid': str(tool_data.uid),
            'messages': [
                (
                    'human',
                    self._user_message(
                        runtime.context.language,
                        zh=f'新的工具节点已保存到数据库, 可添加到工作流中:\n\n{tool_info}',
                        en=f'The new tool node has been saved and is ready to add to a workflow:\n\n{tool_info}',
                    ),
                )
            ],
        }

    @staticmethod
    def whether_local_search(
        state: CreateToolState,
        runtime: Runtime[CreateToolContext],
    ) -> Literal['search_exist_tool', 'get_report', '__end__']:
        """如果是以智能体模式运行, 需要检查冗余."""
        return route_after_image_selection(state, skip_search=runtime.context.skip_search)

    def build(self, memory: Optional[BaseCheckpointSaver] = None, *, visualize: bool = False) -> CompiledStateGraph:
        """Create Tool Agent."""
        # pyrefly: ignore [bad-specialization]
        graph = StateGraph(
            state_schema=CreateToolState,
            context_schema=CreateToolContext,
            output_schema=CreateToolOutput,
        )

        graph.add_node('select_image', self.select_image_node)
        graph.add_node('search_exist_tool', self.search_exist_tool_node)
        graph.add_node('get_report', self.get_report_node)
        graph.add_node('choose_group', self.choose_group_node)
        graph.add_node('generate_tool_config', self.generate_tool_node)
        graph.add_node('process_help_doc', self.process_help_doc_node)
        graph.add_node('save_tool', self.save_tool_node)

        graph.set_entry_point('select_image')
        graph.add_conditional_edges('select_image', self.whether_local_search)
        graph.add_conditional_edges('get_report', route_after_tool_report)
        graph.add_edge(['choose_group', 'generate_tool_config', 'process_help_doc'], 'save_tool')
        graph.add_edge('save_tool', END)

        compiled_graph = graph.compile(checkpointer=memory) if memory is not None else graph.compile()

        if visualize:
            mermaid_code = compiled_graph.get_graph().draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with Path('images/create_tool_agent.md').open('w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        # pyrefly: ignore [bad-return]
        return compiled_graph
