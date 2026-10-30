import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, cast

from langchain.agents import AgentState, create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import CurveStyle
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import bioflow_ai.crud.tools.image as image_crud
import bioflow_ai.crud.tools.tool as tool_crud
import bioflow_ai.crud.tools.tool_doc as doc_crud
import bioflow_ai.crud.tools.tool_group as group_crud
import bioflow_ai.crud.tools.tool_tag as tag_crud
from bioflow_ai.infrastructure.containers import run_command_async
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.llm.agents.tool_agent.config_generator import config_generate_agent
from bioflow_ai.llm.agents.tool_agent.report_generator_v2 import test_run_agent
from bioflow_ai.llm.agents.tool_agent.searcher import search_image_agent
from bioflow_ai.llm.agents.tool_agent.searcher.tool_node import search_exist_tool_agent
from bioflow_ai.llm.context import BaseContext, CreateToolContext, TestRunContext, ToolGeneratorContext
from bioflow_ai.llm.middleware import (
    ChatAttachmentMiddleware,
    CheckStateExistMiddleware,
    EndLoopMiddleware,
    TokenStatisticMiddleware,
)
from bioflow_ai.llm.middleware.frontend_progress import tool_generator_step
from bioflow_ai.llm.models import get_agent_model
from bioflow_ai.llm.states import CreateToolOutput, CreateToolState
from bioflow_ai.llm.tools.database import select_classification
from bioflow_ai.llm.tools.test_run_report import TestRunReport
from bioflow_ai.models.tools import ToolDefine, ToolHelpDoc
from bioflow_ai.schemas.tool_config import ToolType
from bioflow_ai.schemas.tools import ToolNodePublic
from bioflow_ai.services.tools.index import index_tools
from bioflow_ai.utils.formatters import handle_formatter, search_nodes_formatter
from bioflow_ai.utils.uid_mapper import BidirectionalMapper

if TYPE_CHECKING:
    from bioflow_ai.llm.llm_client import LLMMetadata

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
Assign the node's primary Function group and Function/Domain tags based on its verified command, inputs and main outputs.

<Function>
- group_id=1, tag_id=4, Data Processing
- group_id=2, tag_id=5, Reference Setup
- group_id=3, tag_id=6, Quality Control
- group_id=4, tag_id=7, Alignment
- group_id=5, tag_id=8, Assembly
- group_id=6, tag_id=9, Variant Analysis
- group_id=7, tag_id=10, Annotation
- group_id=8, tag_id=11, Quantification
- group_id=9, tag_id=12, Feature Detection
- group_id=10, tag_id=13, Statistical Analysis
- group_id=11, tag_id=14, Clustering
- group_id=12, tag_id=15, Taxonomic Profiling
- group_id=13, tag_id=16, Phylogenetics
- group_id=14, tag_id=17, Structure Analysis
- group_id=15, tag_id=18, Visualization
- group_id=16, tag_id=19, Reporting
- group_id=17, tag_id=20, Utilities
</Function>

<Domain>
- tag_id=21, General
- tag_id=22, Genomics
- tag_id=23, Transcriptomics
- tag_id=24, Epigenomics
- tag_id=25, Genome Architecture
- tag_id=26, Metagenomics
- tag_id=27, Population Genetics
- tag_id=28, Evolution
- tag_id=29, Proteomics
- tag_id=30, Metabolomics
- tag_id=31, Structural Biology
- tag_id=32, Immunology
- tag_id=33, Single Cell
- tag_id=34, Cell Imaging
- tag_id=35, Pharmacology
</Domain>

<ThinkingProcess>
1. Select exactly one primary Function group based on the node's command, inputs and outputs.
2. Select at least one Domain tag. Multiple Domain tags and additional Function tags are allowed.
   If you select General, do not combine it with other Domain tags.
3. Call `select_classification(group_id=..., tag_ids=[...])` using the IDs above.
   The primary Function tag is added automatically. Review status is assigned separately when saving.
4. If validation reports an error, correct the selection. Finish only after select_classification succeeds.
</ThinkingProcess>
"""


class ToolInfoSchema(BaseModel):
    homepage: str = Field(description="Tool's homepage, prioritizing the GitHub source code repository")
    paper_url: str | None = Field(description="Link to the tool's publication, leave blank if not applicable")


class TempIDClass(BaseModel):
    uid: uuid.UUID


class SelectClassificationState(AgentState):
    group_id: NotRequired[int]
    tag_ids: NotRequired[list[int]]


PostReportRoute = Literal['__end__'] | list[Literal['choose_group', 'generate_tool_config', 'process_help_doc']]


class CreateToolAgent:
    def __init__(self) -> None:
        self.think_llm = get_agent_model()

        # sub agents
        metadata = cast('LLMMetadata', self.think_llm.metadata)
        self.select_group_agent = create_agent(
            model=self.think_llm,
            tools=[select_classification],
            system_prompt=SELECT_GROUP_SYSTEM,
            middleware=[
                ChatAttachmentMiddleware(),
                CheckStateExistMiddleware('tag_ids', 'select_classification'),
                EndLoopMiddleware(),
                TokenStatisticMiddleware('select_group', **metadata),
            ],
            state_schema=SelectClassificationState,
            context_schema=BaseContext,
        )

    @staticmethod
    def _user_message(language: str | None, *, zh: str, en: str) -> str:
        return zh if language == 'zh' else en

    @staticmethod
    @tool_generator_step('select_image')
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
    @tool_generator_step('search_exist_tool')
    async def search_exist_tool_node(
        state: CreateToolState, runtime: Runtime[CreateToolContext]
    ) -> Command[Literal['get_report', '__end__']]:
        """检查是否已经有此功能的节点."""
        with Session(engine) as session:
            tools = tool_crud.get_all_by_image(session, str(state['image_uid']))
            if not tools:
                return Command[Literal['get_report', '__end__']](goto='get_report')

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
            return Command[Literal['get_report', '__end__']](
                goto=cast('Literal["__end__"]', END),
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

        return Command[Literal['get_report', '__end__']](goto='get_report')

    @staticmethod
    @tool_generator_step('get_report')
    async def get_report_node(state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """获取工具的输出结果."""
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

    @tool_generator_step('choose_group')
    async def choose_group_node(self, state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """为工具选择主功能及 Function/Domain 标签."""
        response = await self.select_group_agent.ainvoke(
            {'messages': state['messages']},
            context=BaseContext(user_id=runtime.context.user_id),
        )

        # 不更新消息
        return {
            'group_id': response['group_id'],
            'tag_ids': response['tag_ids'],
        }

    @staticmethod
    @tool_generator_step('generate_tool_config')
    async def generate_tool_node(state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """生成工具配置."""
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

    @tool_generator_step('process_help_doc')
    async def process_help_doc_node(self, state: CreateToolState) -> dict[str, Any]:
        """处理帮助文档表, 并在状态中更新 doc_uid."""
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

    @tool_generator_step('save_tool')
    async def save_tool_node(self, state: CreateToolState, runtime: Runtime[CreateToolContext]) -> dict[str, Any]:
        """保存工具节点, 或在非自动保存模式下返回结构化结果."""
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

            tag_ids = [*state['tag_ids'], 3]
            if group.tag_id is not None:
                tag_ids.append(group.tag_id)

            tool_data = ToolDefine(
                tool_type=ToolType.DOCKER,
                image_uid=state['image_uid'],
                help_doc_uid=state['doc_uid'],
                group_id=group.id,
                tags=list(tag_crud.get_all_by_id(session, tag_ids)),
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
                        'tags': [{'id': tag.id, 'name': tag.name} for tag in tool_data.tags],
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
            await index_tools([tool_data])

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
    def route_after_image_selection(
        state: CreateToolState,
        runtime: Runtime[CreateToolContext],
    ) -> Literal['search_exist_tool', 'get_report', '__end__']:
        """如果是以智能体模式运行, 需要检查冗余."""
        if state.get('return_to_parent'):
            return '__end__'
        return 'get_report' if runtime.context.skip_search else 'search_exist_tool'

    @staticmethod
    def route_after_tool_report(state: CreateToolState) -> PostReportRoute:
        """End on a parent handoff or fan out the independent post-report work."""
        if state.get('return_to_parent'):
            return '__end__'
        return ['choose_group', 'generate_tool_config', 'process_help_doc']

    def build(
        self, memory: BaseCheckpointSaver | None = None, *, visualize: bool = False
    ) -> CompiledStateGraph[Any, CreateToolContext, Any, CreateToolOutput]:
        """Create Tool Agent."""
        graph = StateGraph(
            state_schema=CreateToolState,  # ty: ignore[invalid-argument-type]
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

        graph.add_edge(START, 'select_image')
        graph.add_conditional_edges('select_image', self.route_after_image_selection)
        graph.add_conditional_edges(
            'get_report',
            self.route_after_tool_report,
            ['choose_group', 'generate_tool_config', 'process_help_doc', END],
        )
        graph.add_edge(['choose_group', 'generate_tool_config', 'process_help_doc'], 'save_tool')
        graph.add_edge('save_tool', END)

        compiled_graph = graph.compile(checkpointer=memory) if memory is not None else graph.compile()

        if visualize:
            mermaid_code = compiled_graph.get_graph().draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with Path('images/create_tool_agent.md').open('w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')

        return compiled_graph
