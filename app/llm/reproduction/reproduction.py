import json
from typing import Annotated, Literal, TypedDict, Optional

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langchain_core.tools import tool, InjectedToolCallId, ToolException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.types import Command, interrupt
from sqlmodel import Session
from uuid_extensions import uuid7

import app.crud.tools.tool as tool_crud
import app.crud.resource.bio_db as db_crud
import app.crud.workflow as wf_crud
from app.core.llm_chat import gemini_25_flash, gpt_o4_mini
from app.core.session import engine
from app.llm.formatters import node_formatter
from app.llm.reproduction.create_tool_agent import CreateToolAgent
from app.llm.states import CreateToolState, BuildNode, BuildEdge, ReproductionState, ToolsOutput
from app.llm.tools.database import search_tool, search_biodb
from app.llm.tools.human import ask_human
from app.models.resource import BioDb
from app.models.workflow import WorkflowType, Workflow
from app.schemas.react_flow import WorkflowDefinition, Node, Position, NodeData, Edge
from app.schemas.tools import ToolNodePublic, Handle

GET_TOOL_SYSTEM = """You are a bioinformatics researcher helping set up software tools for a research group's server.
Your task is to analyze academic text and extract information about bioinformatics tools mentioned in the literature.

## Important Instructions:
- **Return only actual bioinformatics tools**
    – These include tools used for quality control, sequence alignment, genome assembly, annotation, variant calling, transcript quantification, metagenomic profiling, etc.
- **Do NOT include:**
    - General-purpose programming languages (e.g., R, Python, Perl)
    - Statistical functions or downstream analysis methods (e.g., PCA, t-SNE, DESeq2)

## Output Format
Your output needs to include the following two pieces of information:
- `tools`: a list containing all the tool names you have extracted
- `workflow_structure`: a **mermaid code** for the workflow diagram derived from the original paper
    - Starts with "```mermaid" and ends with "```" (that is, the standard markdown code block format), does not contain any other text in the task.
    - Use regular nodes to represent tools, and hexagon nodes to represent files
"""

GET_TOOL_USER = """Here are the relevant paragraphs of the paper:
<paper_content>
{paper_content}
</paper_content>
"""

BUILD_FLOW_SYSTEM = """你是一名生物信息分析工程师，正在使用一个节点式的生物信息分析平台。
这个平台将分析工具、数据库、分析数据等都封装成了一个个的节点，你的任务是使用它们复现一篇论文中的原始工作流。

<NodeDefine>
本系统将所有的工具、资源、操作等都封装成了节点。每个节点都包含以下五部分：
- **id**: 每添加一个新的节点，会自动为其分配一个id
- **type**: 节点的类型，包括tool、resource、data_process和note四类
    - tool: 生物信息工具节点
    - resource: 资源节点，一般代表静态的文件资源，包括原始测序数据、数据库、参考基因组等
    - data_process: 对一些数据操作的封装，如将文件复制、截取、更进一步的数据过滤等等非生信工具本身能够实现的操作
    - note: 笔记，可以对某些节点加以说明
- **input_handles**: 节点接受的输入
- **output_handles**: 节点产生的输出
- **data**: 节点自身的其他数据，不同的节点类型对应不同的具体格式

此外，为了便于理解，tool类型的节点还存在一个command属性，展示了这个节点的完整运行命令。
</NodeDefine>

<ThinkingProcess>
你的任务就是根据原始文章的内容，调用相应的工具，逐步向工作流中添加节点。
你可能已经注意到了节点的data字段并可能为此困扰，实际上我们对此封装了完善的工具，你只需要将简单的参数传递给工具，系统能够自动根据现有数据完成data的内容。你只需要按照以下思路来进行即可：
1. 根据历史进度，决定当前要添加什么节点。按照工具调用顺序，先添加前置工具再添加后面的；先添数据库再添加工具本体。总之遵循的原则就是：当前添加节点的输入来源一定要全部存在
2. 接下来，调用合适的搜索工具(search_开头)，并从返回的结果中选择合适的“节点模板”
    - 如果这一步有可用的工具节点，直接进入下一步
    - 当没有找到可用的工具的节点时，调用`create_tool_node`工具新建节点
3. 调用添加节点工具，将你选择的模板实际添加到工作流中，这个工具会返回实例化后的节点信息，包括data的内容以及分配给这个节点的id。后者有助于你在后续添加节点时定义边连接使用
    - 这一步你主要需要决定两个内容，一个是节点的坐标，一个是节点的输入边连接
    - 根据现有工具已经占用的坐标位置和整个工作流的结构，为新节点分配合适的坐标
    - 利用现有工具的输出，定义通向此节点的输入边连接
    - 具体的描述可见于<ToolCalling>部分
4. 在添加节点时，你可能会发现由于最开始考虑不周，某些节点缺失了必要的前置分析节点。此时你可以先暂缓添加该节点，先去添加所需的前置节点
5. 如果在任意一处遇到设想之外的情况（如数据库找不到、节点没有合适的边连接对象等），都可以调用`ask_human`工具，询问人类意见

这样，当你一个节点一个节点的添加完毕的时候，实际上也就得到了整个完整的工作流，你的任务就完成了。
整个工作流中初始已经添加了一个reads资源节点，输出r1和r2的双端reads数据，一般都可以作为常规生物信息工作流的起点。如果这个输入不合适，建议先询问人类意见。
</ThinkingProcess>

<ToolCalling>
为了帮助你更好的搜索节点和构筑工作流，我们定义了以下工具：
- `search_tool`: 用于搜索工具节点，使用工具名称进行搜索
- `add_tool_node`: 用于向工具流中添加工具节点，你需要从搜索结果中选择一个合适的对象，将他的id传给此工具。除此之外，你还主要需要思考以下两点
    - 节点的位置：以初始的read节点开始，剩下的节点向x轴正方向铺开，同一个层次的工具可以放在同一个x坐标的不同y坐标上。总之，遵顼以下两点：
        - 一个坐标只能放一个节点
        - 每次改动以1为单位，如（上一个工具B在[1, 0]，下一个工具C基于B的输出，放在[2, 0]; 工具D也基于B的输出，但是[2, 0]已经有节点了，则放在[2, 1]）
    - 节点的输入边：此节点的所有必须连接的input_handle对应的边列表。每一条边主要就是看来自哪个节点的哪个output_handle，输入到这个节点的哪个input_handle
- `create_tool_node`: 当使用`search_tool`没有找到可用的工具节点时，调用此工具新建工具节点
- `search_biodb`: 用于搜索数据库节点，一般使用调用此数据库的工具名称搜索，如果找不到，可以再试试直接使用工具库本身的名称搜索
- `add_resource_node`: 用于向工具流中添加资源节点（有的资源节点也需要id，有的则不需要）。添加的思考过程与工具节点类似
- `ask_human`: 当遇到所有无法解决的问题（一般是搜不到资源）时，调用这个工具，它会向人类请示并返回人类的反馈信息
- `save_workflow`: Once the workflow is built, call this tool to save the workflow.
</ToolCalling>
"""


class ReproductionAgent:
    def __init__(self):
        self.struct_llm: ChatOpenAI = gpt_o4_mini()
        self.tool_llm: ChatGoogleGenerativeAI = gemini_25_flash()

        # agents
        self.create_tool_agent = CreateToolAgent().build()

        # tools
        self.call_tool_node = ToolNode(
            [search_tool, search_biodb, add_tool_node, add_resource_node, create_tool, ask_human, save_workflow]
        )

    def extract_tools_node(self, state: ReproductionState):
        """从原始文章中提取出所使用的工具"""

        class StructureOutput(TypedDict):
            raw: AIMessage
            parsed: Optional[ToolsOutput]
            parsing_error: Optional[BaseException]

        if len(state.messages) == 0:
            human_question = (
                ChatPromptTemplate(messages=[('human', GET_TOOL_USER)]).invoke(state.paper_content).to_messages()
            )
            state.messages = human_question

        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=GET_TOOL_SYSTEM), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.struct_llm.with_structured_output(ToolsOutput, include_raw=True)
        output: StructureOutput = chain.invoke({'history': state.messages})

        return {'paper_tools': output['parsed'], 'messages': [output['raw']]}

    @staticmethod
    def analyse_extract_result_node(
        state: ReproductionState,
    ) -> Command[Literal['extract_tools_node', 'build_workflow_node']]:
        """工具提取的结果处理

        如果没有找到工具，触发中断询问人类，并将结果返回到提取节点
        否则，将提取的工具和流程图返回给人类，询问意见。如果同意则进入下一步，否则将反馈信息返回并重新提取。
        """

        if len(state.paper_tools.tools) == 0:
            question = 'I found 0 tools in the paper, please checkout the content you provide.'
            answer = interrupt(question)
            return Command(
                goto='extract_tools_node',
                update={'messages': [AIMessage(content=question), HumanMessage(content=answer)]},
            )

        question = (
            f'I found {len(state.paper_tools.tools)} tools in the paper, '
            f'Their workflow is shown in the figure below. Do you think there is any problem?\n'
            f'{state.paper_tools.workflow_structure}'
        )
        answer = interrupt(question)
        if isinstance(answer, bool) and answer:
            init_node = BuildNode(
                id='A',
                type='reads',
                input_handles=[],
                output_handles=[
                    Handle(name='r1', description='the first read of a paired-end sequence.'),
                    Handle(name='r2', description='the second read of a paired-end sequence.'),
                ],
                data=NodeData(args='{"r1":"","r2":""}'),
                x=0,
                y=0,
            )
            return Command(
                goto='build_workflow_node',
                update={
                    'exist_nodes': [init_node],
                    'messages': [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES),
                        ('human', GET_TOOL_USER.format(paper_content=state.paper_content)),
                        (
                            'ai',
                            f'I found {len(state.paper_tools.tools)} tools in the paper: [{",".join(state.paper_tools.tools)}]. '
                            f'Their general workflow is shown in the figure below:\n{state.paper_tools.workflow_structure}',
                        ),
                        (
                            'human',
                            f'Prefect, now start calling tools to perform specific workflow construction tasks. The following nodes have been initially added:\n{node_formatter(init_node)}',
                        ),
                    ],
                },
            )
        return Command(
            goto='extract_tools_node', update={'messages': [AIMessage(content=question), HumanMessage(content=answer)]}
        )

    def build_workflow_node(self, state: ReproductionState):
        """根据返回的结构构建工作流"""
        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=BUILD_FLOW_SYSTEM), MessagesPlaceholder(variable_name='history')]
        )
        llm = self.tool_llm.bind_tools(
            [search_tool, search_biodb, create_tool, add_tool_node, add_resource_node, ask_human, save_workflow],
            tool_choice='any',
        )
        chain = prompt | llm

        response = chain.invoke({'history': state.messages})
        return {'messages': [response]}

    @staticmethod
    def should_continue_build(
        state: ReproductionState,
    ) -> Literal['call_tool_node', 'create_tool_node', 'save_workflow_node']:
        last_message = state.messages[-1]
        assert last_message.tool_calls
        tool_call = last_message.tool_calls[0]

        if tool_call['name'] == 'save_workflow':
            return 'save_workflow_node'
        elif tool_call['name'] == 'create_tool_node':
            return 'create_tool_node'
        return 'call_tool_node'

    @staticmethod
    def should_compress(state: ReproductionState) -> Literal['compress_message_node', 'build_workflow_node']:
        ai_message = state.messages[-2]
        assert ai_message.tool_calls
        tool_call = ai_message.tool_calls[0]

        if tool_call['name'] == 'add_node_by_id':
            return 'compress_message_node'
        return 'build_workflow_node'

    @staticmethod
    def compress_message_node(state: ReproductionState):
        """压缩对话上下文"""
        wf_data = {
            'nodes': [n.model_dump() for n in state.exist_nodes],
            'edges': [e.model_dump() for e in state.exist_edges],
        }
        wf_str = json.dumps(wf_data, indent=2, sort_keys=False)
        return {
            'messages': [
                *[RemoveMessage(id=m.id) for m in state.messages[2:-2]],
                (
                    'human',
                    f'Prefect, now the following are some of the workflows that have been completed:\n<workflow>\n{wf_str}\n</workflow>\n\nPlease continue to call the tools to work.',
                ),
            ]
        }

    def create_tool_node(self, state: ReproductionState):
        last_message: AIMessage = state.messages[-1]

        writer = get_stream_writer()
        writer({'custom_msg': f'search query: {last_message.tool_calls[0]["args"]["request"]}'})

        sub_agent_input = CreateToolState(
            messages=[HumanMessage(content=last_message.tool_calls[0]['args']['request'])]
        )
        response = self.create_tool_agent.invoke(sub_agent_input)
        agent_msg = response['messages'][-1].content

        return {'messages': [ToolMessage(content=agent_msg, tool_call_id=last_message.tool_calls[0]['id'])]}

    @staticmethod
    def save_workflow_node(state: ReproductionState):
        nodes = []
        for n in state.exist_nodes:
            if n.type == 'tool':
                node_type = 'tool'
            elif n.type == 'db':
                node_type = 'dbInputNode'
            elif n.type == 'genome':
                node_type = 'refInputNode'
            elif n.type == 'reads':
                node_type = 'sequenceInputNode'
            else:
                node_type = ''

            node = Node(
                id=n.id,
                type=node_type,
                data=n.data,
                position=Position(x=n.x * 500, y=n.y * 400),
                dragHandle='.nodeDragable',
                zIndex=20,
            )
            nodes.append(node)

        edges = []
        for e in state.exist_edges:
            source_handle = f'{e.source_node}-out-{e.source_handle}'
            target_handle = f'{e.target_node}-in-{e.target_handle}'
            edge = Edge(
                id=f'xy-edge__{source_handle}-{target_handle}',
                animated=True,
                source=e.source_node,
                target=e.target_node,
                sourceHandle=source_handle,
                targetHandle=target_handle,
            )
            edges.append(edge)

        workflow_define = WorkflowDefinition(nodes=nodes, edges=edges)

        wf_name = state.messages[-1].tool_calls[0]['args']['name']
        with Session(engine) as session:
            uid = uuid7(as_type='str')
            workflow = Workflow(
                uid=uid,
                name=f'{wf_name}(AI)',
                owner_id=1,
                wf_type=WorkflowType.TEMPLATE,
                workflow=workflow_define,
                public=True,
            )

            wf_crud.insert(session, workflow)

        return

    def build(self, memory: BaseCheckpointSaver) -> CompiledStateGraph:
        graph = StateGraph(ReproductionState)
        graph.add_node('extract_tools_node', self.extract_tools_node)
        graph.add_node('analyse_extract_result_node', self.analyse_extract_result_node)
        graph.add_node('build_workflow_node', self.build_workflow_node)
        graph.add_node('call_tool_node', self.call_tool_node)
        graph.add_node('compress_message_node', self.compress_message_node)
        graph.add_node('create_tool_node', self.create_tool_node)
        graph.add_node('save_workflow_node', self.save_workflow_node)

        graph.add_edge(START, 'extract_tools_node')
        graph.add_edge('extract_tools_node', 'analyse_extract_result_node')
        graph.add_conditional_edges('build_workflow_node', self.should_continue_build)
        graph.add_conditional_edges('call_tool_node', self.should_compress)
        graph.add_edge('compress_message_node', 'build_workflow_node')
        graph.add_edge('create_tool_node', 'build_workflow_node')
        graph.add_edge('save_workflow_node', END)

        compiled_graph = graph.compile(checkpointer=memory)
        return compiled_graph

    def visualize(self):
        graph = StateGraph(ReproductionState)
        graph.add_node('extract_tools_node', self.extract_tools_node)
        graph.add_node('analyse_extract_result_node', self.analyse_extract_result_node)
        graph.add_node('build_workflow_node', self.build_workflow_node)
        graph.add_node('call_tool_node', self.call_tool_node)
        graph.add_node('compress_message_node', self.compress_message_node)
        graph.add_node('create_tool_node', self.create_tool_agent)
        graph.add_node('save_workflow_node', self.save_workflow_node)

        graph.add_edge(START, 'extract_tools_node')
        graph.add_edge('extract_tools_node', 'analyse_extract_result_node')
        graph.add_conditional_edges('build_workflow_node', self.should_continue_build)
        graph.add_conditional_edges('call_tool_node', self.should_compress)
        graph.add_edge('compress_message_node', 'build_workflow_node')
        graph.add_edge('create_tool_node', 'build_workflow_node')
        graph.add_edge('save_workflow_node', END)

        compiled_graph = graph.compile()
        mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
            curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
        )
        with open('images/reproduction_agent.md', 'w') as f:
            f.write(f'```mermaid\n{mermaid_code}\n```')


@tool('create_tool_node')
def create_tool(request: str):
    """Create new tool node

    Args:
        request: Node creation requirements. Format: "Using XXX (tool name) to XXX (specific task description, including what you want to do with this tool, what the tool's input is, and what the expected output is)"
    """
    return


@tool
def add_tool_node(
    pos_x: int,
    pos_y: int,
    selected_id: int,
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add tool nodes to the workflow by tool ID

    Args:
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        selected_id: The tool ID of the corresponding node source
        input_edges: Input edge connection list for this node
    """
    node_count = len(state.exist_nodes)
    node_id = generate_char_index(node_count + 1)

    with Session(engine) as session:
        db_tool = tool_crud.get(session, selected_id)
        db_node = ToolNodePublic.model_validate(db_tool.model_dump())
        new_node = BuildNode(
            id=node_id,
            type='tool',
            input_handles=db_node.input_handles,
            output_handles=db_node.output_handles,
            data=NodeData(tool_id=selected_id, args=db_node.static_params),
            command=db_node.complete_command,
            x=pos_x,
            y=pos_y,
        )

    for edge in input_edges:
        edge.target_node = node_id

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'messages': [
                ToolMessage(content=f'add new tool node:\n{node_formatter(new_node)}', tool_call_id=tool_call_id)
            ],
        }
    )


@tool
def add_resource_node(
    resource_type: Literal['db', 'ref_genome'],
    pos_x: int,
    pos_y: int,
    selected_id: Optional[int],
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add a resource node to the workflow

    Args:
        resource_type: Resource node type (db: biological database; ref_genome: reference genome)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        selected_id: ID of the corresponding resource (required for DB type)
        input_edges: Input edge connection list for this node
    """
    node_count = len(state.exist_nodes)
    node_id = generate_char_index(node_count + 1)

    match resource_type:
        case 'db':
            with Session(engine) as session:
                db_db: BioDb = db_crud.get(session, selected_id)
                new_node = BuildNode(
                    id=node_id,
                    type='db',
                    input_handles=[],
                    output_handles=[Handle(name='file_path', description='db file path')],
                    data=NodeData(args=f'db:{db_db.name}'),
                    x=pos_x,
                    y=pos_y,
                )
        case 'ref_genome':
            new_node = BuildNode(
                id=node_id,
                type='genome',
                input_handles=[
                    Handle(
                        name='species_name',
                        description='(Optional) String variable, species name of the reference gene to be used',
                    ),
                    Handle(
                        name='ncbi_tax_id',
                        description='(Optional) String variable, String variable, NCBI taxonomy ID of the species of the reference gene to be used',
                    ),
                ],
                output_handles=[
                    Handle(name='genome_fasta', description='Gene fasta file of the reference genome'),
                    Handle(name='annotation_gff', description='GFF annotation file of the reference genome'),
                    Handle(name='bowtie2_index', description='Bowtie2 index of the reference genome'),
                    Handle(name='bwa_index', description='BWA index of the reference genome'),
                    Handle(name='hisat2_index', description='HISAT2 index of the reference genome'),
                    Handle(name='minimap2_index', description='minimap2 index of the reference genome'),
                    Handle(name='star_index', description='STAR index of the reference genome'),
                ],
                data=NodeData(args=''),
                x=pos_x,
                y=pos_y,
            )
        case _:
            raise ToolException('Illegal resource type')

    for edge in input_edges:
        edge.target_node = node_id

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'messages': [
                ToolMessage(content=f'add new tool node:\n{node_formatter(new_node)}', tool_call_id=tool_call_id)
            ],
        }
    )


@tool
def save_workflow(name: str):
    """Save the completed workflow into the database

    Args:
        name: Workflow name, brief description of this workflow's tasks
    """
    return


def generate_char_index(n: int) -> str:
    """将一个正整数转换为（A, B, ..., Z, AA, AB, ...）形式的索引"""
    assert n >= 1, '输入必须是一个大于等于1的正整数'

    result = []
    while n > 0:
        remainder = (n - 1) % 26
        result.append(chr(65 + remainder))
        n = (n - 1) // 26
    return ''.join(reversed(result))
