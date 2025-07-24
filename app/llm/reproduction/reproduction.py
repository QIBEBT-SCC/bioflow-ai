import operator
from typing import Annotated, Literal, TypedDict, Optional

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from langchain_core.tools import tool, InjectedToolCallId
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import START, END
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from sqlmodel import Session
from uuid_extensions import uuid7

import app.crud.tools.tool as tool_crud
import app.crud.resource.bio_db as db_crud
import app.crud.workflow as wf_crud
from app.core.llm_chat import gpt_41_mini, gemini_25_flash
from app.core.session import engine
from app.llm.reproduction.check_tool_agent import CheckToolAgent
from app.models.resource import BioDb
from app.models.tools import ToolData
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
    - Statistical functions or downstream analysis methods (e.g., PCA, t-SNE, DESeq2 if only used for plotting or statistical testing)

## Output Format
Your output needs to include the following two pieces of information:
- `tools`: a list containing all the tool names you have extracted
- `workflow_structure`: a **mermaid code** for the workflow diagram derived from the original paper, using the extracted tool names as node names. Starts with "```mermaid" and ends with "```" (that is, the standard markdown code block format), does not contain any other text in the task.
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
- **type**: 节点的类型，包括tool、reads、db和note五类
    - tool: 生物信息工具节点
    - reads: 测序数据输入节点（只有output_handles）
    - db: 生物信息数据库输入节点（只有output_handles）
    - data_process: 对一些数据操作的封装，如将文件复制、截取、更进一步的数据过滤等等非生信工具本身能够实现的操作
    - note: 
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
3. 将调用添加节点工具，将你选择的模板实际添加到工作流中，这个工具会返回实例化后的节点信息，包括data的内容以及分配给这个节点的id。后者有助于你在后续添加节点时定义边连接使用
    - 这一步你主要需要决定两个内容，一个是节点的坐标，一个是节点的输入边连接
    - 根据现有工具已经占用的坐标位置和整个工作流的结构，为新节点分配合适的坐标
    - 利用现有工具的输出，定义通向此节点的输入边连接
    - 具体的描述可见于<ToolCalling>部分
4. 如果在任意一处遇到设想之外的情况（如工具找不到、节点没有合适的边连接对象等），都可以调用`ask_human`工具，询问人类意见

这样，当你一个节点一个节点的添加完毕的时候，实际上也就得到了整个完整的工作流，你的任务就完成了。
整个工作流中初始已经添加了一个reads资源节点，输出r1和r2的双端reads数据，一般都可以作为常规生物信息工作流的起点。如果这个输入不合适，建议先询问人类意见。
</ThinkingProcess>

<ToolCalling>
为了帮助你更好的搜索节点和构筑工作流，我们定义了以下工具：
- `search_tool`: 用于搜索工具节点，使用工具名称进行搜索
- `search_db`: 用于搜索数据库节点，一般使用调用此数据库的工具名称搜索，如果找不到，可以再试试直接使用工具库本身的名称搜索
- `add_node_by_id`: 用于向工具流中添加工具和数据库节点，你需要从搜索结果中选择一个合适的对象，将他的id传给此工具。除此之外，你还主要需要思考以下两点
    - 节点的位置：以初始的read节点开始，剩下的节点向x轴正方向铺开，同一个层次的工具可以放在同一个x坐标的不同y坐标上。总之，遵顼以下两点：
        - 一个坐标只能放一个节点
        - 每次改动以1为单位，如（上一个工具B在[1, 0]，下一个工具C基于B的输出，放在[2, 0]; 工具D也基于B的输出，但是[2, 0]已经有节点了，则放在[2, 1]）
    - 节点的输入边：此节点的所有必须连接的input_handle对应的边列表。每一条边主要就是看来自哪个节点的哪个output_handle，输入到这个节点的哪个input_handle
- `ask_human`: 当遇到所有无法解决的问题（一般是搜不到资源）时，调用这个工具，它会向人类请示并返回人类的反馈信息
</ToolCalling>

当所有节点添加完毕后，不再调用工具，返回 'FINISH' 代表任务结束
"""


class ToolsOutput(BaseModel):
    """Bioinformatics tools used in the literature"""

    tools: list[str] = Field(description='All tools used in the paper', default_factory=list)
    workflow_structure: str = Field(description="Workflow's Mermaid flowchart code. The flowchart's layout is LR")


def generate_char_index(n: int) -> str:
    """将一个正整数转换为（A, B, ..., Z, AA, AB, ...）形式的索引"""
    assert n >= 1, '输入必须是一个大于等于1的正整数'

    result = []
    while n > 0:
        remainder = (n - 1) % 26
        result.append(chr(65 + remainder))
        n = (n - 1) // 26
    return ''.join(reversed(result))


class BuildNode(BaseModel):
    id: str
    type: str
    input_handles: list[Handle]
    output_handles: list[Handle]
    data: NodeData
    command: Optional[str] = None
    x: int
    y: int


class BuildEdge(BaseModel):
    source_node: str = Field(description='Node ID as data source (not tool or database ID)')
    source_handle: str = Field(description='The name value of the corresponding handle on the source node')
    target_node: Optional[str] = Field(
        description='The ID of the target node (i.e., this node). It will be automatically assigned by the program, just leave it blank.',
        default=None,
    )
    target_handle: str = Field(description='The name value of the corresponding handle on the target node')


class AgentState(MessagesState):
    paper_content: str
    paper_tools: ToolsOutput
    checked_tools: Annotated[list[int], operator.add]
    exist_nodes: Annotated[list[BuildNode], operator.add]
    exist_edges: Annotated[list[BuildEdge], operator.add]

    price: Annotated[float, operator.add]


class ReproductionAgent:
    def __init__(self):
        self.struct_llm: ChatOpenAI = gpt_41_mini()
        self.tool_llm: ChatGoogleGenerativeAI = gemini_25_flash()

        self.check_tool_agent = CheckToolAgent().build()

        self.call_tool_node = ToolNode([self.search_tool, self.search_db, self.add_node_by_id, self.ask_human])

    def extract_tools_node(self, state: AgentState):
        """从原始文章中提取出所使用的工具"""

        class StructureOutput(TypedDict):
            raw: AIMessage
            parsed: Optional[ToolsOutput]
            parsing_error: Optional[BaseException]

        if len(state['messages']) == 0:
            human_question = (
                ChatPromptTemplate(messages=[('human', GET_TOOL_USER)]).invoke(state.get('paper_content')).to_messages()
            )
            state['messages'] = human_question

        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=GET_TOOL_SYSTEM), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.struct_llm.with_structured_output(ToolsOutput, include_raw=True)
        output: StructureOutput = chain.invoke({'history': state['messages']})

        return {'paper_tools': output['parsed'], 'messages': [output['raw']]}

    @staticmethod
    def analyse_extract_result_node(state: AgentState) -> Command[Literal['extract_tools_node', 'build_workflow_node']]:
        """工具提取的结果处理

        如果没有找到工具，触发中断询问人类，并将结果返回到提取节点
        否则，将提取的工具和流程图返回给人类，询问意见。如果同意则进入下一步，否则将反馈信息返回并重新提取。
        """

        if len(state['paper_tools'].tools) == 0:
            question = 'I found 0 tools in the paper, please checkout the content you provide.'
            answer = interrupt(question)
            return Command(
                goto='extract_tools_node',
                update={'messages': [AIMessage(content=question), HumanMessage(content=answer)]},
            )

        question = (
            f'I found {len(state["paper_tools"].tools)} tools in the paper, '
            f'Their workflow is shown in the figure below. Do you think there is any problem?\n'
            f'{state["paper_tools"].workflow_structure}'
        )
        answer = interrupt(question)
        if isinstance(answer, bool) and answer:
            state['messages'] = []
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
                        ('human', GET_TOOL_USER.format(paper_content=state['paper_content'])),
                        (
                            'ai',
                            f'I found {len(state["paper_tools"].tools)} tools in the paper: [{",".join(state["paper_tools"].tools)}]. '
                            f'Their general workflow is shown in the figure below:\n{state["paper_tools"].workflow_structure}',
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

    @staticmethod
    @tool
    def search_tool(query: str):
        """Search for the corresponding tool node

        Args:
            query: Tool name, used to search for corresponding tool nodes
        """
        with Session(engine) as session:
            exist_tools: list[ToolData] = tool_crud.get_by_name_fuzzy(session, query, 50, 0)
        if len(exist_tools) == 0:
            return 'No results found'

        tool_nodes = [ToolNodePublic.model_validate(exist_tool.model_dump()) for exist_tool in exist_tools]
        tool_blocks = [
            (
                f'  <tool>\n'
                f'    <name>{t.name}</name>\n'
                f'    <description>{t.description}</description>\n'
                f'    <input_handles>\n'
                f'{handle_formatter(t.input_handles)}\n'
                f'    </input_handles>\n'
                f'    <output_handles>\n'
                f'{handle_formatter(t.output_handles)}\n'
                f'    </output_handles>\n'
                f'    <data>\n'
                f'      <tool_id>{t.id}</tool_id>\n'
                f'      <args>{t.static_params}<args>\n'
                f'    </data>\n'
                f'    <command>{t.complete_command}</command>'
                f'  </tool>'
            )
            for t in tool_nodes
        ]
        return f'Based on the keywords, the following tool nodes are available (To aid in judgment, each tool node has an additional <command> field, indicating the command actually executed by that node):\n<tools>\n{"\n\n".join(tool_blocks)}\n</tools>'

    @staticmethod
    @tool
    def search_db(query: str):
        """Search for the biological databases used by the tool using its name

        Args:
            query: Tool Name for Search
        """
        with Session(engine) as session:
            exist_dbs: list[BioDb] = db_crud.get_by_name_fuzzy(session, query, 10, offset=0)

        if len(exist_dbs) == 0:
            return 'No results found'

        db_blocks = [
            (
                f'<db>\n'
                f'  <id>{db.id}</id>\n'
                f'  <name>{db.name}</name>\n'
                f'  <description>{db.description}</description>\n'
                f'</db>'
            )
            for db in exist_dbs
        ]

        return f'Based on the keywords, the following db nodes are available:\n\n{"\n".join(db_blocks)}'

    @staticmethod
    @tool
    def add_node_by_id(
        node_type: Literal['tool', 'db'],
        pos_x: int,
        pos_y: int,
        selected_id: int,
        input_edges: list[BuildEdge],
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Add nodes to the workflow by template ID, applicable to nodes of type **tool** and **db**

        Args:
            node_type: Type of Node
            pos_x: The x-coordinate of the node's layout on the panel, only positive integer
            pos_y: The y-coordinate of the node's layout on the panel
            selected_id: The ID of the corresponding node source (i.e., tool ID or database ID)
            input_edges: Input edge connection list for this node
        """
        node_count = len(state['exist_nodes'])
        node_id = generate_char_index(node_count + 1)
        if node_type == 'tool':
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
        else:
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

        for edge in input_edges:
            edge.target_node = node_id

        return Command(
            update={
                'exist_nodes': [new_node],
                'exist_edges': input_edges,
                'messages': [
                    ToolMessage(
                        content=f'add new {node_type} node:\n{node_formatter(new_node)}', tool_call_id=tool_call_id
                    )
                ],
            }
        )

    @staticmethod
    @tool
    def ask_human(question: str):
        """Consult humans for advice

        Args:
            question: To understand the specific content of human questions
        """
        answer = interrupt(question)
        return f'Human feedback is: "{answer}"'

    def build_workflow_node(self, state: AgentState):
        """根据返回的结构构建工作流"""
        # TODO 假设这里已经是全部工具了
        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=BUILD_FLOW_SYSTEM), MessagesPlaceholder(variable_name='history')]
        )
        llm = self.tool_llm.bind_tools(
            [self.search_tool, self.search_db, self.add_node_by_id, self.ask_human], tool_choice='auto'
        )
        chain = prompt | llm

        response = chain.invoke({'history': state['messages']})
        return {'messages': [response]}

    @staticmethod
    def should_continue_build(state: AgentState) -> Literal['call_tool_node', 'save_build_node']:
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return 'call_tool_node'

        return 'save_build_node'

    @staticmethod
    def save_build_node(state: AgentState):
        nodes = []
        for n in state['exist_nodes']:
            if n.type == 'tool':
                node_type = 'tool'
            elif n.type == 'db':
                node_type = 'dbInputNode'
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
        for e in state['exist_edges']:
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

        with Session(engine) as session:
            uid = uuid7(as_type='str')
            workflow = Workflow(
                uid=uid,
                name='ai_gen_wf',
                owner_id=1,
                wf_type=WorkflowType.TEMPLATE,
                workflow=workflow_define,
                public=True,
            )

            wf_crud.insert(session, workflow)

        return

    def build(self, memory: BaseCheckpointSaver) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node('extract_tools_node', self.extract_tools_node)
        graph.add_node('analyse_extract_result_node', self.analyse_extract_result_node)
        graph.add_node('build_workflow_node', self.build_workflow_node)
        graph.add_node('call_tool_node', self.call_tool_node)
        graph.add_node('save_build_node', self.save_build_node)

        graph.add_edge(START, 'extract_tools_node')
        graph.add_edge('extract_tools_node', 'analyse_extract_result_node')
        graph.add_conditional_edges('build_workflow_node', self.should_continue_build)
        graph.add_edge('call_tool_node', 'build_workflow_node')
        graph.add_edge('save_build_node', END)

        compiled_graph = graph.compile(checkpointer=memory)
        return compiled_graph

    def visualize(self):
        graph = StateGraph(AgentState)
        graph.add_node('extract_tools_node', self.extract_tools_node)
        graph.add_node('analyse_extract_result_node', self.analyse_extract_result_node)
        graph.add_node('build_workflow_node', self.build_workflow_node)
        graph.add_node('call_tool_node', self.call_tool_node)
        graph.add_node('save_build_node', self.save_build_node)

        graph.add_edge(START, 'extract_tools_node')
        graph.add_edge('extract_tools_node', 'analyse_extract_result_node')
        graph.add_conditional_edges('build_workflow_node', self.should_continue_build)
        graph.add_edge('call_tool_node', 'build_workflow_node')
        graph.add_edge('save_build_node', END)

        compiled_graph = graph.compile()
        compiled_graph.get_graph(xray=1).draw_mermaid_png(
            output_file_path='reproduction_agent.png',
            draw_method=MermaidDrawMethod.API,
            curve_style=CurveStyle.BASIS,
            frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}},
        )


def handle_formatter(handles: list[Handle], space: int = 6) -> str:
    return '\n'.join(
        [
            f'{" " * space}<handle>\n'
            f'{" " * space}  <name>{h.name}</name>\n'
            f'{" " * space}  <description>{h.description}</description>\n'
            f'{" " * space}</handle>'
            for h in handles
        ]
    )


def node_formatter(node: BuildNode) -> str:
    if node.type == 'tool':
        node_data = (
            f'  <data>\n'
            f'    <tool_id>{node.data.tool_id}</tool_id>\n'
            f'    <args>{node.data.args}</args>\n'
            f'  </data>\n'
            f'  <command>{node.command}</command>\n'
        )
    else:
        node_data = f'  <data>\n    <args>{node.data.args}</args>\n  </data>\n'

    input_handles = (
        f'  <input_handles>\n{handle_formatter(node.input_handles, 4)}\n  </input_handles>\n'
        if node.input_handles
        else ''
    )
    output_handles = (
        f'  <output_handles>\n{handle_formatter(node.output_handles, 4)}\n  </output_handles>\n'
        if node.output_handles
        else ''
    )

    return (
        f'<node>\n'
        f'  <id>{node.id}</id>\n'
        f'  <type>{node.type}</type>\n'
        f'  <pos>(x:{node.x}, y:{node.y})</pos>\n'
        f'{input_handles}'
        f'{output_handles}'
        f'{node_data}'
        f'</node>'
    )
