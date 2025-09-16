import json
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import CurveStyle
from langchain_core.tools import tool, InjectedToolCallId, ToolException
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.constants import START
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.types import Command, interrupt
from loguru import logger
from sqlmodel import Session

import app.crud.tools.tool as tool_crud
import app.crud.resource.bio_db as db_crud
from llm.models import gemini_25_flash, gpt_5_mini
from app.core.session import engine
from llm.reproduction.create_tool_agent import CreateToolAgent
from llm.utils.formatters import node_formatter
from llm.states import CreateToolState, BuildNode, BuildEdge, ReproductionState, ExtractOutput1, ExtractOutput2
from llm.tools.database import search_tool, search_biodb, save_workflow
from llm.tools.human import ask_human
from app.models.resource import BioDb
from app.schemas.react_flow import NodeData
from app.schemas.tools import ToolNodePublic, Handle

GET_TOOL_SYSTEM_1 = """You are a bioinformatics engineer, and you will be presented with a paper or a text of an oral description.
Your task is to extract the bioinformatics analysis workflow from the content of the literature (or text), organize and summarize it, and output it in a fixed format.
**Do not** analyze the specific implementation details of the workflow, that's not your job.

<OutputFormat>
Your output consists of two parts:
- `overall_summary`: A brief summary of the content provided by the user
- `major_finding`: If you are looking at a complete document, summarize the main conclusions/findings of this document in this section, presented as an **markdown ordered list**; if there is only a description of a bioinformatics analysis task, then this section does not need to be filled in, just leave it blank.
- `workflow_overview`: The analysis workflow(s) described in the paper to obtain the aforementioned results.
    - We **only** concerned with bioinformatics analysis workflows; other wet lab-related methods do not need to be extracted.
    - If there is only one analysis workflow, describe it directly; if there are multiple, list them one by one using **markdown list**. See the <WorkflowOverviewExample> section for the specific format
    - For each workflow, you need to summarize:
        1. Explain the function of this workflow from a purely bioinformatics perspective.
        2. (If `major_finding` has content) Combine with the main conclusions of the article to explain how the results of this workflow support or prove these conclusions.
    - We consider it a workflow as long as the relevant analyses are in the same directed acyclic graph. Only analyses where upstream and downstream data are completely unrelated can be considered different workflows. **Avoid unnecessary splitting.**
    - You may see users asking to reproduce the workflow, but this is a subsequent task. You currently only need to briefly describe the workflow, without describing specific implementation details (such as what tools were used, what parameters were used, etc.)

The language of your generated output should depends on the language of the user's query.
</OutputFormat>

<WorkflowOverviewExample>
Based on the information you provided, N bioinformatics analysis workflows were found:
1. **(e.g., single-cell transcriptome annotation)**
    - **What does this workflow do?**: ...
    - (Optional) **How do the analysis results support the conclusion?**: ...
2. (If there are multiple workflows, the format is the same as above)
...
</WorkflowOverviewExample>
"""

GET_TOOL_SYSTEM_2 = """You are a bioinformatics engineer, and you will be presented with a paper or a text of an oral description.
Your task is to extract the bioinformatics analysis workflow from the content of the literature (or text), organize and summarize it, and output it in a fixed format.

<ThinkingProcess>
To accomplish this task, you can proceed in two steps:
1. First, briefly summarize this document (requirements) and describe the general situation of the extractable workflow.
2. After receiving confirmation, further extract specific tools and workflow structures, organize them into the corresponding format; at the same time, use mermaid to draw a flowchart of the workflow to assist user understanding.
</ThinkingProcess>

<OutputFormat>
Your output should consists these parts:

- The extracted tools, specifically, are the following three lists:
    - `tools`: bioinformatics tools, generally referring to those installed via conda or directly as binary packages
    - `r_packages` and `python_packages`: specific programming language packages used for downstream analysis
- `workflow_details`: Based on the specific information extracted from the files and articles, describe the composition of the workflow in detail using a markdown list.
    - Each list item describes one operation, with the text format "(Tool Type): Use (Tool Name) to ...". Tool Type must be one of `tools`, `r packages`, or `python packages`.
    - List detailed information related to tool parameter selection from the original text using sub-lists
    - If there is only one workflow, return the list directly; if there are multiple workflows, summarize an appropriate title for each workflow, separated by markdown level-three headings (###).
- `mermaid_code`: Workflow flowchart drawn using Mermaid
    - Starts with "```mermaid" and ends with "```" (that is, the standard markdown code block format), does not contain any other text in the task.
    - Use default nodes to represent analysis tools (`id`)
    - Use an asymmetric shape node to represent intermediate files (`id>file description]`)
    - Use a subroutine shape node to represent R or Python packages (`id[[tool/package name]]`)
    - Note that the node display **only needs** the most basic tool/package name and file description. Version numbers, parameters, and other details have already appeared in the description above and **should not be** displayed in the node.
    - For Mermaid to render correctly, **do not use any special symbols** (especially parentheses, quotation marks, etc.) in the node display content, other than those required by the syntax.
</OutputFormat>

To help you understand the output style, here is an example for your reference.
<Example>
{
    "tools": ["fastp","STAR","featureCounts"],
    "r_packages": ["DESeq2","clusterProfiler"],
    "python_packages": [],
    "workflow_details": "1.  (tool): Perform quality control on raw data using `fastp`\\n    * Filter out reads shorter than 35bp\\n2.  (tool): Align high-quality reads to the reference genome using `STAR`\\n3.  (tool): Quantify gene expression using `featureCounts`\\n    * Count at the gene level, summarizing reads aligned to exons\\n4.  (r package): Perform differential expression analysis using the `DESeq2` R package\\n5.  (r package): Perform functional enrichment analysis using the `clusterProfiler` R package",
    "mermaid_code": "```mermaid\\ngraph LR;\\n    A>Raw FASTQ] --> B(fastp);\\n    B --> C>Clean FASTQ];\\n    C --> D(STAR);\\n    D --> E>Aligned BAM];\\n    E --> F(featureCounts);\\n    F --> G>Counts Matrix];\\n    G --> H[[DESeq2]];\\n    H --> I>DEG List];\\n    I --> J[[clusterProfiler]];\\n    J --> K>Enrichment Results];\\n```"
}
</Example>
"""

BUILD_FLOW_SYSTEM = """You are a bioinformatics researcher helping set up software tools for a research group's server.
Your task is to analyze academic text and extract information about bioinformatics tools mentioned in the literature.

<NodeDefine>
This system encapsulates all tools, resources, operations, etc., into nodes. Each node includes the following five parts:
- **id**: Each time a new node is added, an ID is automatically assigned to it.
- **type**: The type of node, including four categories: tool, resource, data_process, code, and note.
    - tool: Bioinformatics tool node.
    - resource: Resource node, generally representing static file resources, including raw sequencing data, databases, reference genomes, etc.
    - data_process: Encapsulation of some data operations, such as copying, concat, and further data filtering of files, which cannot be achieved by bioinformatics tools themselves.
    - code: Different from tool, it is a node used for R or Python code analysis and execution.
    - note: Notes, used to add supplementary explanations on the workflow canvas.
- **input_handles**: Inputs accepted by the node.
- **output_handles**: Outputs generated by the node.
- **data**: Other data of the node itself, with different specific formats corresponding to different node types.

In addition, for ease of understanding, tool-type nodes also have a command attribute, which displays the complete running command of the node.
</NodeDefine>

<ThinkingProcess>
Your task is to call the corresponding tools and add nodes to the workflow step by step based on the content of the original article.
You may have noticed and been bothered by the `data` field of the nodes. In fact, we have a complete set of tools encapsulated for this. You only need to pass simple parameters to the tools, and the system can automatically complete the `data` content based on existing data. You just need to follow these steps:
1. Based on the historical progress, decide what node to add next. Follow the tool calling order: add pre-requisite tools first, then the subsequent ones; add the database first, then the tool itself. In short, the principle to follow is: all input sources for the node currently being added must exist.
2. Next, call the appropriate search tool (starting with `search_`) and select the appropriate "node template" from the returned results.
    - If there are available nodes at this step, proceed directly to the next step.
    - When no available nodes are found, follow the advice in the tool feedback for further processing.
3. Call the add node tool (starting with `add_`) to add the corresponding type of node to the workflow. At this step, you mainly need to decide two things: the coordinates of the node and the input edge connections of the node.
    - Assign appropriate coordinates to the new node based on the coordinates already occupied by existing nodes and the overall structure of the workflow.
    - Use the output of existing tools to define the input edge connections to this node.
    - Specific descriptions can be found in the <ToolCalling> section.
4. When adding nodes, you may find that some necessary pre-analysis nodes are missing due to initial oversight. In this case, you can temporarily postpone adding that node and first add the required pre-requisite nodes.
5. If you encounter an unexpected situation at any point, you can call the `ask_human` tool to ask for human advice.

In this way, when you have added all the nodes one by one, you will actually have the complete workflow, and your task will be completed.
Initially, a `sequence` resource node has been added to the entire workflow, outputting paired-end reads data for r1 and r2, which can usually serve as the starting point for a conventional bioinformatics workflow. If this input is not suitable, it is recommended to first ask for human advice.
</ThinkingProcess>

<ToolCalling>
To help you better search for nodes and build workflows, we have defined the following tools:
1. Tool node related
    - `search_tool`: Used to search for tool nodes, using the tool name for searching.
    - `add_tool_node`: Used to add tool nodes to the workflow. You need to select a suitable object from the search results and pass its ID to this tool. In addition, you mainly need to consider the following two points:
        - Node position: Starting with the initial read node, the remaining nodes spread out in the positive direction of the x-axis. Tools at the same level can be placed at different y-coordinates on the same x-coordinate. In short, follow these two points:
            - Only one node can be placed at one coordinate.
            - Each change is made in units of 1, e.g., (the previous tool B is at [1, 0], the next tool C based on B's output is placed at [2, 0]; tool D is also based on B's output, but [2, 0] already has a node, so it is placed at [2, 1]).
        - Node input edges: A list of edges corresponding to all required input_handles for this node. Each edge mainly looks at which node's output_handle it comes from and which input_handle it goes into on this node.
    - `create_tool_node`: Call this tool to create a new tool node when `search_tool` does not find any available tool nodes.
2. Resource node related
    - `search_biodb`: Used to search for database nodes, generally by searching using the name of the tool that calls this database. If not found, you can try searching directly using the database's own name.
    - `add_resource_node`: Used to add resource nodes to the workflow. The thought process for adding is similar to tool nodes.
3. Data process node related
    - `search_processor`: Searches for available data processing nodes. Operations outside of tools and downstream code analysis are basically handled by data_process nodes.
    - `add_data_process_node`: Adds data processing nodes to the workflow.
4. Other nodes
    - `add_code_node`: Add a code node to the workflow, more information about this node can be found in the <CodeNode> section.
    - `add_note_node`: Note nodes have no input or output, only need to be placed in an appropriate position.
5. Other tools
    - `ask_human`: When encountering any unsolvable problems (resource not found, network connection failed, etc.), call this tool. It will consult a human and return human feedback.
    - `compress_history`: After **successfully** executing the `add_tool_node` and `add_code_node` tools, call this tool to compress the conversation context.
    - `save_workflow`: Once the workflow is built, call this tool to save the workflow.
</ToolCalling>

<CodeNode>
Since writing code requires more professional knowledge, this task will be handed over to more professional people.
Your task is to break down the downstream code analysis in the text into appropriate parts, place these code blocks to be implemented in appropriate positions on the canvas, and describe what kind of code is expected to be written in these code blocks to achieve what.

All code blocks have the same input and output:
- The only input handle `input_files` is all external files that the code needs to read. You only need to directly connect the output generated by the corresponding node to this handle.
- The only output handle `output_folder` is the folder where all output files generated by the code are located. If other code blocks need to use it, it can be directly connected as an output to the `input_files` of the corresponding code block. If other types of nodes need to use it, you may need to add corresponding `data_process` nodes to further utilize the files in that folder.
</CodeNode>

<InputEdges>
<InputEdges>
"""


class ReproductionAgent:
    def __init__(self):
        self.struct_llm = gpt_5_mini(0.2)
        self.tool_llm = gemini_25_flash(0.2)

        # tools
        self.tools = [
            search_tool,
            create_tool_node,
            add_tool_node,
            search_biodb,
            add_resource_node,
            search_processor,
            add_data_process_node,
            add_code_node,
            add_note_node,
            ask_human,
            compress_history,
            save_workflow,
        ]
        self.call_tool_node = ToolNode(self.tools)

    async def summary_paper_node(self, state: ReproductionState):
        writer = get_stream_writer()
        writer({'info': 'Summarizing literature...'})

        class StructureOutput(TypedDict):
            raw: AIMessage
            parsed: Optional[ExtractOutput1]
            parsing_error: Optional[BaseException]

        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=GET_TOOL_SYSTEM_1), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.struct_llm.with_structured_output(ExtractOutput1, include_raw=True)
        output: StructureOutput = await chain.ainvoke({'history': state.messages})

        if output['parsed'].major_finding:
            formatted_msg = (
                f'## Overall Summary\n'
                f'{output["parsed"].overall_summary}\n\n'
                f'## Major Finding\n'
                f'{output["parsed"].major_finding}\n\n'
                f'## Workflow Overview\n'
                f'{output["parsed"].workflow_overview}'
            )
        else:
            formatted_msg = (
                f'## Overall Summary\n'
                f'{output["parsed"].overall_summary}\n\n'
                f'## Workflow Overview\n'
                f'{output["parsed"].workflow_overview}'
            )

        return {'formatted_msg': formatted_msg, 'messages': [output['raw']]}

    async def extract_tools_node(self, state: ReproductionState):
        """从原始文章中提取出所使用的工具"""
        writer = get_stream_writer()
        writer({'info': 'Extracting workflow from the literature...'})

        class StructureOutput(TypedDict):
            raw: AIMessage
            parsed: Optional[ExtractOutput2]
            parsing_error: Optional[BaseException]

        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=GET_TOOL_SYSTEM_2), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.struct_llm.with_structured_output(ExtractOutput2, include_raw=True)
        output: StructureOutput = await chain.ainvoke({'history': state.messages})
        formatted_msg = (
            f'## Workflow Details\n'
            f'{output["parsed"].workflow_details}\n\n'
            f'## Workflow Structure\n'
            f'{output["parsed"].mermaid_code}'
        )

        return {'paper_tools': output['parsed'], 'formatted_msg': formatted_msg, 'messages': [output['raw']]}

    @staticmethod
    def analyse_extract_result_node(
        state: ReproductionState,
    ) -> Command[Literal['summary_paper_node', 'extract_tools_node', 'build_workflow_node']]:
        """工具提取的结果处理

        如果没有找到工具，触发中断询问人类，并将结果返回到提取节点
        否则，将提取的工具和流程图返回给人类，询问意见。如果同意则进入下一步，否则将反馈信息返回并重新提取。
        """

        answer = interrupt({'confirm': f"{state.formatted_msg}\n\nDo you think it's okay? Can we continue?"})
        if isinstance(answer, bool) and answer:
            if state.analyse_step == 0:
                filtered_history = [
                    m if isinstance(m, HumanMessage) else AIMessage(content='Blocked history messages')
                    for m in state.messages[1:-1]
                ]
                return Command(
                    goto='extract_tools_node',
                    update={
                        'analyse_step': 1,
                        'messages': [
                            *[RemoveMessage(id=m.id) for m in state.messages[1:]],
                            *filtered_history,
                            ('ai', state.formatted_msg),
                            ('human', 'Very good, now continue to complete the remaining steps.'),
                        ],
                    },
                )
            else:
                init_node = BuildNode(
                    id='A',
                    type='resource_sequence',
                    input_handles=[],
                    output_handles=[
                        Handle(name='r1', description='the first read of a paired-end sequence.'),
                        Handle(name='r2', description='the second read of a paired-end sequence.'),
                    ],
                    data=NodeData(args='{"r1":"","r2":""}'),
                    x=0,
                    y=0,
                )
                workflow_description = state.messages[-1].content
                return Command(
                    goto='build_workflow_node',
                    update={
                        'analyse_step': 1,
                        'exist_nodes': [init_node],
                        'messages': [
                            RemoveMessage(id=REMOVE_ALL_MESSAGES),
                            ('human', workflow_description),
                            (
                                'ai',
                                f'I found {len(state.paper_tools.tools)} tools({",".join(state.paper_tools.tools)}), '
                                f'{len(state.paper_tools.r_packages)} r packages({", ".join(state.paper_tools.r_packages)}) '
                                f'and {len(state.paper_tools.python_packages)} python packages({", ".join(state.paper_tools.python_packages)}) in the paper.\n'
                                f'Their general workflow is shown in the figure below:\n{state.paper_tools.mermaid_code}',
                            ),
                            (
                                'human',
                                f'Prefect, now start calling tools to perform specific workflow construction tasks. The following nodes have been initially added:\n{node_formatter(init_node)}',
                            ),
                        ],
                    },
                )

        next_node = 'summary_paper_node' if state.analyse_step == 0 else 'extract_tools_node'
        return Command(goto=next_node, update={'messages': [HumanMessage(content=answer)]})

    async def build_workflow_node(self, state: ReproductionState):
        """根据返回的结构构建工作流"""
        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=BUILD_FLOW_SYSTEM), MessagesPlaceholder(variable_name='history')]
        )
        llm = self.tool_llm.bind_tools(self.tools)
        chain = prompt | llm

        response = await chain.ainvoke({'history': state.messages})
        if state.workflow_uid:
            writer = get_stream_writer()
            writer({'link': f'/editor/{state.workflow_uid}'})
        return {'messages': [response]}

    @staticmethod
    def tool_call_route(
        state: ReproductionState,
    ) -> Literal['call_tool_node', 'compress_message_node', 'fix_toolcall_miss_node', '__end__']:
        last_message = state.messages[-1]
        assert isinstance(last_message, AIMessage)

        if not last_message.tool_calls:
            return '__end__' if state.workflow_uid else 'fix_toolcall_miss_node'

        return 'compress_message_node' if last_message.tool_calls[0]['name'] == 'compress_history' else 'call_tool_node'

    @staticmethod
    async def fix_toolcall_miss_node(state: ReproductionState):
        return {'messages': [HumanMessage('Okay, continue calling tools to complete the work.')]}

    @staticmethod
    def compress_message_node(state: ReproductionState):
        """压缩对话上下文"""
        wf_data = {
            'nodes': [n.model_dump() for n in state.exist_nodes],
            'edges': [e.model_dump() for e in state.exist_edges],
        }
        wf_str = json.dumps(wf_data, indent=2, sort_keys=False)

        # human: origin question (keep)
        # ai: description (keep)
        # ....(remove)
        # ai: call add_node (keep)
        # tool: add node (keep)
        # - ai: message (remove)
        #   human: fix (remove)
        # ai: call compress (remove)
        if isinstance(state.messages[-2], HumanMessage):
            remove_list = [
                *[RemoveMessage(id=m.id) for m in state.messages[2:-5]],
                *[RemoveMessage(id=m.id) for m in state.messages[-3:]],
            ]
        else:
            remove_list = [
                *[RemoveMessage(id=m.id) for m in state.messages[2:-3]],
                RemoveMessage(id=state.messages[-1].id),
            ]

        return {
            'messages': [
                *remove_list,
                (
                    'human',
                    f'Prefect, now the following are some of the workflows that have been completed:\n<workflow>\n{wf_str}\n</workflow>\n\nPlease continue to call the tools to work.',
                ),
            ]
        }

    def build(self, memory: BaseCheckpointSaver, visualize: bool = False) -> CompiledStateGraph:
        graph = StateGraph(ReproductionState)
        graph.add_node('summary_paper_node', self.summary_paper_node)
        graph.add_node('extract_tools_node', self.extract_tools_node)
        graph.add_node('analyse_extract_result_node', self.analyse_extract_result_node)
        graph.add_node('build_workflow_node', self.build_workflow_node)
        graph.add_node('call_tool_node', self.call_tool_node)
        graph.add_node('compress_message_node', self.compress_message_node)
        graph.add_node('fix_toolcall_miss_node', self.fix_toolcall_miss_node)

        graph.add_edge(START, 'summary_paper_node')
        graph.add_edge('summary_paper_node', 'analyse_extract_result_node')
        graph.add_edge('extract_tools_node', 'analyse_extract_result_node')
        graph.add_conditional_edges('build_workflow_node', self.tool_call_route)
        graph.add_edge('call_tool_node', 'build_workflow_node')
        graph.add_edge('compress_message_node', 'build_workflow_node')
        graph.add_edge('fix_toolcall_miss_node', 'build_workflow_node')

        compiled_graph = graph.compile(checkpointer=memory)
        if visualize:
            mermaid_code = compiled_graph.get_graph(xray=2).draw_mermaid(
                curve_style=CurveStyle.BASIS, frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}}
            )
            with open('images/reproduction_agent.md', 'w') as f:
                f.write(f'```mermaid\n{mermaid_code}\n```')
        return compiled_graph


@tool
async def create_tool_node(request: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Create new tool node

    Args:
        request: Node creation requirements. Format: "Using XXX (tool name) to XXX (specific task description, including what you want to do with this tool, what the tool's input is, and what the main expected output is)"
    """

    writer = get_stream_writer()
    writer({'info': f'Creating new tool: {request}'})

    sub_agent = CreateToolAgent().build()
    sub_agent_input = CreateToolState(messages=[HumanMessage(content=request)])
    response = await sub_agent.ainvoke(sub_agent_input)

    return Command(
        update={
            'id_mapper': response['id_mapper'],
            'messages': [ToolMessage(response['messages'][-1].content, tool_call_id=tool_call_id)],
        }
    )


@tool
async def add_tool_node(
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
    tool_uid = state.id_mapper.get_uid_by_index(selected_id)

    with Session(engine) as session:
        db_tool = tool_crud.get(session, tool_uid)
        db_node = ToolNodePublic.model_validate(db_tool.model_dump())
        new_node = BuildNode(
            id=node_id,
            type='tool',
            input_handles=db_node.input_handles,
            output_handles=db_node.output_handles,
            data=NodeData(tool_uid=tool_uid, args=db_node.static_params),
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
async def add_resource_node(
    resource_type: Literal['db', 'genome'],
    pos_x: int,
    pos_y: int,
    selected_id: Optional[int],
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add a resource node to the workflow

    Args:
        resource_type: Resource node type (db: biological database; genome: reference genome)
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
                    type='resource_db',
                    input_handles=[],
                    output_handles=[Handle(name='file_path', description='db file path')],
                    data=NodeData(args=f'db:{db_db.name}'),
                    x=pos_x,
                    y=pos_y,
                )
        case 'genome':
            new_node = BuildNode(
                id=node_id,
                type='resource_genome',
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
                ToolMessage(content=f'add new resource node:\n{node_formatter(new_node)}', tool_call_id=tool_call_id)
            ],
        }
    )


@tool
async def search_processor(query: str):
    """Find the data_process node for the relevant function

    Args:
        query: Briefly describe the function you need to search for nodes
    """
    logger.debug(query)
    return (
        'Based on your needs, no suitable data_process node was found. '
        'You can consider changing your search query or using the `ask_human` tool to report this situation to a human.'
    )


@tool
async def add_data_process_node(
    processor_id: int,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add a data_process node to the workflow

    Args:
        processor_id: program language(R or python)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        input_edges: Code node only have one edge `files`, just need to connect the files generated during the preliminary analysis, which are required by the code here, to this handle.
    """
    node_count = len(state.exist_nodes)
    node_id = generate_char_index(node_count + 1)

    # TODO: 这里实际应该是处理器名称
    new_node = BuildNode(
        id=node_id,
        type=f'processor_{processor_id}',
        input_handles=[Handle(name='files', description='The files required by this code')],
        output_handles=[],
        data=NodeData(args='', tool_uid=''),
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
                    content=f'add new data_process node:\n{node_formatter(new_node)}', tool_call_id=tool_call_id
                )
            ],
        }
    )


@tool
async def add_code_node(
    language: Literal['R', 'python'],
    pos_x: int,
    pos_y: int,
    program_request: str,
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add a code node to the workflow

    Args:
        language: program language(R or python)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        program_request: What functionality needs to be implemented in the code. The description needs to be **accurate**, as subsequent work will directly base code writing on your description.
        input_edges: Code node only have one edge `files`, just need to connect the files generated during the preliminary analysis, which are required by the code here, to this handle.
    """
    node_count = len(state.exist_nodes)
    node_id = generate_char_index(node_count + 1)

    new_node = BuildNode(
        id=node_id,
        type=f'code_{language}',
        input_handles=[Handle(name='input_files', description='The files required by this code')],
        output_handles=[Handle(name='output_folder', description='The files required by this code')],
        data=NodeData(args=f'{{"description":"{program_request}","code":""}}'),
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
                ToolMessage(content=f'add new code node:\n{node_formatter(new_node)}', tool_call_id=tool_call_id)
            ],
        }
    )


@tool
async def add_note_node(
    note_content: str,
    pos_x: int,
    pos_y: int,
    input_edges: list[BuildEdge],
    state: Annotated[ReproductionState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Add a note node to the workflow

    Args:
        note_content: Specific content of the notes (as brief and precise as possible)
        pos_x: The x-coordinate of the node's layout on the panel, only positive integer
        pos_y: The y-coordinate of the node's layout on the panel
        input_edges: Code node only have one edge `files`, just need to connect the files generated during the preliminary analysis, which are required by the code here, to this handle.
    """
    node_count = len(state.exist_nodes)
    node_id = generate_char_index(node_count + 1)

    new_node = BuildNode(
        id=node_id, type='note', input_handles=[], output_handles=[], data=NodeData(args=note_content), x=pos_x, y=pos_y
    )

    for edge in input_edges:
        edge.target_node = node_id

    return Command(
        update={
            'exist_nodes': [new_node],
            'exist_edges': input_edges,
            'messages': [ToolMessage(content=f'add new note:\n{node_formatter(new_node)}', tool_call_id=tool_call_id)],
        }
    )


@tool
async def compress_history():
    """Compress context to refine relevant information"""
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
