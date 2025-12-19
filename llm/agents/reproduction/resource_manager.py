from typing import Annotated, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, ToolException, tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from sqlmodel import Session

from app.core.session import engine
from app.crud.resource import bio_db as db_crud
from app.models.resource import BioDb
from app.schemas.react_flow import NodeData
from app.schemas.tools import Handle
from llm.models import gpt_5_mini
from llm.states import BuildEdge, BuildNode, ReproductionState
from llm.tools.database import search_biodb
from llm.tools.human import ask_human
from llm.utils.formatters import node_formatter
from llm.utils.uid_mapper import generate_char_index


class ResourceManager:
    def __init__(self):
        self.name = 'resource_manager'

        self.tools = [
            search_biodb,
            add_resource_node,
            need_other_node,
            ask_human,
        ]

        self.llm = gpt_5_mini(temperature=0.2).bind_tools(self.tools, tool_choice='auto')

    def build(self):
        graph = StateGraph(state_schema=ReproductionState)

        compiled_graph = graph.compile(name=self.name)
        return compiled_graph


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
async def need_other_node(node_type: Literal['tool', 'processor', 'code', 'note'], requirement: str):
    """

    Args:
        node_type:
        requirement:
    """

    return ''
