import datetime
from typing import Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlmodel import Session

import app.crud.resource.bio_db as db_crud
import app.crud.tools.image as image_crud
import app.crud.tools.tool as tool_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool_tag as tag_crud
from app.core.config import get_settings
from app.core.session import engine
from app.crud import workflow as wf_crud
from app.models.resource import BioDb
from app.models.tools import ImageConfig, ToolDefine, ToolGroup, ToolHelpDoc, ToolImage, ToolTag
from app.models.workflow import Workflow, WorkflowType
from app.schemas.react_flow import Edge, Node, Position, WorkflowDefinition
from app.schemas.tools import ToolNodePublic
from llm.states import UserContext
from llm.utils.formatters import handle_formatter
from llm.utils.uid_mapper import BidirectionalMapper


@tool(parse_docstring=True)
def search_local_image(query: str, runtime: ToolRuntime) -> Command:
    """Search the database for images that have already been pulled locally.
    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        query: Search keywords, generally tool name
    """
    with Session(engine) as session:
        images: list[ToolImage] = image_crud.get_by_name_fuzzy(session, query, 50, 0)
        mappers = BidirectionalMapper(datas=images)

    if images:
        image_blocks = [
            (
                f'  <image>\n'
                f'    <id>{mappers.get_index_by_uid(str(image.uid))}</id>\n'
                f'    <name>{image.name}</name>\n'
                f'    <description>{image.description}</description>\n'
                f'    <url>{image.image.get_image()}</url>\n'
                f'  <image>'
            )
            for image in images
        ]

        return Command(
            update={
                'id_mapper': mappers,
                'messages': [
                    ToolMessage(
                        content=f'Based on the keywords you provided, the following potentially related images were found:\n<images>\n{"\n\n".join(image_blocks)}\n</images>',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            'messages': [
                ToolMessage(
                    content='Based on the keywords you provided, no results found',
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(parse_docstring=True)
def select_image(
    image_id: int,
    runtime: ToolRuntime,
):
    """Select existing image in the database

    Args:
        image_id: selected image id
    """
    mapper: BidirectionalMapper = runtime.state.get('id_mapper')
    image_uid = mapper.get_uid_by_index(image_id)

    with Session(engine) as session:
        tool_image = image_crud.get(session, image_uid)

    return Command(
        update={
            'image_uid': image_uid,
            'image': tool_image.image,
            'homepage': tool_image.homepage,
            'messages': [
                ToolMessage(
                    content='The correct image has been selected, you can summarize the information and end the task.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
async def pull_and_save_image(
    name: str,
    image_url: str,
    version: str,
    description: str,
    runtime: ToolRuntime,
) -> Command:
    """Pull down the searched images and save relevant information to the database

    It should be noted that the tag and version of an image are not necessarily the same; some image tags may contain other content after them.

    Args:
        name: Name of the tool, can contain upper and lower case letters
        image_url: url of the image, the format is [registry]/[namespace]/[repository]:tag
        version: version of tool, can be judged by the image tag
        description: Description of the tool corresponding to this image. This is a description of the tool, not the current requirements.
    """
    # 实际上这里并不会拉取镜像，只不过为了让LLM不要乱想我们谎称会拉取♪
    with Session(engine) as session:
        new_image = ToolImage(
            name=name,
            image=ImageConfig.from_str(image_url),
            version=version,
            description=description,
            homepage=runtime.state.get('homepage', ''),
            paper_link=runtime.state.get('paper_link', ''),
        )
        new_image = image_crud.insert(session, new_image)
        image_uid = new_image.uid

    return Command(
        update={
            'image_uid': image_uid,
            'image': new_image.image,
            'messages': [
                ToolMessage(
                    content="The tool's image has been saved locally, you can summarize and finish the work.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def list_docs(runtime: ToolRuntime) -> Command:
    """List existing document records under the corresponding image"""
    image_uid = runtime.state.get('image_uid')

    with Session(engine) as session:
        docs: list[ToolHelpDoc] = doc_crud.get_by_image(session, str(image_uid))
        mappers = BidirectionalMapper(datas=docs)

    if docs:
        doc_blocks = [
            (
                f'  <doc>\n'
                f'    <id>{mappers.get_index_by_uid(str(d.uid))}</id>\n'
                f'    <desc>{d.description}</desc>\n'
                f'    <command>{d.help_command}</command>\n'
                f'  </doc>'
            )
            for d in docs
        ]

        return Command(
            update={
                'id_mapper': mappers,
                'messages': [
                    ToolMessage(
                        content=f'Under this image, we found {len(docs)} corresponding help document records:\n\n<docs>\n{"\n".join(doc_blocks)}\n</docs>',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            'messages': [
                ToolMessage(
                    content='No existing help documentation records found under this image.',
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(parse_docstring=True)
def view_doc(doc_id: int, runtime: ToolRuntime) -> str:
    """View the specific content of the corresponding help document

    Args:
        doc_id: ID of the document to view
    """
    mapper: BidirectionalMapper = runtime.state.get('id_mapper')
    doc_uid = mapper.get_uid_by_index(doc_id)

    with Session(engine) as session:
        doc: ToolHelpDoc = doc_crud.get(session, doc_uid)

    return doc.content


@tool(parse_docstring=True)
def select_doc(doc_id: int, runtime: ToolRuntime):
    """Select existing help document in the database

    Args:
        doc_id: selected doc id
    """
    mapper: BidirectionalMapper = runtime.state.get('id_mapper')

    return Command(
        update={
            'doc_uid': mapper.get_uid_by_index(doc_id),
            'messages': [
                ToolMessage(
                    content='The corresponding help document has been found, you can summarize the information and end the task.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
def save_doc(
    help_command: str,
    description: str,
    runtime: ToolRuntime,
) -> Command:
    """Save the results of help_command to the database as help documentation records.

    Args:
        help_command: The command corresponding to this help document
        description: Describe the function or purpose of the subcommand itself, not the action of retrieving its help documentation.
    """
    image: ImageConfig = runtime.state.get('image')
    if get_settings().docker.CONTAINER == 'docker':
        from app.utils.container.docker import run_docker

        code, result = run_docker(image, help_command, [])
    else:
        from app.utils.container.podman import run_podman

        code, result = run_podman(image, help_command, [])

    with Session(engine) as session:
        new_doc = ToolHelpDoc(
            help_command=help_command,
            description=description,
            content=result,
            image_uid=runtime.state.get('image_uid'),
        )
        new_doc = doc_crud.insert(session, new_doc)
        doc_uid = new_doc.uid

    return Command(
        update={
            'doc_uid': doc_uid,
            'messages': [
                ToolMessage(
                    content='The corresponding help document has been found, you can summarize the information and end the task.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
def search_tool(query: str, runtime: ToolRuntime) -> Command:
    """Search for the corresponding tool node
    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        query: Name of the tool to search for
    """
    with Session(engine) as session:
        exist_tools: list[ToolDefine] = tool_crud.get_by_name_fuzzy(session, query, 50, 0)
        mappers = BidirectionalMapper(datas=exist_tools)

    if exist_tools:
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
                f'      <tool_id>{mappers.get_index_by_uid(str(t.uid))}</tool_id>\n'
                f'      <args>{t.static_params}<args>\n'
                f'    </data>\n'
                f'    <command>{t.complete_command}</command>'
                f'  </tool>'
            )
            for t in tool_nodes
        ]
        return Command(
            update={
                'id_mapper': mappers,
                'messages': [
                    ToolMessage(
                        f'Based on the keywords, the following tool nodes are available '
                        f'(To aid in judgment, each tool node has an additional <command> field, indicating the command actually executed by that node):\n'
                        f'<tools>\n{"\n\n".join(tool_blocks)}\n</tools>\n\n'
                        f'Note that if you are reproducing the workflow rather than designing it yourself, the selection of tools must strictly follow the original paper (the versions can be different).',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            'messages': [
                ToolMessage(
                    'Based on the keywords you provided, no results found. '
                    'Consider using new search keywords or calling the `create_tool_node` tool to create new tool nodes',
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(parse_docstring=True)
def select_tool(tool_id: int, runtime: ToolRuntime):
    """Select existing tool in the database

    Args:
        tool_id: selected tool id
    """
    mapper: BidirectionalMapper = runtime.state.get('id_mapper')

    return Command(
        update={
            'tool_uid': mapper.get_uid_by_index(tool_id),
            'messages': [
                ToolMessage(
                    content='The corresponding tool has been found, you can summarize the information and end the task.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def view_groups() -> str:
    """View all existing groups"""
    with Session(engine) as session:
        groups = group_crud.get_all(session)

    if groups:

        def build_tree_string(_groups_list, _parent_id=None, _indent=0):
            result = []
            # 找到当前层级的所有分组
            current_level_groups = [g for g in _groups_list if g.parent_id == _parent_id]

            for group in current_level_groups:
                prefix = '    ' * _indent + '- '
                result.append(f'{prefix}{group.id}: {group.name}')

                # 递归添加子分组
                child_result = build_tree_string(_groups_list, group.id, _indent + 1)
                result.extend(child_result)

            return result

        tree_lines = build_tree_string(groups)
        group_str = '\n'.join(tree_lines)
        return group_str

    return 'No groups found'


@tool(parse_docstring=True)
def select_group(group_id: int, runtime: ToolRuntime):
    """Select the corresponding group for the tool

    Args:
        group_id: selected group id
    """

    return Command(
        update={
            'group_id': group_id,
            'messages': [
                ToolMessage(
                    content='The corresponding help document has been found, you can summarize the information and end the task.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
def add_group(name: str, parent_id: Optional[int] = None):
    """create new tool group

    Args:
        name: group name
        parent_id: Leave None if it is a top-level group, otherwise it is the ID of its parent group if it is a sub-group.
    """
    with Session(engine) as session:
        tag = ToolTag(name=name)
        tag = tag_crud.insert(session, tag)

        if parent_id is not None:
            group = ToolGroup(name=name, tag_id=tag.id, parent_id=parent_id)
        else:
            group = ToolGroup(name=name, tag_id=tag.id)
        group = group_crud.insert(session, group)

        return f'Create new group "{group.name}", id: {group.id}'


@tool(parse_docstring=True)
def search_biodb(query: str) -> str:
    """Search for the biological databases used by the tool using its name
    **This tool prohibits parallel calls! You are only allowed to make one search at a time!**

    Args:
        query: Keyword for Search, It can be the tool that uses this database (e.g., Kraken2), or the name of this database (e.g., nr or eggnog).
    """
    with Session(engine) as session:
        exist_dbs: list[BioDb] = db_crud.get_by_name_fuzzy(session, query, 10, offset=0)

    if exist_dbs:
        db_blocks = [
            (
                f'<db>\n'
                f'  <id>{db.id}</id>\n'
                f'  <name>{db.name}</name>\n'
                f'  <description>{db.description}</description>\n'
                f'  <path>{db.path}</path>\n'
                f'</db>'
            )
            for db in exist_dbs
        ]

        return f'Based on the keywords, the following db nodes are available:\n\n{"\n".join(db_blocks)}'

    return (
        'Based on the keywords you provided, no results found. '
        'Consider using new search keywords or add a note node at the corresponding location, explaining that a database is needed here and how to download it.'
    )


@tool(parse_docstring=True)
def save_workflow(name: str, runtime: ToolRuntime[UserContext]):
    """Save the completed workflow into the database

    Args:
        name: Workflow name, brief description of this workflow's tasks
    """
    nodes = []
    for n in runtime.state.get('exist_nodes', []):
        node = Node(
            id=n.id,
            type=n.type,
            data=n.data,
            position=Position(x=n.x * 500, y=n.y * 400),
            dragHandle='.nodeDragable',
            zIndex=-10 if n.type == 'note' else 20,
        )
        nodes.append(node)

    edges = []
    for e in runtime.state.get('exist_edges', []):
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

    now = datetime.datetime.now()
    with Session(engine) as session:
        workflow = Workflow(
            name=f'{now.strftime("%y%m%d")}_{name}(AI)',
            owner_id=runtime.context.user_id,
            wf_type=WorkflowType.TEMPLATE,
            workflow=workflow_define,
            public=True,
        )

        workflow = wf_crud.insert(session, workflow)

    return Command(
        update={
            'workflow_uid': workflow.uid,
            'messages': [
                ToolMessage(content='The workflow has been saved to the database.', tool_call_id=runtime.tool_call_id)
            ],
        }
    )
