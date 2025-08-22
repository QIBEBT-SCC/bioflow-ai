from typing import Annotated, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from sqlmodel import Session

import app.crud.tools.image as image_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool_tag as tag_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool as tool_crud
import app.crud.resource.bio_db as db_crud
from app.core.session import engine
from app.llm.formatters import handle_formatter
from app.models.resource import BioDb
from app.models.tools import ToolImage, ToolDefine, ToolHelpDoc, ImageConfig, ToolTag, ToolGroup
from app.schemas.tools import ToolNodePublic


@tool
def search_local_image(query: str) -> str:
    """Search the database for images that have already been pulled locally.

    Args:
        query: Search keywords, generally tool name
    """
    with Session(engine) as session:
        images: list[ToolImage] = image_crud.get_by_name_fuzzy(session, query, 50, 0)

    if images:
        image_blocks = [
            (
                f'  <image>\n'
                f'    <id>{image.id}</id>\n'
                f'    <name>{image.name}</name>\n'
                f'    <description>{image.description}</description>\n'
                f'    <url>{image.image.get_image()}</url>\n'
                f'  <image>'
            )
            for image in images
        ]
        return f'Based on the keywords you provided, the following potentially related images were found:\n<images>\n{"\n\n".join(image_blocks)}\n</images>'

    return 'Based on the keywords you provided, no results found'


@tool
def pull_and_save_image(
    name: str, image_url: str, version: str, description: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Pull down the searched images and save relevant information to the database

    Args:
        name: Name of the tool, can contain upper and lower case letters
        image_url: url of the image
        version: version of tool, can be judged by the image tag
        description: A brief description of the tool corresponding to this image
    """
    # 实际上这里并不会拉取镜像，只不过为了让LLM不要乱想我们谎称会拉取♪
    with Session(engine) as session:
        new_image = ToolImage(
            name=name, image=ImageConfig.from_str(image_url), version=version, description=description
        )
        new_image = image_crud.insert(session, new_image)
        image_id = new_image.id

    return Command(
        update={
            'image_id': image_id,
            'image': new_image.image,
            'messages': [
                ToolMessage(
                    content='Available mirrors have been set up, and the next step can begin.',
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def list_docs(image_id: Annotated[int, InjectedState('image_id')]) -> str:
    """List existing document records under the corresponding image"""
    with Session(engine) as session:
        docs: list[ToolHelpDoc] = doc_crud.get_image_docs(session, image_id)

    if docs:
        doc_blocks = [
            (
                f'  <doc>\n'
                f'    <id>{d.id}</id>\n'
                f'    <desc>{d.description}</desc>\n'
                f'    <command>{d.help_command}</command>\n'
                f'  </doc>'
            )
            for d in docs
        ]
        return f'Under this image, we found {len(docs)} corresponding help document records:\n\n<docs>\n{"\n".join(doc_blocks)}\n</docs>'

    return 'No existing help documentation records found under this image.'


@tool
def view_doc(doc_id: int) -> str:
    """View the specific content of the corresponding help document

    Args:
        doc_id: ID of the document to view
    """
    with Session(engine) as session:
        doc: ToolHelpDoc = doc_crud.get(session, doc_id)

    return doc.content


@tool
def save_doc(
    help_command: str,
    description: str,
    content: Annotated[str, InjectedState('last_output')],
    image_id: Annotated[int, InjectedState('image_id')],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Save the results from the last run to the database as help documentation records.

    Args:
        help_command: The command corresponding to this help document
        description: Describe the function or purpose of the subcommand itself, not the action of retrieving its help documentation.
    """
    with Session(engine) as session:
        new_doc = ToolHelpDoc(help_command=help_command, description=description, content=content, image_id=image_id)
        new_doc = doc_crud.insert(session, new_doc)
        doc_id = new_doc.id

    return Command(
        update={
            'doc_id': doc_id,
            'messages': [
                ToolMessage(
                    content=f'The corresponding help document has been found, and its ID is {doc_id}. You can proceed to the next step.',
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def search_tool(query: str) -> str:
    """Search for the corresponding tool node

    Args:
        query: ID of the document to view
    """
    with Session(engine) as session:
        exist_tools: list[ToolDefine] = tool_crud.get_by_name_fuzzy(session, query, 50, 0)

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
                f'      <tool_id>{t.id}</tool_id>\n'
                f'      <args>{t.static_params}<args>\n'
                f'    </data>\n'
                f'    <command>{t.complete_command}</command>'
                f'  </tool>'
            )
            for t in tool_nodes
        ]
        return f'Based on the keywords, the following tool nodes are available (To aid in judgment, each tool node has an additional <command> field, indicating the command actually executed by that node):\n<tools>\n{"\n\n".join(tool_blocks)}\n</tools>'

    return 'Based on the keywords you provided, no results found'


@tool
def view_groups() -> str:
    """View all existing groups"""
    with Session(engine) as session:
        # 如果超过200个group，该考虑下你自己的问题了
        groups = group_crud.get_all(session, 200, 0)

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


@tool
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


@tool
def search_biodb(query: str) -> str:
    """Search for the biological databases used by the tool using its name

    Args:
        query: Keyword for Search
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

    return 'Based on the keywords you provided, no results found'
