from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from sqlalchemy.orm import Session

import bioflow_ai.crud.resource.bio_db as db_crud
import bioflow_ai.crud.tools.image as image_crud
import bioflow_ai.crud.tools.tool as tool_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.models.resource import BioDb
from bioflow_ai.services.resource import bio_db as db_service
from bioflow_ai.services.tools import catalog as tool_catalog

if TYPE_CHECKING:
    from bioflow_ai.utils.uid_mapper import BidirectionalMapper


@tool(parse_docstring=True)
def search_local_image(query: str, runtime: ToolRuntime[Any]) -> Command:
    """Search the database for images that have already been pulled locally.

    Args:
        query: Search keywords, generally tool name
    """
    with Session(engine) as session:
        images = image_crud.get_by_name_fuzzy(session, query, 50, 0)

    if images:
        image_blocks = [
            (
                f'  <image>\n'
                f'    <name>{image.name}</name>\n'
                f'    <image_ref>{image.ref}</image_ref>\n'
                f'    <url>{image.image.get_image()}</url>\n'
                + (f'    <aliases>{", ".join(image.aliases)}</aliases>\n' if image.aliases else '')
                + f'    <description>{image.description}</description>\n'
                f'  </image>'
            )
            for image in images
        ]

        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'Based on the keywords you provided, the following potentially related images were found:\n<images>\n{"\n\n".join(image_blocks)}\n</images>',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    if ' ' in query:
        feedback_msg = (
            'Based on the keywords you provided, no results found. '
            'Consider first searching with short relatively concise keywords (e.g., STAR), then gradually adding restrictions.'
        )
    else:
        feedback_msg = 'Based on the keywords you provided, no results found. Try searching with different keywords.'

    return Command(
        update={
            'messages': [
                ToolMessage(
                    content=feedback_msg,
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(parse_docstring=True)
def select_image(
    image_ref: str,
    runtime: ToolRuntime[Any],
) -> Command:
    """Select existing image in the database.

    Args:
        image_ref: Exact `<image_ref>` returned by `search_local_image`.
    """
    with Session(engine) as session:
        tool_image = tool_catalog.resolve_image_reference(session, image_ref)

    if tool_image is None:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='The corresponding image was not found in the local database. Please use the exact `<image_ref>` from `search_local_image` or search again.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    return Command(
        update={
            'image_uid': tool_image.uid,
            'image': tool_image.image,
            'homepage': tool_image.homepage,
            'should_end': True,
            'messages': [
                ToolMessage(
                    content='Image selected.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
def select_classification(group_id: int, tag_ids: list[int], runtime: ToolRuntime[Any]) -> Command:
    """Select a primary Function group and Function/Domain tags using IDs from the prompt.

    Args:
        group_id: Primary Function group ID listed in the prompt.
        tag_ids: Domain tag IDs and any additional Function tag IDs listed in the prompt.
            Include at least one Domain. The primary Function tag is included automatically.
            Exclude Default, AI Checked and AI Unchecked.
    """
    return Command(
        update={
            'group_id': group_id,
            'tag_ids': tag_ids,
            'updated': True,
            'should_end': True,
            'messages': [ToolMessage(content='Tool classification selected.', tool_call_id=runtime.tool_call_id)],
        }
    )


@tool(parse_docstring=True)
def read_tool_doc(tool_id: int, runtime: ToolRuntime[Any]) -> str:
    """Read existing tool's help document.

    Args:
        tool_id: tool ID for which you wish to view the help documentation
    """
    mapper = cast('BidirectionalMapper', runtime.state.get('id_mapper'))
    tool_uid = mapper.get_uid_by_index(tool_id)

    with Session(engine) as session:
        tool_data = tool_crud.get(session, tool_uid)

        if tool_data and tool_data.help_doc:
            doc_content = f'<Doc help_command="{tool_data.help_doc.help_command}">\n{tool_data.help_doc.content}\n<Doc>'
        else:
            doc_content = 'No help documentation was found for this tool.'

    return doc_content


@tool(parse_docstring=True)
def select_tool(tool_id: int, runtime: ToolRuntime[Any]) -> Command:
    """Select existing tool in the database.

    Args:
        tool_id: selected tool id
    """
    mapper = cast('BidirectionalMapper | None', runtime.state.get('id_mapper'))
    tool_uid = mapper.get_uid_by_index(tool_id) if mapper is not None else ''
    if not tool_uid:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content='The corresponding tool ID was not found in the current search results. Search again or use a valid tool_id.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    return Command(
        update={
            'tool_uid': tool_uid,
            'should_end': True,
            'messages': [
                ToolMessage(
                    content='Tool selected.',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
def search_biodb(query: str) -> str:
    """Search for the biological databases used by the tool using its name.

    Args:
        query: Keyword for Search, It can be the tool that uses this database (e.g., Kraken2), or the name of this database (e.g., nr or eggnog).
    """
    with Session(engine) as session:
        exist_dbs = db_crud.get_by_name_fuzzy(session, query, 10, offset=0)

    if exist_dbs:
        db_blocks = [
            (
                f'<db id="{db.id}">\n'
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
def create_biodb(
    name: str,
    description: str,
    download_command: str,
    download_image: str,
    runtime: ToolRuntime[Any],
) -> Command:
    """Add a new downloadable biological database to the system.

    Args:
        name: Unique database name shown to workflow builders.
        description: Functional database description, including source/version/use case when known.
        download_command: Non-interactive shell command that downloads and prepares files under `/data/db`.
        download_image: Container image used to run the command. Use an empty string for the default download image.
    """
    with Session(engine) as session:
        new_db = BioDb(
            name=name, description=description, download_command=download_command, download_image=download_image
        )
        new_db = db_service.create_bio_db(session, new_db)
        update: dict[str, Any] = {
            'should_end': True,
            'messages': [
                ToolMessage(
                    content=(
                        f'Based on your request, a biological database has been added:\n'
                        f'<db id="{new_db.id}">\n'
                        f'  <name>{new_db.name}</name>\n'
                        f'  <description>{new_db.description}</description>\n'
                        f'  <download_command>{new_db.download_command}</download_command>\n'
                        f'  <download_image>{new_db.download_image}</download_image>\n'
                        f'</db>'
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
        if new_db.id is not None:
            update['new_db'] = new_db.id

        return Command(update=update)
