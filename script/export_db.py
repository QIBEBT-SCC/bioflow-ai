import json

from sqlmodel import Session

import app.crud.tools.tool_tag as tag_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.image as image_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool as tool_crud
from app.core.session import engine
from app.models.tools import ToolImage, ToolHelpDoc, ToolData, ToolTag, ToolGroup


def export_tags():
    with Session(engine) as session:
        tags: list[ToolTag] = tag_crud.get_all(session)

    json_data = [tag.model_dump() for tag in tags]
    with open('../presets/tags.json', 'w') as f:
        f.write(json.dumps(json_data))


def export_groups():
    with Session(engine) as session:
        groups: list[ToolGroup] = group_crud.get_all(session, 80, 0)

    json_data = [group.model_dump() for group in groups]
    with open('../presets/groups.json', 'w') as f:
        f.write(json.dumps(json_data))


def export_images():
    with Session(engine) as session:
        images: list[ToolImage] = image_crud.get_all(session, 80, 0)

    json_data = [image.model_dump() for image in images]
    with open('../presets/images.json', 'w') as f:
        f.write(json.dumps(json_data))


def export_docs():
    with Session(engine) as session:
        docs: list[ToolHelpDoc] = doc_crud.get_all(session, 80, 0)

    json_data = [doc.model_dump() for doc in docs]
    with open('../presets/tool_docs.json', 'w') as f:
        f.write(json.dumps(json_data))


def export_tools():
    with Session(engine) as session:
        tools: list[ToolData] = tool_crud.get_all(session, 80, 0)

        json_data = []
        for tool in tools:
            tool_dict = tool.model_dump(mode='json')
            tool_dict['tags'] = [tag.model_dump() for tag in tool.tags]
            json_data.append(tool_dict)

    with open('../presets/tools.json', 'w') as f:
        f.write(json.dumps(json_data))


if __name__ == '__main__':
    export_tags()
    export_groups()
    export_images()
    export_docs()
    export_tools()
