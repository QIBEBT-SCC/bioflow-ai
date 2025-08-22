import json

from sqlmodel import Session

import app.crud.tools.tool_tag as tag_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.image as image_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool as tool_crud
from app.core.session import engine
from app.models.tools import ToolImage, ToolHelpDoc, ToolDefine, ToolTag, ToolGroup


def export_tags():
    with Session(engine) as session:
        tags: list[ToolTag] = tag_crud.get_all(session)

    json_data = []
    for tag in tags:
        # 我发现model_dump会打乱字段顺序，还是手动做吧
        ordered_tag = {'id': tag.id, 'name': tag.name}
        json_data.append(ordered_tag)

    with open('../presets/tags.json', 'w') as f:
        f.write(json.dumps(json_data, sort_keys=False, indent=2))


def export_groups():
    with Session(engine) as session:
        groups: list[ToolGroup] = group_crud.get_all(session, 80, 0)

    json_data = []
    for group in groups:
        ordered_group = {'id': group.id, 'name': group.name, 'parent_id': group.parent_id, 'tag_id': group.tag_id}
        json_data.append(ordered_group)

    with open('../presets/groups.json', 'w') as f:
        f.write(json.dumps(json_data, sort_keys=False, indent=2))


def export_images():
    with Session(engine) as session:
        images: list[ToolImage] = image_crud.get_all(session, 80, 0)

    json_data = []
    for image in images:
        ordered_image = {
            'id': image.id,
            'name': image.name,
            'version': image.version,
            'description': image.description,
            'homepage': image.homepage,
            'paper_link': image.paper_link,
            'image': image.image.model_dump(),
        }
        json_data.append(ordered_image)

    with open('../presets/images.json', 'w') as f:
        f.write(json.dumps(json_data, sort_keys=False, indent=2))


def export_docs():
    with Session(engine) as session:
        docs: list[ToolHelpDoc] = doc_crud.get_all(session, 80, 0)

    json_data = []
    for doc in docs:
        ordered_doc = {
            'id': doc.id,
            'help_command': doc.help_command,
            'description': doc.description,
            'content': doc.content,
            'image_id': doc.image_id,
            'parent_id': doc.parent_id,
        }
        json_data.append(ordered_doc)

    with open('../presets/tool_docs.json', 'w') as f:
        f.write(json.dumps(json_data, sort_keys=False, indent=2))


def export_tools():
    with Session(engine) as session:
        tools: list[ToolDefine] = tool_crud.get_all(session, 80, 0)

        json_data = []
        for tool in tools:
            ordered_tool = {
                'id': tool.id,
                'name': tool.name,
                'image_id': tool.image_id,
                'description': tool.description,
                'tool_type': tool.tool_type,
                'group_id': tool.group_id,
                'help_doc_id': tool.help_doc_id,
                'command_template': tool.command_template,
                'dynamic_params': [param.model_dump() for param in tool.dynamic_params],
                'static_params': tool.static_params,
                'file_mounts': [output.model_dump() for output in tool.file_mounts],
                'mkdir_output': tool.mkdir_output,
                'use_temp_dir': tool.use_temp_dir,
                'tags': [tag.model_dump() for tag in tool.tags],
            }
            json_data.append(ordered_tool)

        with open('../presets/tools.json', 'w') as f:
            f.write(json.dumps(json_data, sort_keys=False, indent=2))


if __name__ == '__main__':
    export_tags()
    export_groups()
    export_images()
    export_docs()
    export_tools()
