from typing import Sequence

from fastapi import APIRouter

import app.crud.tool_tag as tag_crud
import app.crud.tool as tool_crud
from app.bioinfo.tool_info import DefaultArgs
from app.core.session import SessionDep
from app.models.tools import DockerToolCreate, ToolTag

router = APIRouter()
tag_router = APIRouter()


@tag_router.get('')
def get_exist_tags(session: SessionDep) -> Sequence[ToolTag]:
    return tag_crud.get_all(session)


@router.post('')
def add_new_tool(tool_prop: DockerToolCreate, session: SessionDep):
    tags_to_add = [ToolTag(name=tag.name) for tag in tool_prop.tool_tag if tag.id == -1]
    exist_ids = [tag.id for tag in tool_prop.tool_tag if tag.id != -1]
    if tags_to_add:
        tags_to_add = tag_crud.insert_list(session, tags_to_add)
        exist_ids.extend([tag.id for tag in tags_to_add])

    tags = tag_crud.get_all_by_id(session, exist_ids)
    tags.extend(tags_to_add)

    tool_data = tool_prop.to_tool_data()
    tool_data.tags = tags

    new_tool = tool_crud.insert(session, tool_data)
    return new_tool


@router.get('/args')
def get_default_args() -> DefaultArgs:
    default_args = DefaultArgs()
    return default_args
