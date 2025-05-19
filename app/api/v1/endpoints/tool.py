from typing import Sequence, Annotated, Optional

from fastapi import APIRouter, Query

import app.crud.tool_tag as tag_crud
import app.crud.tool_group as group_crud
import app.crud.tool as tool_crud
from app.bioinfo.tool_info import DefaultArgs
from app.core.session import SessionDep
from app.models.tools import (
    DockerToolCreate,
    ToolTag,
    SimpleToolPublic,
    ToolGroup,
    ToolPublic,
    ToolArgPublic,
)

router = APIRouter()
tag_router = APIRouter()
group_router = APIRouter()


@tag_router.get('')
def get_exist_tags(session: SessionDep) -> Sequence[ToolTag]:
    return tag_crud.get_all(session)


@group_router.post('')
def add_group(session: SessionDep, group: ToolGroup) -> ToolGroup:
    return group_crud.insert(session, group)


@group_router.get('')
def get_group_list(session: SessionDep, parent_id: Optional[int] = None) -> list[ToolGroup]:
    return group_crud.get_all_recursive(session, parent_id)


@group_router.get('/tools')
def get_group_tools(
    session: SessionDep,
    parent_id: Optional[int] = None,
    limit: Annotated[int, Query(le=20)] = 20,
    offset: int = 0,
) -> list[SimpleToolPublic]:
    return tool_crud.get_all_by_group(session, parent_id, limit, offset)


@router.post('')
def add_new_tool(tool_prop: DockerToolCreate, session: SessionDep):
    tags_to_add = [ToolTag(name=tag.name) for tag in tool_prop.tool_tag if tag.id == -1]
    exist_ids = [tag.id for tag in tool_prop.tool_tag if tag.id != -1]
    if tags_to_add:
        tags_to_add = tag_crud.insert_list(session, tags_to_add)
        exist_ids.extend([tag.id for tag in tags_to_add])

    tags = tag_crud.get_all_by_id(session, exist_ids)

    tool_data = tool_prop.to_tool_data()
    tool_data.tags = tags

    new_tool = tool_crud.insert(session, tool_data)
    return new_tool


@router.get('/args')
def get_default_args() -> DefaultArgs:
    default_args = DefaultArgs()
    return default_args


@router.get('')
def get_tool_list(
    session: SessionDep, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20
) -> list[SimpleToolPublic]:
    return tool_crud.get_all(session, limit, offset)


@router.get('/search')
def get_tools_by_name(
    session: SessionDep, name: str, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20
) -> list[SimpleToolPublic]:
    return tool_crud.get_by_name_fuzzy(session, name, limit, offset)


@router.get('/{tool_uid}')
def get_tool_detail(session: SessionDep, tool_uid: str) -> Optional[ToolPublic]:
    tool_data = tool_crud.get(session, tool_uid)
    if tool_data:
        return ToolPublic.from_orm(tool_data)

    return None


@router.get('/{tool_uid}/args')
def get_tool_args(session: SessionDep, tool_uid: str) -> ToolArgPublic:
    tool_data = tool_crud.get(session, tool_uid)
    return ToolArgPublic.from_orm(tool_data)
