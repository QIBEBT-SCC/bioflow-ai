from typing import Sequence, Annotated, Optional

from fastapi import APIRouter, Query, Form, Depends
from loguru import logger
from starlette.responses import StreamingResponse

import app.crud.tool_tag as tag_crud
import app.crud.tool_group as group_crud
import app.crud.tool as tool_crud
from app.api.sse_base import SSEMessage, EventType
from app.core.session import SessionDep
from app.llm.llm_function import generate_tool_config
from app.models.tools import ToolTag, ToolGroup, ImageConfig
from app.models.user import User
from app.schemas.tools import DockerToolCreate, SimpleToolPublic, ToolPublic, ToolArgPublic, ToolTaskInfo, AIGenProp
from app.utils.container.podman import run_podman
from app.utils.security import get_current_active_user

router = APIRouter()
tag_router = APIRouter()
group_router = APIRouter()


@tag_router.get('')
async def get_exist_tags(session: SessionDep) -> Sequence[ToolTag]:
    return tag_crud.get_all(session)


@group_router.post('')
async def add_group(session: SessionDep, group: ToolGroup) -> ToolGroup:
    return group_crud.insert(session, group)


@group_router.get('')
async def get_group_list(session: SessionDep, parent_id: Optional[int] = None) -> list[ToolGroup]:
    return group_crud.get_all_recursive(session, parent_id)


@group_router.get('/tools', response_model=list[SimpleToolPublic])
async def get_group_tools(
    session: SessionDep, parent_id: Optional[int] = None, limit: Annotated[int, Query(le=20)] = 20, offset: int = 0
) -> list[SimpleToolPublic]:
    return tool_crud.get_all_by_group(session, parent_id, limit, offset)


@router.post('')
async def add_new_tool(tool_prop: DockerToolCreate, session: SessionDep):
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


@router.get('', response_model=list[SimpleToolPublic])
async def get_tool_list(
    session: SessionDep, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20
) -> list[SimpleToolPublic]:
    return tool_crud.get_all(session, limit, offset)


@router.get('/search', response_model=list[SimpleToolPublic])
def get_tools_by_name(
    session: SessionDep, name: str, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20
) -> list[SimpleToolPublic]:
    return tool_crud.get_by_name_fuzzy(session, name, limit, offset)


@router.post('/ai/config')
async def gen_tool_config(
    prop: Annotated[AIGenProp, Form()], current_user: Annotated[User, Depends(get_current_active_user)]
):
    async def generator():
        yield SSEMessage(event=EventType.LOADING, data='reading tool help doc...').to_sse()
        code, help_doc = run_podman(ImageConfig(repository=prop.repository, tag=prop.tag), prop.help_command, [])
        if help_doc:
            yield SSEMessage(event=EventType.GENERATING, data='generating tool config...').to_sse()
            logger.debug(f'generating tool config for {prop.repository}:{prop.tag}')
            tool_config = await generate_tool_config(prop.name, help_doc, prop.description)
            logger.debug(tool_config)
            yield SSEMessage(event=EventType.SUCCESS, data=tool_config.model_dump()).to_sse()
        else:
            logger.error(f'help doc is empty for {prop.repository}:{prop.tag}')
            yield SSEMessage(event=EventType.ERROR, data='help doc is empty').to_sse()

    return StreamingResponse(generator(), media_type='text/event-stream')


@router.get('/{tool_uid}', response_model=ToolPublic)
async def get_tool_detail(session: SessionDep, tool_uid: str) -> ToolPublic:
    return tool_crud.get(session, tool_uid)


@router.get('/{tool_uid}/args', response_model=ToolArgPublic)
async def get_tool_args(session: SessionDep, tool_uid: str) -> ToolArgPublic:
    return tool_crud.get(session, tool_uid)


@router.get('/{tool_uid}/task_info', response_model=ToolTaskInfo)
async def get_tool4task(session: SessionDep, tool_uid: str):
    return tool_crud.get(session, tool_uid)
