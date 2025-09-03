from typing import Sequence, Annotated, Optional

from fastapi import APIRouter, Query, Form, HTTPException
from starlette.responses import StreamingResponse
from starlette.status import HTTP_404_NOT_FOUND

import app.crud.tools.tool_tag as tag_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.image as image_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool as tool_crud
from app.api.sse_base import SSEMessage, EventType
from app.core.session import SessionDep
from app.models.tools import ToolTag, ToolGroup, ToolImage, ToolHelpDoc
from app.schemas.tools import (
    DockerToolCreate,
    SimpleToolPublic,
    ToolPublic,
    ToolNodePublic,
    ToolTaskInfo,
    ToolGroupPublic,
    SimpleToolDoc,
    AiGenRequest,
)
from app.utils.container.podman import run_podman
from app.utils.security import UserDep, AdminDep, MemberDep

router = APIRouter()
tag_router = APIRouter()
group_router = APIRouter()
image_router = APIRouter()
doc_router = APIRouter()


# ===============tag and group===============


@tag_router.get('')
async def get_exist_tags(session: SessionDep) -> Sequence[ToolTag]:
    return tag_crud.get_all(session)


@group_router.post('')
async def add_group(session: SessionDep, group: ToolGroup, current_user: AdminDep) -> ToolGroup:
    group_tag = ToolTag(name=group.name)
    group_tag = tag_crud.insert(session, group_tag)
    group.tag_id = group_tag.id

    return group_crud.insert(session, group)


@group_router.get('', response_model=list[ToolGroupPublic])
async def get_group_list(session: SessionDep) -> list[ToolGroupPublic]:
    """获取工具分组列表，包含每个分组的工具数量"""
    group_data = group_crud.get_all_with_tool_counts(session)
    return [ToolGroupPublic(**group) for group in group_data]


@group_router.get('/tools', response_model=list[SimpleToolPublic])
async def get_group_tools(
    session: SessionDep, parent_id: Optional[int] = None, limit: Annotated[int, Query(le=20)] = 20, offset: int = 0
) -> list[SimpleToolPublic]:
    return tool_crud.get_all_by_group(session, parent_id, limit, offset)


# =============== image ===============
@image_router.post('', response_model=ToolImage)
async def create_image(image: ToolImage, session: SessionDep, current_user: AdminDep):
    return image_crud.insert(session, image)


@image_router.get('')
async def get_images(session: SessionDep, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20):
    return image_crud.get_all(session, limit, offset)


@image_router.get('/search', response_model=list[ToolImage])
def get_images_by_name(session: SessionDep, name: str, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20):
    return image_crud.get_by_name_fuzzy(session, name, limit, offset)


@image_router.get('/{uid}/documents', response_model=list[SimpleToolDoc])
async def get_image_docs(session: SessionDep, uid: str):
    return doc_crud.get_image_docs(session, uid)


@image_router.post('/{uid}')
async def run_in_image(
    uid: str, command: Annotated[str, Form(description='要运行的命令')], session: SessionDep, current_user: MemberDep
):
    image = image_crud.get(session, uid)
    if not image:
        raise HTTPException(HTTP_404_NOT_FOUND, 'No corresponding image found')

    code, result = run_podman(image.image, command, [])
    return {'code': code, 'result': result}


# =============== doc ===============
@doc_router.get('/{uid}', response_model=ToolHelpDoc)
async def get_doc(session: SessionDep, uid: str):
    return doc_crud.get(session, uid)


# =============== tool ===============
@router.post('')
async def add_new_tool(tool_prop: DockerToolCreate, session: SessionDep, current_user: AdminDep):
    tags_to_add = [ToolTag(name=tag.name) for tag in tool_prop.tool_tag if tag.id == -1]
    exist_ids: list[int] = [tag.id for tag in tool_prop.tool_tag if tag.id != -1]
    if tags_to_add:
        tags_to_add = tag_crud.insert_all(session, tags_to_add)
        exist_ids.extend([tag.id for tag in tags_to_add])

    tags = tag_crud.get_all_by_id(session, exist_ids)

    tool_data = tool_prop.to_tool_data()
    tool_data.tags = tags

    new_tool = tool_crud.insert(session, tool_data)
    return new_tool


@router.get('/count')
async def get_tool_count(session: SessionDep) -> int:
    return tool_crud.get_counts(session)


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


@router.post('/generate')
async def gen_tool_config(prop: Annotated[AiGenRequest, Form()], session: SessionDep, current_user: UserDep):
    image = image_crud.get(session, prop.image_uid)

    async def generator():
        yield SSEMessage(event=EventType.LOADING, data='reading tool help doc...').to_sse()

    return StreamingResponse(generator(), media_type='text/event-stream')


@router.get('/{tool_uid}', response_model=ToolPublic)
async def get_tool_detail(session: SessionDep, tool_uid: str) -> ToolPublic:
    return tool_crud.get(session, tool_uid)


@router.get('/{tool_uid}/args', response_model=ToolNodePublic)
async def get_tool_args(session: SessionDep, tool_uid: str) -> ToolNodePublic:
    return tool_crud.get(session, tool_uid)


@router.get('/{tool_uid}/task_info', response_model=ToolTaskInfo)
async def get_tool4task(session: SessionDep, tool_uid: str):
    return tool_crud.get(session, tool_uid)
