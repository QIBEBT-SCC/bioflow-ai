from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_CONTENT

import bioflow_ai.crud.tools.image as image_crud
import bioflow_ai.crud.tools.tool as tool_crud
import bioflow_ai.crud.tools.tool_doc as doc_crud
import bioflow_ai.crud.tools.tool_group as group_crud
import bioflow_ai.crud.tools.tool_tag as tag_crud
from bioflow_ai.core.auth import AdminDep, MemberDep
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.infrastructure.redis.client import RedisDep
from bioflow_ai.schemas.common import Paginated
from bioflow_ai.schemas.tools import (
    DockerToolCreate,
    DockerToolUpdate,
    ImageCreate,
    ImagePublic,
    ImageUpdate,
    RunRequest,
    SimpleToolPublic,
    ToolGroupCreate,
    ToolGroupPublic,
    ToolGroupRecord,
    ToolHelpDocPublic,
    ToolImagePublic,
    ToolNodePublic,
    ToolPublic,
    ToolTagPublic,
    ToolTaskInfo,
    ToolUsagePublic,
)
from bioflow_ai.services import tool_cache, tool_catalog
from bioflow_ai.services.image_references import DuplicateImageUrlError

router = APIRouter()
tag_router = APIRouter()
group_router = APIRouter()
image_router = APIRouter()
doc_router = APIRouter()


# ===============tag and group===============


@tag_router.get('')
async def get_exist_tags(session: SessionDep) -> list[ToolTagPublic]:
    """Get all existing tool tags."""
    return list(tag_crud.get_all(session))  # type: ignore[return-value]


@group_router.post('')
async def add_group(
    session: SessionDep,
    group_data: ToolGroupCreate,
    _current_user: AdminDep,
) -> ToolGroupRecord:
    """Add a new tool group."""
    return tool_catalog.create_group(session, group_data)  # type: ignore[return-value]


@group_router.get('')
async def get_group_list(session: SessionDep) -> list[ToolGroupPublic]:
    """获取工具分组列表, 包含每个分组的工具数量."""
    group_data = group_crud.get_all_with_tool_counts(session)
    return [ToolGroupPublic(**group) for group in group_data]


@group_router.get('/tools')
async def get_group_tools(
    session: SessionDep,
    parent_id: int | None = None,
    limit: Annotated[int, Query(le=20)] = 20,
    offset: int = 0,
) -> list[SimpleToolPublic]:
    """Get tools belonging to a specific group."""
    return tool_crud.get_all_by_group(session, parent_id, limit, offset)  # type: ignore


# =============== image ===============
@image_router.post('')
async def create_image(
    image_data: ImageCreate,
    session: SessionDep,
    _current_user: AdminDep,
) -> ToolImagePublic:
    """Create a new Docker image registry entry."""
    try:
        return tool_catalog.create_image(session, image_data)  # type: ignore[return-value]
    except DuplicateImageUrlError as error:
        raise HTTPException(HTTP_409_CONFLICT, str(error)) from error


@image_router.get('')
async def get_images(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[ToolImagePublic]:
    """Get a paginated list of Docker images."""
    images, total = image_crud.get_page(session, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[ToolImagePublic.model_validate(image) for image in images],
    )


@image_router.get('/search')
def get_images_by_name(
    session: SessionDep,
    name: str,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[ToolImagePublic]:
    """Search Docker images by name or alias (fuzzy match)."""
    images, total = image_crud.get_page_by_name_fuzzy(session, name, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[ToolImagePublic.model_validate(image) for image in images],
    )


@image_router.get('/{uid}')
async def get_image(session: SessionDep, uid: str) -> ImagePublic:
    """Get a Docker image entry by UID."""
    image = image_crud.get(session, uid)
    if not image:
        raise HTTPException(HTTP_404_NOT_FOUND, 'No corresponding image found')
    return ImagePublic.model_validate(image)


@image_router.patch('/{uid}')
async def update_image(
    uid: str,
    image_update: ImageUpdate,
    session: SessionDep,
    _current_user: AdminDep,
) -> ToolImagePublic:
    """更新镜像信息."""
    try:
        updated_image = tool_catalog.update_image(session, uid, image_update)
    except DuplicateImageUrlError as error:
        raise HTTPException(HTTP_409_CONFLICT, str(error)) from error
    if updated_image is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Image not found')
    return updated_image  # type: ignore[return-value]


@image_router.delete('/{uid}')
async def delete_image(uid: str, session: SessionDep, _current_user: AdminDep) -> dict[str, str]:
    """Delete an image and its vector index entry."""
    await tool_catalog.delete_image(session, uid)
    return {'message': 'Image deleted successfully', 'uid': uid}


@image_router.post('/{uid}/run')
async def run_in_image(uid: str, request: RunRequest, session: SessionDep, _current_user: MemberDep) -> dict:
    """Run a command inside a Docker image."""
    try:
        result = tool_catalog.execute_catalog_image_command(
            session,
            uid,
            request.command,
            ensure_ascii=False,
        )
    except tool_catalog.ToolCatalogNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, str(error)) from error
    return {'code': result.exit_code, 'result': result.output}


# =============== doc ===============
@doc_router.get('/{uid}')
async def get_doc(session: SessionDep, uid: str) -> ToolHelpDocPublic:
    """Get a tool help document by UID."""
    doc = doc_crud.get(session, uid)
    if doc is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'No corresponding doc found')
    return ToolHelpDocPublic.model_validate(doc)


@doc_router.post('/{uid}/refresh')
async def refresh_doc(session: SessionDep, uid: str, _current_user: AdminDep) -> ToolHelpDocPublic:
    """Refresh a tool help document by re-running its help command."""
    try:
        return tool_catalog.refresh_help_document(session, uid)  # type: ignore[return-value]
    except tool_catalog.ToolCatalogNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, str(error)) from error
    except tool_catalog.ToolCommandExecutionError as error:
        raise HTTPException(HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error


# =============== tool ===============
@router.post('')
async def add_new_tool(tool_prop: DockerToolCreate, session: SessionDep, _current_user: AdminDep) -> SimpleToolPublic:
    """Add a new Docker-based tool and index it in the vector store."""
    try:
        return await tool_catalog.create_tool(session, tool_prop)  # type: ignore[return-value]
    except tool_catalog.ToolCatalogNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, str(error)) from error
    except tool_catalog.ToolCommandExecutionError as error:
        raise HTTPException(HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error


@router.get('')
async def get_tool_list(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[SimpleToolPublic]:
    """Get a paginated list of tools."""
    tools, total = tool_crud.get_page(session, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[SimpleToolPublic.model_validate(tool) for tool in tools],
    )


@router.get('/search')
def get_tools_by_name(
    session: SessionDep,
    name: str,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[SimpleToolPublic]:
    """Search tools by name (fuzzy match)."""
    tools, total = tool_crud.get_page_by_name_fuzzy(session, name, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[SimpleToolPublic.model_validate(tool, from_attributes=True) for tool in tools],
    )


@router.get('/{tool_uid}')
async def get_tool_detail(session: SessionDep, tool_uid: str) -> ToolPublic:
    """Get full details of a tool by its UID."""
    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Tool not found')
    return ToolPublic.model_validate(tool)


@router.get('/{tool_uid}/usage')
async def get_tool_usage(
    tool_uid: str,
    session: SessionDep,
    _current_user: AdminDep,
    workflow_offset: Annotated[int, Query(ge=0)] = 0,
    run_offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> ToolUsagePublic:
    """List saved workflows and historical runs that reference a tool."""
    if tool_crud.get(session, tool_uid) is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Tool not found')

    workflow_total, run_total = tool_crud.get_usage_counts(session, tool_uid)
    return ToolUsagePublic(
        workflow_total=workflow_total,
        run_total=run_total,
        workflow_offset=workflow_offset,
        run_offset=run_offset,
        limit=limit,
        workflows=tool_crud.get_workflow_usage_page(session, tool_uid, limit, workflow_offset),
        runs=tool_crud.get_run_usage_page(session, tool_uid, limit, run_offset),
    )


@router.patch('/{tool_uid}')
async def update_tool(
    tool_uid: str,
    tool_update: DockerToolUpdate,
    session: SessionDep,
    redis: RedisDep,
    _current_user: AdminDep,
) -> SimpleToolPublic:
    """更新工具配置."""
    tool = await tool_catalog.update_tool(session, redis, tool_uid, tool_update)
    if tool is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Tool not found')
    return tool  # type: ignore[return-value]


@router.patch('/{tool_uid}/ai-checked')
async def mark_tool_ai_checked(
    tool_uid: str,
    session: SessionDep,
    redis: RedisDep,
    _current_user: AdminDep,
) -> SimpleToolPublic:
    """Mark a tool as AI Checked while preserving its unrelated tags."""
    try:
        return await tool_catalog.mark_tool_as_ai_checked(session, redis, tool_uid)  # type: ignore[return-value]
    except tool_catalog.ToolCatalogNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, str(error)) from error
    except tool_catalog.ToolCatalogConfigurationError as error:
        raise HTTPException(HTTP_409_CONFLICT, str(error)) from error


@router.delete('/{tool_uid}')
async def delete_tool(tool_uid: str, session: SessionDep, redis: RedisDep, _current_user: AdminDep) -> dict:
    """删除指定工具."""
    try:
        await tool_catalog.delete_tool(session, redis, tool_uid)
    except tool_catalog.ToolInUseError as error:
        raise HTTPException(HTTP_409_CONFLICT, detail=str(error)) from error
    return {'message': 'Tool deleted successfully', 'uid': tool_uid}


@router.get('/{tool_uid}/args')
async def get_tool_args(session: SessionDep, redis: RedisDep, tool_uid: str) -> ToolNodePublic:
    """Get the argument schema for a tool."""
    tool_node = await tool_cache.get_tool_node(session, redis, tool_uid)
    if tool_node is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Tool not found')
    return tool_node


@router.get('/{tool_uid}/task_info')
async def get_tool4task(session: SessionDep, redis: RedisDep, tool_uid: str) -> ToolTaskInfo:
    """Get task-relevant metadata for a tool."""
    execution_config = await tool_cache.get_tool_execution_config(session, redis, tool_uid)
    if execution_config is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Tool not found')
    return execution_config
