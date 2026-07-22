from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from langchain_core.documents import Document
from pydantic import ValidationError
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT

import app.crud.tools.image as image_crud
import app.crud.tools.tool as tool_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool_tag as tag_crud
from app.core.config import get_settings
from app.core.redis_client import RedisDep
from app.core.session import SessionDep
from app.models.tools import ToolGroup, ToolHelpDoc, ToolImage, ToolTag
from app.schemas.common import Paginated
from app.schemas.tools import (
    DockerToolCreate,
    DockerToolUpdate,
    DocUpdate,
    ImagePublic,
    ImageUpdate,
    RunRequest,
    SimpleToolPublic,
    ToolGroupPublic,
    ToolNodePublic,
    ToolPublic,
    ToolTaskInfo,
)
from app.utils.container import run_command
from app.utils.security import AdminDep, MemberDep
from llm.models import qwen3_embedding
from llm.vectorstores import MilvusVectorStore

router = APIRouter()
tag_router = APIRouter()
group_router = APIRouter()
image_router = APIRouter()
doc_router = APIRouter()


# ===============tag and group===============


@tag_router.get('')
async def get_exist_tags(session: SessionDep) -> Sequence[ToolTag]:
    """Get all existing tool tags."""
    return tag_crud.get_all(session)


@group_router.post('')
async def add_group(session: SessionDep, group: ToolGroup, _current_user: AdminDep) -> ToolGroup:
    """Add a new tool group."""
    group_tag = ToolTag(name=group.name)
    group_tag = tag_crud.insert(session, group_tag)
    group.tag_id = group_tag.id

    return group_crud.insert(session, group)


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
async def create_image(image: ToolImage, session: SessionDep, _current_user: AdminDep) -> ToolImage:
    """Create a new Docker image registry entry."""
    return image_crud.insert(session, image)


@image_router.get('')
async def get_images(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[ToolImage]:
    """Get a paginated list of Docker images."""
    images, total = image_crud.get_page(session, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=list(images),
    )


@image_router.get('/search')
def get_images_by_name(
    session: SessionDep,
    name: str,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[ToolImage]:
    """Search Docker images by name (fuzzy match)."""
    images, total = image_crud.get_page_by_name_fuzzy(session, name, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=list(images),
    )


@image_router.get('/{uid}')
async def get_image(session: SessionDep, uid: str) -> ImagePublic:
    """Get a Docker image entry by UID."""
    image = image_crud.get(session, uid)
    if not image:
        raise HTTPException(HTTP_404_NOT_FOUND, 'No corresponding image found')
    return image  # type: ignore


@image_router.patch('/{uid}')
async def update_image(uid: str, image_update: ImageUpdate, session: SessionDep, _current_user: AdminDep) -> ToolImage:
    """更新镜像信息."""
    updated_image = image_crud.update(session, uid, image_update)
    if updated_image is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Image not found')
    return updated_image


@image_router.delete('/{uid}')
async def delete_image(uid: str, session: SessionDep, _current_user: AdminDep) -> dict[str, str]:
    """Delete an image and its vector index entry."""
    image_crud.delete(session, uid)
    vectorstore = MilvusVectorStore(
        collection_name='ImageCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
    )
    await vectorstore.adelete(
        filter_expression='image_uid == {uid}',
        filter_params={'uid': uid},
    )
    return {'message': 'Image deleted successfully', 'uid': uid}


@image_router.post('/{uid}/run')
async def run_in_image(uid: str, request: RunRequest, session: SessionDep, _current_user: MemberDep) -> dict:
    """Run a command inside a Docker image."""
    image = image_crud.get(session, uid)
    if not image:
        raise HTTPException(HTTP_404_NOT_FOUND, 'No corresponding image found')

    code, result = run_command(image.image, [], request.command, ensure_ascii=False)
    return {'code': code, 'result': result}


# =============== doc ===============
@doc_router.get('/{uid}')
async def get_doc(session: SessionDep, uid: str) -> ToolHelpDoc:
    """Get a tool help document by UID."""
    return doc_crud.get(session, uid)


@doc_router.post('/{uid}/refresh')
async def refresh_doc(session: SessionDep, uid: str) -> ToolHelpDoc:
    """Refresh a tool help document by re-running its help command."""
    doc = doc_crud.get(session, uid)
    if not doc:
        raise HTTPException(HTTP_404_NOT_FOUND, 'No corresponding doc found')

    if doc.image is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Image associated with doc not found')
    _, result = run_command(doc.image.image, [], doc.help_command)
    update_data = DocUpdate(content=result)
    return doc_crud.update(session, uid, update_data)


# =============== tool ===============
@router.post('')
async def add_new_tool(tool_prop: DockerToolCreate, session: SessionDep, _current_user: AdminDep) -> SimpleToolPublic:
    """Add a new Docker-based tool and index it in the vector store."""
    # 解析tag
    tags = tag_crud.get_all_by_id(session, tool_prop.tag_ids)

    # 解析帮助命令
    tool_image = image_crud.get(session, str(tool_prop.image_uid))
    if tool_image is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Image not found')
    tool_doc = doc_crud.get_by_command(session, tool_prop.help_command)
    if tool_doc is None:
        _, result = run_command(tool_image.image, [], tool_prop.help_command)
        tool_doc = ToolHelpDoc(help_command=tool_prop.help_command, content=result, image_uid=tool_prop.image_uid)
        tool_doc = doc_crud.insert(session, tool_doc)

    tool_data = tool_prop.to_tool_data()
    tool_data.tags = list(tags)
    tool_data.help_doc_uid = tool_doc.uid

    new_tool = tool_crud.insert(session, tool_data)

    # 添加向量索引
    embedding_model = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        collection_name='ToolCollection',
        embedding_function=embedding_model,
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
    )

    docs = [
        Document(
            page_content=new_tool.description,
            metadata={
                'image_uid': str(new_tool.image_uid),
                'tool_uid': str(new_tool.uid),
                'name': new_tool.name,
            },
        ),
    ]
    async with vectorstore:
        await vectorstore.aadd_documents(docs)

    # pyrefly: ignore [bad-return]
    return new_tool


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
    # pyrefly: ignore [bad-return]
    return tool


@router.patch('/{tool_uid}')
async def update_tool(
    tool_uid: str,
    tool_update: DockerToolUpdate,
    session: SessionDep,
    redis: RedisDep,
    _current_user: AdminDep,
) -> SimpleToolPublic:
    """更新工具配置."""
    tool = tool_crud.update_partial(session, tool_uid, tool_update)
    await redis.delete(f'tool-args:{tool_uid}', f'tool-task-info:{tool_uid}')
    # pyrefly: ignore [bad-return]
    return tool


@router.patch('/{tool_uid}/ai-checked')
async def mark_tool_ai_checked(
    tool_uid: str,
    session: SessionDep,
    redis: RedisDep,
    _current_user: AdminDep,
) -> SimpleToolPublic:
    """Mark a tool as AI Checked while preserving its unrelated tags."""
    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Tool not found')

    checked_tag = tag_crud.get_by_name(session, 'AI Checked')
    if checked_tag is None:
        raise HTTPException(status_code=HTTP_409_CONFLICT, detail='AI Checked tag is not configured')

    tool.tags = [tag for tag in tool.tags if tag.name not in {'AI Checked', 'AI Unchecked'}]
    tool.tags.append(checked_tag)
    tool = tool_crud.update(session, tool)

    await redis.delete(f'tool-args:{tool_uid}', f'tool-task-info:{tool_uid}')
    # pyrefly: ignore [bad-return]
    return tool


@router.delete('/{tool_uid}')
async def delete_tool(tool_uid: str, session: SessionDep, redis: RedisDep, _current_user: AdminDep) -> dict:
    """删除指定工具."""
    tool_crud.delete(session, tool_uid)
    vectorstore = MilvusVectorStore(
        collection_name='ToolCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
    )
    await vectorstore.adelete(
        filter_expression='tool_uid == {uid}',
        filter_params={'uid': tool_uid},
    )
    await redis.delete(f'tool-args:{tool_uid}', f'tool-task-info:{tool_uid}')
    return {'message': 'Tool deleted successfully', 'uid': tool_uid}


@router.get('/{tool_uid}/args')
async def get_tool_args(session: SessionDep, redis: RedisDep, tool_uid: str) -> ToolNodePublic:
    """Get the argument schema for a tool."""
    cache_key = f'tool-args:{tool_uid}'
    cached = await redis.get(cache_key)
    if cached:
        try:
            return ToolNodePublic.model_validate_json(cached)
        except ValidationError:
            await redis.delete(cache_key)

    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Tool not found')
    await redis.set(cache_key, ToolNodePublic.model_validate(tool).model_dump_json(), ex=86400)

    # pyrefly: ignore [bad-return]
    return tool


@router.get('/{tool_uid}/task_info')
async def get_tool4task(session: SessionDep, redis: RedisDep, tool_uid: str) -> ToolTaskInfo:
    """Get task-relevant metadata for a tool."""
    cache_key = f'tool-task-info:{tool_uid}'
    cached = await redis.get(cache_key)
    if cached:
        try:
            return ToolTaskInfo.model_validate_json(cached)
        except ValidationError:
            await redis.delete(cache_key)

    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail='Tool not found')
    await redis.set(cache_key, ToolTaskInfo.model_validate(tool).model_dump_json(), ex=86400)

    # pyrefly: ignore [bad-return]
    return tool
