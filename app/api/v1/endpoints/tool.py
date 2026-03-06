from collections.abc import Sequence
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query
from langchain_core.documents import Document
from langchain_milvus import Milvus
from starlette.status import HTTP_404_NOT_FOUND

import app.crud.tools.image as image_crud
import app.crud.tools.tool as tool_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool_tag as tag_crud
from app.core.config import get_settings
from app.core.session import SessionDep
from app.models.tools import ToolGroup, ToolHelpDoc, ToolImage, ToolTag
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
    parent_id: Optional[int] = None,
    limit: Annotated[int, Query(le=20)] = 20,
    offset: int = 0,
) -> list[SimpleToolPublic]:
    """Get tools belonging to a specific group."""
    return tool_crud.get_all_by_group(session, parent_id, limit, offset)


# =============== image ===============
@image_router.post('')
async def create_image(image: ToolImage, session: SessionDep, _current_user: AdminDep) -> ToolImage:
    """Create a new Docker image registry entry."""
    return image_crud.insert(session, image)


@image_router.get('')
async def get_images(session: SessionDep, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20) -> list[ToolImage]:
    """Get a paginated list of Docker images."""
    return image_crud.get_all(session, limit, offset)


@image_router.get('/count')
async def get_image_count(session: SessionDep) -> int:
    """Get the total count of Docker images."""
    return image_crud.get_counts(session)


@image_router.get('/search')
def get_images_by_name(
    session: SessionDep, name: str, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20
) -> list[ToolImage]:
    """Search Docker images by name (fuzzy match)."""
    return image_crud.get_by_name_fuzzy(session, name, limit, offset)


@image_router.get('/{uid}')
async def get_image(session: SessionDep, uid: str) -> ImagePublic:
    """Get a Docker image entry by UID."""
    return image_crud.get(session, uid)


@image_router.patch('/{uid}')
async def update_image(uid: str, image_update: ImageUpdate, session: SessionDep, _current_user: AdminDep) -> ToolImage:
    """更新镜像信息."""
    updated_image = image_crud.update(session, uid, image_update)
    if updated_image is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Image not found')
    return updated_image


@image_router.post('/{uid}/run')
async def run_in_image(uid: str, request: RunRequest, session: SessionDep, _current_user: MemberDep) -> dict:
    """Run a command inside a Docker image."""
    image = image_crud.get(session, uid)
    if not image:
        raise HTTPException(HTTP_404_NOT_FOUND, 'No corresponding image found')

    code, result = run_command(image.image, [], request.command)
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

    code, result = run_command(doc.image.image, [], doc.help_command)
    update_data = DocUpdate(content=result)
    return doc_crud.update(session, uid, update_data)


# =============== tool ===============
@router.post('')
async def add_new_tool(tool_prop: DockerToolCreate, session: SessionDep, _current_user: AdminDep) -> SimpleToolPublic:
    """Add a new Docker-based tool and index it in the vector store."""
    # 解析tag
    tags = tag_crud.get_all_by_id(session, tool_prop.tag_ids)

    # 解析帮助命令
    tool_image = image_crud.get(session, tool_prop.image_uid)
    tool_doc = doc_crud.get_by_command(session, tool_prop.help_command)
    if tool_doc is None:
        code, result = run_command(tool_image.image, [], tool_prop.help_command)
        tool_doc = ToolHelpDoc(help_command=tool_prop.help_command, content=result, image_uid=tool_prop.image_uid)
        tool_doc = doc_crud.insert(session, tool_doc)

    tool_data = tool_prop.to_tool_data()
    tool_data.tags = tags
    tool_data.help_doc_uid = tool_doc.uid

    new_tool = tool_crud.insert(session, tool_data)

    # 添加向量索引
    embedding_model = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='ToolCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding_model,
        index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
        consistency_level='Strong',
        drop_old=False,
        auto_id=True,
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
    await vectorstore.aadd_documents(docs)

    return new_tool


@router.get('/count')
async def get_tool_count(session: SessionDep) -> int:
    """Get the total count of tools."""
    return tool_crud.get_counts(session)


@router.get('')
async def get_tool_list(
    session: SessionDep, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20
) -> list[SimpleToolPublic]:
    """Get a paginated list of tools."""
    return tool_crud.get_all(session, limit, offset)


@router.get('/search')
def get_tools_by_name(
    session: SessionDep, name: str, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20
) -> list[SimpleToolPublic]:
    """Search tools by name (fuzzy match)."""
    return tool_crud.get_by_name_fuzzy(session, name, limit, offset)


@router.get('/{tool_uid}')
async def get_tool_detail(session: SessionDep, tool_uid: str) -> ToolPublic:
    """Get full details of a tool by its UID."""
    return tool_crud.get(session, tool_uid)


@router.patch('/{tool_uid}')
async def update_tool(
    tool_uid: str, tool_update: DockerToolUpdate, session: SessionDep, _current_user: AdminDep
) -> SimpleToolPublic:
    """更新工具配置."""
    tool = tool_crud.update_partial(session, tool_uid, tool_update)
    if tool is None:
        raise HTTPException(status_code=404, detail='Tool not found')
    return tool


@router.delete('/{tool_uid}')
async def delete_tool(tool_uid: str, session: SessionDep, _current_user: AdminDep) -> dict:
    """删除指定工具."""
    tool = tool_crud.get(session, tool_uid)
    if tool is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Tool not found')

    tool_crud.delete(session, tool_uid)
    return {'message': 'Tool deleted successfully', 'uid': tool_uid}


@router.get('/{tool_uid}/args')
async def get_tool_args(session: SessionDep, tool_uid: str) -> ToolNodePublic:
    """Get the argument schema for a tool."""
    return tool_crud.get(session, tool_uid)


@router.get('/{tool_uid}/task_info')
async def get_tool4task(session: SessionDep, tool_uid: str) -> ToolTaskInfo:
    """Get task-relevant metadata for a tool."""
    return tool_crud.get(session, tool_uid)
