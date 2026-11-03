import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND

from bioflow_ai.core.auth import AdminDep
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.schemas.common import Paginated
from bioflow_ai.schemas.node_memory import ToolMemoryCreate, ToolMemoryListItem, ToolMemoryPublic, ToolMemoryUpdate
from bioflow_ai.services.tools import memory as memory_service

router = APIRouter()


@router.get('')
def list_memories(
    session: SessionDep,
    tool_uid: uuid.UUID | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Paginated[ToolMemoryListItem]:
    """List shared memory summaries, optionally filtered by tool."""
    memories, total = memory_service.get_tool_memory_page(session, tool_uid=tool_uid, offset=offset, limit=limit)
    return Paginated(total=total, offset=offset, limit=limit, has_more=offset + len(memories) < total, data=memories)


@router.post('', status_code=HTTP_201_CREATED)
def create_memory(data: ToolMemoryCreate, session: SessionDep, current_user: AdminDep) -> ToolMemoryPublic:
    """Create a shared memory for an existing tool."""
    memory = memory_service.create_tool_memory(session, data, user_id=current_user.id)
    if memory is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Tool not found')
    return memory


@router.get('/{memory_uid}')
def get_memory(memory_uid: uuid.UUID, session: SessionDep) -> ToolMemoryPublic:
    """Get a memory, including its body."""
    memory = memory_service.read_tool_memory(session, memory_uid)
    if memory is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Tool memory not found')
    return memory


@router.patch('/{memory_uid}')
def update_memory(
    memory_uid: uuid.UUID, data: ToolMemoryUpdate, session: SessionDep, _current_user: AdminDep
) -> ToolMemoryPublic:
    """Edit a memory's description or body."""
    memory = memory_service.update_tool_memory(session, memory_uid, data)
    if memory is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Tool memory not found')
    return memory


@router.delete('/{memory_uid}', status_code=HTTP_204_NO_CONTENT)
def delete_memory(memory_uid: uuid.UUID, session: SessionDep, _current_user: AdminDep) -> Response:
    """Delete a shared memory."""
    if not memory_service.delete_tool_memory(session, memory_uid):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Tool memory not found')
    return Response(status_code=HTTP_204_NO_CONTENT)
