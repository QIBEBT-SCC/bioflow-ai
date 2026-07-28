import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from starlette.status import HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_CONTENT

import app.crud.code as code_crud
from app.core.session import SessionDep
from app.schemas.code import (
    CodeCreate,
    CodeMetadataPublic,
    CodeMetadataRequest,
    CodeNodeType,
    CodePublic,
    CodeUpdate,
    SimpleCodePublic,
)
from app.schemas.common import Paginated
from app.services import code_catalog
from app.utils.security import AdminDep, UserDep
from llm.prebuild.code_node import generate_code_metadata

router = APIRouter()


@router.get('')
async def get_codes(
    session: SessionDep,
    _current_user: UserDep,
    query: str | None = None,
    node_type: CodeNodeType | None = None,
    offset: int = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Paginated[SimpleCodePublic]:
    """Return a filtered page of reusable CODE definitions."""
    codes, total = code_crud.get_page(
        session,
        limit,
        offset,
        query=query,
        node_type=node_type.value if node_type is not None else None,
    )
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[SimpleCodePublic.model_validate(code) for code in codes],
    )


@router.post('')
async def create_code(
    code_data: CodeCreate,
    session: SessionDep,
    _current_user: AdminDep,
) -> CodePublic:
    """Create and index a reusable CODE definition."""
    code = await code_catalog.create_code(session, code_data)
    return CodePublic.model_validate(code)


@router.post('/generate-metadata')
async def generate_metadata(
    request: CodeMetadataRequest,
    current_user: AdminDep,
) -> CodeMetadataPublic:
    """Generate a catalog name and description without persisting the code."""
    return await generate_code_metadata(request, current_user.id)


@router.get('/{code_uid}')
async def get_code(
    code_uid: uuid.UUID,
    session: SessionDep,
    _current_user: UserDep,
) -> CodePublic:
    """Return a complete reusable CODE definition."""
    code = code_crud.get(session, code_uid)
    if code is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'CODE definition not found')
    return CodePublic.model_validate(code)


@router.patch('/{code_uid}')
async def update_code(
    code_uid: uuid.UUID,
    code_data: CodeUpdate,
    session: SessionDep,
    _current_user: AdminDep,
) -> CodePublic:
    """Update an existing CODE definition without changing its node type."""
    try:
        code = await code_catalog.update_code(session, str(code_uid), code_data)
    except code_catalog.CodeCatalogValidationError as error:
        raise HTTPException(HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    if code is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'CODE definition not found')
    return CodePublic.model_validate(code)


@router.delete('/{code_uid}')
async def delete_code(
    code_uid: uuid.UUID,
    session: SessionDep,
    _current_user: AdminDep,
) -> dict[str, str]:
    """Delete a reusable CODE definition and its vector projection."""
    if not await code_catalog.delete_code(session, str(code_uid)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'CODE definition not found')
    return {'message': 'CODE definition deleted successfully', 'uid': str(code_uid)}
