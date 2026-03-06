from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from starlette.status import HTTP_404_NOT_FOUND

import app.crud.resource.bio_db as db_crud
import app.crud.resource.sample as sample_crud
from app.core.session import SessionDep
from app.models.resource import BioDb, ReferenceGenomes
from app.schemas.resource import (
    BioDbCreate,
    BioDbSimple,
    SampleListItem,
)
from app.utils.security import AdminDep, UserDep, get_current_active_user
from app.utils.system import get_folder_size_du

db_router = APIRouter()
reference_router = APIRouter()
sample_router = APIRouter()


@db_router.post('')
async def new_bio_db(
    bio_db_prop: BioDbCreate,
    session: SessionDep,
    _current_user: AdminDep,
) -> BioDb:
    """Create a new bio database entry."""
    if not Path(bio_db_prop.path).exists():
        raise HTTPException(HTTP_404_NOT_FOUND, 'path did not exist')

    db_size = get_folder_size_du(bio_db_prop.path)
    bio_db = BioDb(
        name=bio_db_prop.name,
        description=bio_db_prop.description,
        path=bio_db_prop.path,
        size=db_size,
        last_update=bio_db_prop.last_update,
    )
    return db_crud.insert(session, bio_db)


@db_router.delete('/{db_id}')
async def delete_db(db_id: int, session: SessionDep, _current_user: AdminDep) -> BioDb:
    """Delete a bio database entry by ID."""
    return db_crud.delete(session, db_id)


@db_router.get('')
async def get_db_list(
    session: SessionDep,
    _current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[BioDbSimple]:
    """Get a paginated list of all bio databases."""
    return db_crud.get_all(session, limit, offset)


@db_router.get('/count')
async def get_db_count(session: SessionDep, _current_user: UserDep) -> int:
    """Get the total count of bio databases."""
    return db_crud.get_counts(session)


@db_router.get('/search')
async def get_dbs_by_name(
    session: SessionDep,
    _current_user: UserDep,
    name: str,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[BioDb]:
    """Search bio databases by name (fuzzy match)."""
    return db_crud.get_by_name_fuzzy(session, name, limit, offset)


@db_router.get('/{db_id}')
async def get_db(db_id: int, session: SessionDep, _current_user: UserDep) -> BioDb:
    """Get a bio database entry by ID."""
    return db_crud.get(session, db_id)


@reference_router.post('')
async def new_reference(bio_db_prop: BioDbCreate, session: SessionDep, _current_user: AdminDep) -> None:
    """Create a new reference genome entry."""


@reference_router.get('/{ref_id}', response_model=ReferenceGenomes)
async def get_reference_by_id(ref_id: int, session: SessionDep, _current_user: AdminDep) -> None:
    """Get a reference genome by ID."""


# Sample 全局统计端点(放在 /api/v1/samples 下)
@sample_router.get('/count')
async def get_total_sample_count(
    session: SessionDep,
    current_user: UserDep,
) -> int:
    """获取用户的样本总数."""
    return sample_crud.get_count_by_user(session, current_user.id)


@sample_router.get('')
async def get_all_user_samples(
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> list[SampleListItem]:
    """获取用户的所有样本(跨项目)."""
    return sample_crud.get_all_by_user(session, current_user.id, limit, offset)
