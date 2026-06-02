from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from starlette.status import HTTP_404_NOT_FOUND

import app.crud.resource.bio_db as db_crud
import app.crud.resource.sample as sample_crud
from app.core.session import SessionDep
from app.models.resource import BioDb
from app.schemas.common import Paginated
from app.schemas.resource import (
    BioDbCreate,
    BioDbSimple,
    SampleListItem,
)
from app.utils.security import AdminDep, UserDep
from app.utils.system import get_folder_size_du

db_router = APIRouter()
sample_router = APIRouter()


# ── Bio DB endpoints ───────────────────────────────────────────────────────────


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
async def delete_db(db_id: int, session: SessionDep, _current_user: AdminDep) -> None:
    """Delete a bio database entry by ID."""
    return db_crud.delete(session, db_id)


@db_router.get('')
async def get_db_list(
    session: SessionDep,
    _current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[BioDbSimple]:
    """Get a paginated list of all bio databases."""
    dbs, total = db_crud.get_page(session, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=[BioDbSimple.model_validate(db, from_attributes=True) for db in dbs],
    )


@db_router.get('/search')
async def get_dbs_by_name(
    session: SessionDep,
    _current_user: UserDep,
    name: str,
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> Paginated[BioDb]:
    """Search bio databases by name (fuzzy match)."""
    dbs, total = db_crud.get_page_by_name_fuzzy(session, name, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=dbs,
    )


@db_router.get('/{db_id}')
async def get_db(db_id: int, session: SessionDep, _current_user: UserDep) -> BioDb:
    """Get a bio database entry by ID."""
    return db_crud.get(session, db_id)


# ── Sample endpoints ───────────────────────────────────────────────────────────


@sample_router.get('')
async def get_all_user_samples(
    session: SessionDep,
    current_user: UserDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
) -> Paginated[SampleListItem]:
    """Get all samples owned by the current user (across all projects)."""
    samples, total = sample_crud.get_page_by_user(session, current_user.id, limit, offset)
    return Paginated(
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
        data=samples,
    )
