from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from starlette.status import HTTP_404_NOT_FOUND

import app.crud.resource.bio_db as db_crud
from app.core.session import SessionDep
from app.models.resource import BioDb, ReferenceGenomes
from app.schemas.resource import BioDbCreate, BioDbSimple
from app.utils.system import get_folder_size_du
from app.utils.security import AdminDep, UserDep

db_router = APIRouter()
reference_router = APIRouter()


@db_router.post('', response_model=BioDb)
async def new_bio_db(bio_db_prop: BioDbCreate, session: SessionDep, current_user: AdminDep) -> BioDb:
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
async def delete_db(db_id: int, session: SessionDep, current_user: AdminDep):
    return db_crud.delete(session, db_id)


@db_router.get('', response_model=list[BioDbSimple])
async def get_db_list(
    session: SessionDep, current_user: UserDep, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20
):
    return db_crud.get_all(session, limit, offset)


@db_router.get('/count')
async def get_db_count(session: SessionDep, current_user: UserDep) -> int:
    return db_crud.get_counts(session)


@db_router.get('/search')
async def get_dbs_by_name(
    session: SessionDep, current_user: UserDep, name: str, offset: int = 0, limit: Annotated[int, Query(le=20)] = 20
) -> list[BioDb]:
    return db_crud.get_by_name_fuzzy(session, name, limit, offset)


@db_router.get('/{db_id}', response_model=BioDb)
async def get_db(db_id: int, session: SessionDep, current_user: UserDep) -> BioDb:
    return db_crud.get(session, db_id)


@reference_router.post('')
async def new_reference(bio_db_prop: BioDbCreate, session: SessionDep, current_user: AdminDep):
    pass


@reference_router.get('/{ref_id}', response_model=ReferenceGenomes)
async def get_reference_by_id(ref_id: int, session: SessionDep, current_user: AdminDep):
    pass
