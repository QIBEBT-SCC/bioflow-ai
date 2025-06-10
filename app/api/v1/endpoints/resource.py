from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.status import HTTP_404_NOT_FOUND

import app.crud.resource.bio_db as db_crud
from app.core.session import SessionDep
from app.models.resource import BioDb
from app.models.user import User
from app.schemas.resource import BioDbCreate, BioDbSimple
from app.utils.file import get_folder_size_du
from app.utils.security import get_current_active_user

db_router = APIRouter()


@db_router.post('', response_model=BioDb)
async def new_bio_db(
    bio_db_prop: BioDbCreate, session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]
) -> BioDb:
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
async def delete_db(db_id: int, session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]):
    return db_crud.delete(session, db_id)


@db_router.get('', response_model=list[BioDbSimple])
async def get_db_list(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    return db_crud.get_all(session, limit, offset)


@db_router.get('/count')
async def get_db_count(session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]) -> int:
    return db_crud.get_counts(session)


@db_router.get('/{db_id}', response_model=BioDb)
async def get_db(
    db_id: int, session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]
) -> int:
    return db_crud.get(session, db_id)
