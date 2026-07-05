import asyncio
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from starlette.requests import Request
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

import app.crud.resource.bio_db as db_crud
import app.crud.resource.sample as sample_crud
import app.services.resource.bio_db as db_service
from app.core.session import SessionDep, engine
from app.models.resource import BioDb
from app.schemas.common import Paginated
from app.schemas.resource import (
    BioDbCreate,
    BioDbDownloadResponse,
    BioDbDownloadStatus,
    BioDbDownloadStatusResponse,
    BioDbSimple,
    SampleListItem,
)
from app.utils.security import AdminDep, UserDep

db_router = APIRouter()
sample_router = APIRouter()


# ── Bio DB endpoints ───────────────────────────────────────────────────────────


@db_router.post('')
def new_bio_db(
    bio_db_prop: BioDbCreate,
    session: SessionDep,
    _current_user: AdminDep,
) -> BioDb:
    """Create a new bio database entry."""
    if bio_db_prop.path is None and not bio_db_prop.download_command:
        raise HTTPException(HTTP_422_UNPROCESSABLE_CONTENT, 'Either path or download_command must be provided')

    bio_db = BioDb(
        name=bio_db_prop.name,
        description=bio_db_prop.description,
        path=bio_db_prop.path,
        size=None,
        last_update=bio_db_prop.last_update,
        download_command=bio_db_prop.download_command,
        download_image=bio_db_prop.download_image,
    )
    return db_crud.insert(session, bio_db)


@db_router.delete('/{db_id}')
async def delete_db(db_id: int, session: SessionDep, _current_user: AdminDep) -> None:
    """Delete a bio database entry by ID."""
    return db_crud.delete(session, db_id)


@db_router.post('/{db_id}/download')
async def download_db(db_id: int, session: SessionDep, _current_user: AdminDep) -> BioDbDownloadResponse:
    """Submit a background task to download or re-download a bio database."""
    return await db_service.submit_download_task(db_id, session)


@db_router.get('/{db_id}/download/status')
async def get_download_status(
    db_id: int,
    session: SessionDep,
    _current_user: UserDep,
) -> BioDbDownloadStatusResponse:
    """Get the current or latest download status for a bio database."""
    return await db_service.build_download_status(db_id, session)


@db_router.get('/{db_id}/download/events')
async def stream_download_status(
    db_id: int,
    request: Request,
    _current_user: UserDep,
    interval: Annotated[float, Query(ge=0.5, le=10)] = 2.0,
) -> StreamingResponse:
    """Stream bio database download status updates using Server-Sent Events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        terminal_statuses = {
            BioDbDownloadStatus.ready,
            BioDbDownloadStatus.not_downloaded,
        }

        while True:
            if await request.is_disconnected():
                break

            status = await db_service.build_download_status_from_redis(db_id)
            if status is None:
                with Session(engine) as session:
                    status = await db_service.build_download_status(db_id, session)

            yield f'event: download_status\ndata: {status.model_dump_json()}\n\n'

            if status.status in terminal_statuses:
                break

            await asyncio.sleep(interval)

    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


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
