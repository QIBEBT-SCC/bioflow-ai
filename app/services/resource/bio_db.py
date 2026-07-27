"""Bio database download service."""

from typing import TYPE_CHECKING, cast

from celery import states
from loguru import logger
from sqlalchemy.orm import Session

import app.crud.resource.bio_db as db_crud
from app.core.redis_client import monitor_redis_client
from app.models.resource import BioDb
from app.schemas.resource import BioDbDownloadResponse, BioDbDownloadStatus, BioDbDownloadStatusResponse
from app.services._transaction import committing
from app.utils.redis import decode_redis_value, get_bio_db_download_task_key
from app.utils.system import random_uid
from script.worker import celery_app

if TYPE_CHECKING:
    from collections.abc import Awaitable

BIO_DB_DOWNLOAD_TASK_TTL_SECONDS = 60 * 60 * 24
_DELETE_DOWNLOAD_TASK_LOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


class BioDbServiceError(ValueError):
    """Base error for biological-database application operations."""


class BioDbNotFoundError(BioDbServiceError):
    """Raised when a biological database entry does not exist."""


class BioDbConflictError(BioDbServiceError):
    """Raised when a biological database name already exists."""


class BioDbDownloadNotConfiguredError(BioDbServiceError):
    """Raised when a download is requested without a configured command."""


class BioDbDownloadAlreadyRunningError(BioDbServiceError):
    """Raised when a download task already owns the database lock."""


def get_bio_db(session: Session, db_id: int) -> BioDb:
    """Return a biological database entry or raise a domain error."""
    bio_db = db_crud.get(session, db_id)
    if bio_db is None:
        raise BioDbNotFoundError
    return bio_db


def create_bio_db(session: Session, bio_db: BioDb) -> BioDb:
    """Create and commit a biological database entry."""
    if db_crud.get_by_name(session, bio_db.name) is not None:
        raise BioDbConflictError
    with committing(session):
        db_crud.insert(session, bio_db)
    session.refresh(bio_db)
    return bio_db


def delete_bio_db(session: Session, db_id: int) -> None:
    """Delete and commit a biological database entry."""
    with committing(session):
        if not db_crud.delete(session, db_id):
            raise BioDbNotFoundError


def clear_download_result(session: Session, db_id: int) -> BioDb:
    """Clear and commit the stored result of an earlier download."""
    with committing(session):
        bio_db = db_crud.clear_download_result(session, db_id)
        if bio_db is None:
            raise BioDbNotFoundError
    session.refresh(bio_db)
    return bio_db


def update_download_result(session: Session, db_id: int, *, path: str, last_update: str) -> BioDb:
    """Commit the path and timestamp produced by a successful download."""
    with committing(session):
        bio_db = db_crud.update_download_result(session, db_id, path=path, last_update=last_update)
        if bio_db is None:
            raise BioDbNotFoundError
    session.refresh(bio_db)
    return bio_db


async def build_download_status(db_id: int, session: Session) -> BioDbDownloadStatusResponse:
    """Build the current download status from active Redis lock and stored path."""
    redis_status = await build_download_status_from_redis(db_id)
    if redis_status is not None:
        return redis_status

    bio_db = get_bio_db(session, db_id)
    if bio_db.path:
        return BioDbDownloadStatusResponse(
            status=BioDbDownloadStatus.ready,
            message='database is ready',
        )

    return BioDbDownloadStatusResponse(
        status=BioDbDownloadStatus.not_downloaded,
        message='database has not been downloaded',
    )


async def build_download_status_from_redis(db_id: int) -> BioDbDownloadStatusResponse | None:
    """Return downloading status while the locked Celery task is non-terminal."""
    task_key = get_bio_db_download_task_key(db_id)
    task_id = decode_redis_value(await monitor_redis_client.get(task_key))
    if task_id:
        try:
            state = celery_app.AsyncResult(task_id).state
        except Exception as error:
            logger.warning(f'Failed to query Celery state for BioDb {db_id} download task {task_id}: {error}')
        else:
            if state in states.READY_STATES:
                try:
                    await cast(
                        'Awaitable[str]',
                        monitor_redis_client.eval(
                            _DELETE_DOWNLOAD_TASK_LOCK_SCRIPT,
                            1,
                            task_key,
                            task_id,
                        ),
                    )
                except Exception as error:
                    logger.warning(f'Failed to clear completed BioDb {db_id} download task lock {task_id}: {error}')
                return None

        return BioDbDownloadStatusResponse(
            status=BioDbDownloadStatus.downloading,
            message='database download task is running',
        )

    return None


async def submit_download_task(db_id: int, session: Session) -> BioDbDownloadResponse:
    """Submit a locked background download task for a bio database."""
    # Delayed to avoid the load_bio_db task -> service -> task import cycle.
    from app.tasks.load_tasks.load_bio_db import download_bio_db_task

    bio_db = get_bio_db(session, db_id)
    if not bio_db.download_command:
        raise BioDbDownloadNotConfiguredError

    task_key = get_bio_db_download_task_key(db_id)
    existing_task_id = await monitor_redis_client.get(task_key)
    if existing_task_id:
        raise BioDbDownloadAlreadyRunningError

    task_id = str(random_uid())
    acquired = await monitor_redis_client.set(task_key, task_id, ex=BIO_DB_DOWNLOAD_TASK_TTL_SECONDS, nx=True)
    if not acquired:
        raise BioDbDownloadAlreadyRunningError

    try:
        download_bio_db_task.apply_async(kwargs={'db_id': db_id}, task_id=task_id)
    except Exception:
        await monitor_redis_client.delete(task_key)
        raise

    return BioDbDownloadResponse(
        task_id=task_id,
        message=f'Download task submitted for bio database "{bio_db.name}".',
    )
