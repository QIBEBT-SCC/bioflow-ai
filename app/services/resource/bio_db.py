"""Bio database download service."""

from typing import TYPE_CHECKING, cast

from celery import states
from fastapi import HTTPException
from loguru import logger
from sqlmodel import Session
from starlette.status import HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_CONTENT

import app.crud.resource.bio_db as db_crud
from app.core.redis_client import monitor_redis_client
from app.schemas.resource import BioDbDownloadResponse, BioDbDownloadStatus, BioDbDownloadStatusResponse
from app.tasks.load_tasks.load_bio_db import download_bio_db_task
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


async def build_download_status(db_id: int, session: Session) -> BioDbDownloadStatusResponse:
    """Build the current download status from active Redis lock and stored path."""
    redis_status = await build_download_status_from_redis(db_id)
    if redis_status is not None:
        return redis_status

    bio_db = db_crud.get(session, db_id)
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
    bio_db = db_crud.get(session, db_id)
    if not bio_db.download_command:
        raise HTTPException(HTTP_422_UNPROCESSABLE_CONTENT, 'download_command is not configured')

    task_key = get_bio_db_download_task_key(db_id)
    existing_task_id = await monitor_redis_client.get(task_key)
    if existing_task_id:
        raise HTTPException(HTTP_409_CONFLICT, 'database download task is already running')

    task_id = str(random_uid())
    acquired = await monitor_redis_client.set(task_key, task_id, ex=BIO_DB_DOWNLOAD_TASK_TTL_SECONDS, nx=True)
    if not acquired:
        raise HTTPException(HTTP_409_CONFLICT, 'database download task is already running')

    try:
        download_bio_db_task.apply_async(kwargs={'db_id': db_id}, task_id=task_id)
    except Exception:
        await monitor_redis_client.delete(task_key)
        raise

    return BioDbDownloadResponse(
        task_id=task_id,
        message=f'Download task submitted for bio database "{bio_db.name}".',
    )
