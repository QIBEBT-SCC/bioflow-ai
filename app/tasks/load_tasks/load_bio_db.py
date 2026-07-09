import shutil
from pathlib import Path
from typing import Any, cast

from celery import Task
from loguru import logger
from sqlmodel import Session

import app.crud.resource.bio_db as db_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.resource import BioDb
from app.models.tools import ImageConfig
from app.models.workflow import ToolOutput
from app.schemas.task_instance import TaskUpdate
from app.tasks.base.base_task import BaseBioTask, update_task
from app.utils.container import ContainerMount, run_command
from app.utils.input_format import to_safe_path_segment
from app.utils.redis import decode_redis_value, get_bio_db_download_task_key
from app.utils.system import utc_now
from script.worker import celery_app, monitor_redis_client

BIO_DB_CONTAINER_PATH = '/data/db'
DEFAULT_DOWNLOAD_IMAGE = ImageConfig.from_str('docker.io/aye1032/biokit-download:1.0.0')
DOWNLOAD_TASK_TTL_SECONDS = 60 * 60 * 24


def _get_active_download_task_id(db_id: int) -> str | None:
    raw_task_id = cast('bytes | str | None', monitor_redis_client.get(get_bio_db_download_task_key(db_id)))
    return decode_redis_value(raw_task_id)


def _acquire_download_slot(db_id: int, task_id: str) -> None:
    key = get_bio_db_download_task_key(db_id)
    existing_task_id = _get_active_download_task_id(db_id)
    if existing_task_id == task_id:
        monitor_redis_client.expire(key, DOWNLOAD_TASK_TTL_SECONDS)
        return

    acquired = monitor_redis_client.set(key, task_id, ex=DOWNLOAD_TASK_TTL_SECONDS, nx=True)
    if not acquired:
        msg = f'BioDb {db_id} is already downloading in task {existing_task_id}.'
        raise RuntimeError(msg)


def _release_download_slot(db_id: int, task_id: str) -> None:
    key = get_bio_db_download_task_key(db_id)
    existing_task_id = _get_active_download_task_id(db_id)
    if existing_task_id == task_id:
        monitor_redis_client.delete(key)


def _run_bio_db_download(
    session: Session,
    bio_db: BioDb,
    task_id: str,
    *,
    task_instance_uid: str | None = None,
) -> Path:
    """Download a BioDb by executing its configured command in a container."""
    if bio_db.id is None:
        msg = 'BioDb must be persisted before it can be downloaded.'
        raise ValueError(msg)
    if not bio_db.download_command:
        msg = f'BioDb {bio_db.id} has no download_command.'
        raise ValueError(msg)

    _acquire_download_slot(bio_db.id, task_id)
    try:
        # 初始化下载路径
        if bio_db.path:
            target_dir = Path(bio_db.path).expanduser().absolute()
        else:
            base_dir = Path(get_settings().persistent.storage.BIO_DB_PATH).expanduser().absolute()
            safe_name = to_safe_path_segment(bio_db.name)
            target_dir = base_dir / f'{bio_db.id}_{safe_name}'

        if target_dir.exists() and not target_dir.is_dir():
            msg = f'BioDb download target must be a directory: {target_dir}'
            raise ValueError(msg)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 清理残余数据
        db_crud.clear_download_result(session, bio_db.id)
        for child in target_dir.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

        image_cfg = ImageConfig.from_str(bio_db.download_image) if bio_db.download_image else DEFAULT_DOWNLOAD_IMAGE
        mounts = [
            ContainerMount(source=str(target_dir), target=BIO_DB_CONTAINER_PATH, read_only=False),
            ContainerMount(source='/tmp', target='/tmp', read_only=False),  # noqa: S108
        ]
        command = bio_db.download_command
        if task_instance_uid is not None:
            update_task(
                task_instance_uid,
                TaskUpdate(
                    commands=f'docker run --rm -v {target_dir}:{BIO_DB_CONTAINER_PATH}:rw '
                    f'{image_cfg.get_image()} /bin/sh -c "{command}"'
                ),
            )

        status, output = run_command(
            image_cfg,
            mounts,
            command,
            workdir=BIO_DB_CONTAINER_PATH,
            ensure_ascii=False,
        )
        if status != 0:
            logger.error(output)
            msg = f'BioDb download command failed with exit code {status}: {output}'
            raise RuntimeError(msg)

        db_crud.update_download_result(
            session,
            bio_db.id,
            path=str(target_dir),
            last_update=utc_now().isoformat(),
        )
    finally:
        _release_download_slot(bio_db.id, task_id)

    return target_dir


@celery_app.task(bind=True, pydantic=True)
def download_bio_db_task(
    self: Task,
    *,
    db_id: int,
) -> dict[str, Any]:
    """Download or re-download a BioDb from the admin API."""
    task_id = self.request.id
    with Session(engine) as session:
        bio_db = db_crud.get(session, db_id)
        # pyrefly: ignore [bad-argument-type]
        target_dir = _run_bio_db_download(session, bio_db, task_id)
        return {'db_id': db_id, 'path': str(target_dir)}


@celery_app.task(bind=True, pydantic=True, base=BaseBioTask)
def load_bio_db_task(
    self: Task,
    result_dict: ToolOutput | None = None,  # noqa: ARG001
    *,
    db_id: int,
) -> ToolOutput:
    """Load a BioDb path, downloading it lazily when it has not been prepared yet."""
    with Session(engine) as session:
        bio_db = db_crud.get(session, db_id)

        if bio_db.path:
            return ToolOutput(result={'file_path': Path(bio_db.path)})

        if not bio_db.download_command:
            msg = f'BioDb {db_id} has no valid path and no download_command.'
            raise FileNotFoundError(msg)

        # pyrefly: ignore [bad-argument-type]
        target_dir = _run_bio_db_download(session, bio_db, self.request.id, task_instance_uid=self.request.id)
        return ToolOutput(result={'file_path': target_dir})
