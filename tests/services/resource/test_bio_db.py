import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from celery import states

import app.services.resource.bio_db as bio_db_service
from app.schemas.resource import BioDbDownloadStatus
from app.tasks.load_tasks import load_bio_db


def _configure_download_task(
    monkeypatch: pytest.MonkeyPatch,
    *,
    celery_state: str | Exception,
) -> SimpleNamespace:
    redis = SimpleNamespace(
        get=AsyncMock(return_value=b'task-123'),
        eval=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(bio_db_service, 'monitor_redis_client', redis)

    if isinstance(celery_state, Exception):
        async_result = MagicMock(side_effect=celery_state)
    else:
        async_result = MagicMock(return_value=SimpleNamespace(state=celery_state))
    monkeypatch.setattr(bio_db_service.celery_app, 'AsyncResult', async_result)
    return redis


def test_completed_download_clears_stale_lock_and_returns_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _configure_download_task(monkeypatch, celery_state=states.SUCCESS)
    monkeypatch.setattr(bio_db_service.db_crud, 'get', MagicMock(return_value=SimpleNamespace(path='/data/db')))

    status = asyncio.run(bio_db_service.build_download_status(21, MagicMock()))

    assert status.status == BioDbDownloadStatus.ready
    redis.eval.assert_awaited_once_with(
        ANY,
        1,
        'bio-db:21:download-task-id',
        'task-123',
    )


def test_failed_download_clears_stale_lock_and_returns_not_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _configure_download_task(monkeypatch, celery_state=states.FAILURE)
    monkeypatch.setattr(bio_db_service.db_crud, 'get', MagicMock(return_value=SimpleNamespace(path=None)))

    status = asyncio.run(bio_db_service.build_download_status(21, MagicMock()))

    assert status.status == BioDbDownloadStatus.not_downloaded
    redis.eval.assert_awaited_once()


def test_running_download_keeps_lock_and_returns_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _configure_download_task(monkeypatch, celery_state=states.STARTED)
    get_bio_db = MagicMock()
    monkeypatch.setattr(bio_db_service.db_crud, 'get', get_bio_db)

    status = asyncio.run(bio_db_service.build_download_status(21, MagicMock()))

    assert status.status == BioDbDownloadStatus.downloading
    redis.eval.assert_not_awaited()
    get_bio_db.assert_not_called()


def test_completed_download_returns_ready_when_lock_cleanup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _configure_download_task(monkeypatch, celery_state=states.SUCCESS)
    redis.eval.side_effect = RuntimeError('redis unavailable')
    monkeypatch.setattr(bio_db_service.db_crud, 'get', MagicMock(return_value=SimpleNamespace(path='/data/db')))

    status = asyncio.run(bio_db_service.build_download_status(21, MagicMock()))

    assert status.status == BioDbDownloadStatus.ready


def test_backend_error_preserves_lock_and_reports_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _configure_download_task(monkeypatch, celery_state=RuntimeError('backend unavailable'))

    status = asyncio.run(bio_db_service.build_download_status_from_redis(21))

    assert status is not None
    assert status.status == BioDbDownloadStatus.downloading
    redis.eval.assert_not_awaited()


def test_download_container_runs_as_current_host_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target_dir = tmp_path / 'database'
    bio_db = SimpleNamespace(
        id=21,
        name='antiSMASH',
        path=str(target_dir),
        download_command='download-antismash-databases',
        download_image=None,
    )
    run_command = MagicMock(return_value=(0, ''))
    monkeypatch.setattr(load_bio_db, '_acquire_download_slot', MagicMock())
    monkeypatch.setattr(load_bio_db, '_release_download_slot', MagicMock())
    monkeypatch.setattr(load_bio_db.db_crud, 'clear_download_result', MagicMock())
    monkeypatch.setattr(load_bio_db.db_crud, 'update_download_result', MagicMock())
    monkeypatch.setattr(load_bio_db, 'get_current_user_identity', MagicMock(return_value='1000:1000'))
    monkeypatch.setattr(load_bio_db, 'run_command', run_command)

    result = load_bio_db._run_bio_db_download(MagicMock(), bio_db, 'task-123')  # noqa: SLF001

    assert result == target_dir.absolute()
    assert run_command.call_args.kwargs['user'] == '1000:1000'
