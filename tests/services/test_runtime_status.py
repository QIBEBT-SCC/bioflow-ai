from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from celery import states

from app.models.workflow import Status
from app.schemas.common import Statistics
from app.services import runtime_status


def _set_celery_state(monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    monkeypatch.setattr(
        runtime_status.celery_app,
        'AsyncResult',
        MagicMock(return_value=SimpleNamespace(state=state)),
    )


def test_get_task_status_normalizes_integer_database_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_celery_state(monkeypatch, 'UNKNOWN')

    status = runtime_status.get_task_status(str(uuid4()), Status.RUNNING.value)

    assert status is Status.RUNNING


def test_apply_run_stats_accepts_integer_statuses_from_database(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_celery_state(monkeypatch, states.PENDING)
    task = SimpleNamespace(uid=uuid4(), status=Status.WAITING.value)
    run = SimpleNamespace(status=Status.RUNNING.value, tasks=[task])
    database_stats = Statistics(total=1, running=1)

    result = runtime_status.apply_effective_nonterminal_run_stats(database_stats, [run])

    assert result == Statistics(total=1, waiting=1)
    assert run.status is Status.WAITING
    assert task.status is Status.WAITING


def test_refresh_run_status_preserves_integer_terminal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_celery_state(monkeypatch, states.PENDING)
    run = SimpleNamespace(
        status=Status.SUCCESS.value,
        tasks=[SimpleNamespace(uid=uuid4(), status=Status.WAITING.value)],
    )

    runtime_status.refresh_run_status(run)

    assert run.status is Status.SUCCESS
