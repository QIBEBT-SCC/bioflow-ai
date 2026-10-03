import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import duckdb
import pytest


def _development_database(path: Path) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute("""
            CREATE TABLE tool_functions (
                index INTEGER, tool_name VARCHAR, function_description VARCHAR, help_command VARCHAR,
                status VARCHAR, tool_uid VARCHAR, thread_id VARCHAR, attempt_count INTEGER DEFAULT 0,
                started_at TIMESTAMP, finished_at TIMESTAMP, last_error VARCHAR,
                updated_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        connection.execute("""
            INSERT INTO tool_functions (index, tool_name, function_description, help_command, status)
            VALUES (1, 'demo', 'old requirement', 'demo --help', 'PENDING'),
                   (2, 'other', 'other requirement', 'other --help', 'PENDING')
        """)


def test_development_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from bioflow_ai.services.development import records

    path = tmp_path / 'develop.db'
    _development_database(path)
    monkeypatch.setattr(records, 'DB_PATH', str(path))
    assert [row['index'] for row in records.search_tools('demo')] == [1]
    previous, updated = records.update_tool_requirement(1, 'new requirement')
    assert previous['function_description'] == 'old requirement'
    assert updated['function_description'] == 'new requirement'
    assert records.search_tools('other')[0]['function_description'] == 'other requirement'
    with pytest.raises(ValueError, match='must not be empty'):
        records.update_tool_requirement(1, ' ')
    stats = records.statistic()
    assert stats['total'] == len(records.search_tools(''))
    assert stats['pending'] == stats['total']
    assert stats['latest'] is None


@pytest.mark.parametrize('outcome', ['completed', 'interrupted', 'failed'])
def test_tool_generation_state_and_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str) -> None:
    from bioflow_ai.services.development import tool_generation

    path = tmp_path / 'develop.db'
    _development_database(path)
    monkeypatch.setattr(tool_generation, 'DB_PATH', str(path))
    monkeypatch.setattr(tool_generation, '_prepare_tool_resources', lambda *_: (uuid4(), object(), uuid4()))
    emitted: list[tuple[str, Any]] = []
    tool_uid = str(uuid4())

    class FakeAgent:
        async def astream(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            yield {'type': 'custom', 'data': {'event': 'progress'}}
            if outcome == 'failed':
                message = 'fake model failure'
                raise RuntimeError(message)

        async def aget_state(self, _config: Any) -> Any:
            interrupts = [SimpleNamespace(value='approval needed')] if outcome == 'interrupted' else []
            return SimpleNamespace(interrupts=interrupts, values={'tool_uid': tool_uid, 'name': 'demo'})

    @asynccontextmanager
    async def fake_session(*, setup: bool = False) -> AsyncIterator[tuple[object, FakeAgent]]:
        assert setup
        yield object(), FakeAgent()

    async def fake_notification(_event: Any, _message: str) -> None:
        return None

    monkeypatch.setattr(tool_generation, '_tool_agent_session', fake_session)
    monkeypatch.setattr(tool_generation, '_publish_tool_creation_event', fake_notification)
    invocation = tool_generation.init_tools(1, on_event=lambda kind, payload: emitted.append((kind, payload)))
    if outcome == 'failed':
        with pytest.raises(RuntimeError, match='fake model failure'):
            asyncio.run(invocation)
    else:
        asyncio.run(invocation)
    with duckdb.connect(str(path)) as connection:
        row = connection.execute(
            'SELECT status, attempt_count, tool_uid, last_error FROM tool_functions WHERE index = 1'
        ).fetchone()
    assert row is not None
    assert row[0] == outcome.upper()
    assert row[1] == 1
    assert row[2] == (tool_uid if outcome == 'completed' else None)
    assert bool(row[3]) == (outcome == 'failed')
    assert emitted[0][0] == 'chunk'
    assert any(kind == 'interrupt' for kind, _payload in emitted) == (outcome == 'interrupted')


def test_evaluation_checkpoint_inputs() -> None:
    from bioflow_ai.services.evaluation.workflow import (
        WorkflowResumeRequiredError,
        WorkflowResumeUnavailableError,
        _prepare_workflow_stream_input,
    )

    fresh = {'messages': ['new request']}
    assert _prepare_workflow_stream_input(SimpleNamespace(interrupts=[], next=()), fresh, resume=None) is fresh
    assert _prepare_workflow_stream_input(SimpleNamespace(interrupts=[], next=('tools',)), fresh, resume=None) is None
    paused = SimpleNamespace(interrupts=[object()], next=())
    with pytest.raises(WorkflowResumeRequiredError):
        _prepare_workflow_stream_input(paused, fresh, resume=None)
    assert _prepare_workflow_stream_input(paused, fresh, resume='approved').resume == 'approved'
    with pytest.raises(WorkflowResumeUnavailableError):
        _prepare_workflow_stream_input(SimpleNamespace(interrupts=[], next=()), fresh, resume='approved')


def test_administrator_prompt_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from bioflow_ai.cli.initialization import database as wizard

    monkeypatch.setattr(wizard.database, 'administrator_exists', lambda: False)
    answers = iter(['  ADMIN@example.com ', ' admin '])

    async def text(_message: str, **_kwargs: Any) -> str:
        return next(answers)

    async def password(_message: str, **_kwargs: Any) -> str:
        return 'test-password'

    async def create(data: Any) -> Any:
        assert data.email == 'admin@example.com'
        assert data.username == 'admin'
        assert data.password.get_secret_value() == 'test-password'
        return wizard.AdminInitResult(created=True)

    monkeypatch.setattr(wizard.tui, 'prompt_text_async', text)
    monkeypatch.setattr(wizard.tui, 'prompt_password_async', password)
    monkeypatch.setattr(wizard.database, 'init_admin', create)
    assert asyncio.run(wizard.init_admin()).created
    monkeypatch.setattr(wizard.database, 'administrator_exists', lambda: True)
    assert not asyncio.run(wizard.init_admin()).created


def test_compose_and_settings_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tomllib

    import yaml

    from bioflow_ai.core.config import Settings
    from bioflow_ai.services.initialization.config import apply_update_port, write_settings
    from bioflow_ai.services.initialization.containers import compose_template_file, load_compose_file
    from bioflow_ai.services.initialization.ports import UpdatePort

    monkeypatch.chdir(tmp_path)
    ports = UpdatePort(api=8100, postgres=5433, redis=6380)
    settings = Settings()
    apply_update_port(settings, ports)
    write_settings(settings, tmp_path / 'config.toml')
    saved = tomllib.loads((tmp_path / 'config.toml').read_text())
    assert saved['network']['port'] == ports.api
    assert ':5433/' in saved['persistent']['database']['db_url']
    for runtime in ('docker', 'podman'):
        result = load_compose_file(8100, ports, compose_template_file(runtime))
        assert result.success
        compose = yaml.safe_load((tmp_path / 'docker-compose.yml').read_text())
        assert './presets/sql:/docker-entrypoint-initdb.d' in compose['services']['postgres-db']['volumes']


def test_preset_import_resource_location(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from bioflow_ai.services.initialization import database
    from bioflow_ai.services.initialization.database import _ensure_preset_file

    presets = tmp_path / 'repository' / 'presets'
    presets.mkdir(parents=True)
    (presets / 'images.parquet').write_bytes(b'fixture')
    monkeypatch.setattr(database, 'PRESETS_DIR', presets)
    monkeypatch.chdir(tmp_path)
    result = asyncio.run(_ensure_preset_file('images.parquet'))
    assert result.path == str(presets / 'images.parquet')
    assert not result.downloaded


def test_editable_skills(tmp_path: Path) -> None:
    from bioflow_ai.services.skills.catalog import SkillCatalogError, create_skill, get_resource, update_resource

    skills = tmp_path / 'skills'
    (skills / 'workflow_builder').mkdir(parents=True)
    content = '---\nname: demo\ndescription: Test skill\n---\nInstructions\n'
    created = create_skill('workflow_builder', 'demo', content, skills_dir=skills)
    assert created.content == content
    update_resource('workflow_builder', 'demo', 'references/test.md', 'resource', skills_dir=skills)
    assert get_resource('workflow_builder', 'demo', 'references/test.md', skills_dir=skills).content == 'resource'
    with pytest.raises(SkillCatalogError):
        update_resource('workflow_builder', 'demo', '../../escape', 'invalid', skills_dir=skills)


def test_trajectory_export_is_atomic_and_preserves_events(tmp_path: Path) -> None:
    import json

    from bioflow_ai.services.export.trajectories import compact_trajectories

    source = tmp_path / 'trajectories'
    events = source / 'agent=tool_generator' / 'run=demo' / 'events'
    events.mkdir(parents=True)
    payload = {'schema_version': 1, 'agent': 'tool_generator', 'run_id': 'demo', 'payload': {'text': 'saved'}}
    for index, kind in enumerate(('agent_start', 'agent_end')):
        (events / f'{index}.json').write_text(
            json.dumps({**payload, 'event_id': str(index), 'event_type': kind, 'recorded_at_ns': index})
        )
    output = tmp_path / 'dataset'
    compact_trajectories(str(source), str(output), (), skip_invalid=False, overwrite=False)
    original = (output / 'metadata.json').read_bytes()
    with duckdb.connect() as connection:
        assert connection.execute('SELECT status FROM read_parquet(?)', [str(output / 'runs.parquet')]).fetchall() == [
            ('completed',)
        ]
        assert connection.execute(
            'SELECT payload FROM read_parquet(?) ORDER BY recorded_at_ns', [str(output / 'events.parquet')]
        ).fetchall() == [(json.dumps(payload['payload'], separators=(',', ':')),)] * len(tuple(events.glob('*.json')))
    (events / 'invalid.json').write_text('{}')
    with pytest.raises(ValueError, match='Invalid trajectory file'):
        compact_trajectories(str(source), str(output), (), skip_invalid=False, overwrite=True)
    assert (output / 'metadata.json').read_bytes() == original
    assert not tuple(tmp_path.glob('.dataset.*'))
