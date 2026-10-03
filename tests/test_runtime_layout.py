from pathlib import Path
from textwrap import dedent

import pytest


def test_code_execution_filenames() -> None:
    from bioflow_ai.llm.utils.code_node_contract import SCRIPT_NAMES

    assert set(SCRIPT_NAMES.values()) == {'script.sh', 'script.py', 'script.R'}


def test_worker_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    from bioflow_ai.core import config

    settings = config.Settings()
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    from bioflow_ai.runtime import worker

    arguments = []
    monkeypatch.setattr(worker.celery_app, 'worker_main', lambda *, argv: arguments.append(argv))
    worker.agent_main(concurrency=2)
    worker.execution_main(concurrency=5)
    assert '--queues=agent,acp' in arguments[0]
    assert '--concurrency=2' in arguments[0]
    assert '--queues=workflow-v2,resource' in arguments[1]
    assert '--concurrency=5' in arguments[1]
    assert '--prefetch-multiplier=1' in arguments[1]


def test_preview_worker_subprocess(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    bed = tmp_path / 'intervals.bed'
    bed.write_text('chr1\t0\t10\tfeature\n')
    stat = bed.stat()
    request = {
        'root': str(tmp_path),
        'path': str(bed),
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
        'operation': 'meta',
    }
    result = subprocess.run(
        [sys.executable, '-m', 'bioflow_ai.services.file_preview.worker'],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        cwd=tmp_path,
    )
    response = json.loads(result.stdout)
    assert response['ok'], response
    assert response['data']['chromosomes'] == [{'name': 'chr1', 'length': 10}]


def test_task_discovery_and_checkpoint_types(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    # Only provider/model loading is replaced. Actual task modules, decorators,
    # discovery, routes, and checkpoint serializer run in a fresh interpreter.
    script = dedent("""
        import json
        import socket
        from decimal import Decimal
        from bioflow_ai.core import config
        settings = config.Settings()
        config.get_settings = lambda: settings
        def reject_connection(*args, **kwargs):
            raise AssertionError('Task discovery must not contact external services')
        socket.socket.connect = reject_connection
        from bioflow_ai.llm import models
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        FakeListChatModel.with_structured_output = lambda self, *args, **kwargs: self
        FakeListChatModel.bind_tools = lambda self, *args, **kwargs: self
        def fake_model():
            model = FakeListChatModel(responses=['offline'])
            model.metadata = dict(model_id=1, setting_key='coding', input_price=Decimal(0),
                                  output_price=Decimal(0), cache_read_price=Decimal(0))
            return model
        for name in dir(models):
            if name.startswith('get_') and name.endswith('_model'):
                setattr(models, name, fake_model)
        from bioflow_ai.runtime.celery_app import celery_app
        celery_app.loader.import_default_modules()
        names = sorted(name for name in celery_app.tasks if name.startswith('bioflow_ai.'))
        from bioflow_ai.tasks.agent_run import _checkpoint_serializer
        from bioflow_ai.schemas.tool_config import ImageConfig, FileMount, ParamDefine
        from bioflow_ai.llm.middleware.todo import Todo
        values = [ImageConfig(repository='demo'),
                  FileMount(name='reads', description='input', file_type='INPUT',
                            file_path='reads.fq', mount_path='/data/reads.fq'),
                  ParamDefine(description='threads', command='--threads'),
                  Todo(content='inspect', status='pending')]
        serializer = _checkpoint_serializer()
        for value in values:
            restored = serializer.loads_typed(serializer.dumps_typed(value))
            assert type(restored) is type(value) and restored == value
        print(json.dumps({'names': names, 'routes': celery_app.conf.task_routes}))
    """)
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30
    )
    registered = json.loads(result.stdout)
    expected = {
        'bioflow_ai.tasks.agent_run.execute_agent_run': {'queue': 'agent'},
        'bioflow_ai.tasks.code_agent.execute_code_agent_session': {'queue': 'acp'},
        'bioflow_ai.tasks.code_agent.login_codex_account': {'queue': 'acp'},
        'bioflow_ai.tasks.load_tasks.load_bio_db.download_bio_db_task': {'queue': 'resource'},
        'bioflow_ai.tasks.load_tasks.load_genome.download_genome_admin_task': {'queue': 'resource'},
        'bioflow_ai.tasks.load_tasks.load_genome.build_genome_index_admin_task': {'queue': 'resource'},
        'bioflow_ai.tasks.workflow.execute_node': {'queue': 'workflow-v2'},
    }
    assert registered['names'] == sorted(expected)
    assert registered['routes'] == expected


@pytest.mark.parametrize('runtime', ['docker', 'podman'])
def test_shared_container_process_arguments(runtime: str) -> None:
    from bioflow_ai.infrastructure.containers._general import ContainerProcessSpec, build_process_arguments
    from bioflow_ai.schemas.tool_config import ImageConfig

    spec = ContainerProcessSpec(
        image=ImageConfig(repository='alpine'),
        command=('echo', 'hello world'),
        name='test-container',
        environment={'VISIBLE': 'ordinary value'},
        host_environment={'SECRET': 'private value'},
    )
    argv, host_environment = build_process_arguments(spec, runtime)
    assert argv[-3:] == ['docker.io/library/alpine:latest', 'echo', 'hello world']
    assert 'VISIBLE=ordinary value' in argv
    assert 'SECRET' in argv
    assert 'private value' not in ' '.join(argv)
    assert host_environment == {'SECRET': 'private value'}
