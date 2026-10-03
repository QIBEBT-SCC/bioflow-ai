from pathlib import Path

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
