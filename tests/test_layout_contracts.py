import hashlib
import json
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

BASELINE = json.loads((Path(__file__).parent / 'fixtures' / 'layout_baseline.json').read_text())


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _command_spec(command: click.Command) -> dict[str, Any]:
    spec = {
        'name': command.name,
        'help': command.help,
        'params': [
            {
                'name': parameter.name,
                'opts': getattr(parameter, 'opts', []),
                'secondary_opts': getattr(parameter, 'secondary_opts', []),
                'default': str(parameter.default),
                'required': parameter.required,
                'nargs': parameter.nargs,
                'type': (
                    {
                        'class': 'Path',
                        **{
                            name: getattr(parameter.type, name)
                            for name in ('exists', 'file_okay', 'dir_okay', 'readable', 'writable', 'resolve_path')
                        },
                        'path_type': getattr(parameter.type.type, '__name__', None),
                    }
                    if isinstance(parameter.type, click.Path)
                    else str(parameter.type)
                ),
            }
            for parameter in command.params
        ],
    }
    if isinstance(command, click.Group):
        spec['commands'] = {name: _command_spec(child) for name, child in command.commands.items()}
    return spec


def test_cli_contract_without_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    from bioflow_ai.cli.main import cli

    assert _digest(_command_spec(cli)) == BASELINE['cli']
    for arguments in (['--help'], *[[name, '--help'] for name in cli.commands]):
        result = CliRunner().invoke(cli, arguments)
        assert result.exit_code == 0, result.output


def test_api_and_model_contracts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    from bioflow_ai.core import config

    settings = config.Settings()
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    from bioflow_ai.models import Base
    from bioflow_ai.runtime.server import app

    # Importing the application does not run its lifespan or contact services.
    assert _digest(app.openapi()) == BASELINE['openapi']
    models = {
        name: {
            'columns': {
                column.name: {
                    'type': str(column.type),
                    'nullable': column.nullable,
                    'primary_key': column.primary_key,
                }
                for column in table.columns
            },
            'indexes': sorted(index.name for index in table.indexes),
        }
        for name, table in Base.metadata.tables.items()
    }
    assert _digest(models) == BASELINE['models']


def test_repository_resources() -> None:
    from bioflow_ai.core.paths import DEPLOY_DIR, PRESETS_DIR, SKILLS_DIR
    from bioflow_ai.services.skills.definitions import discover_agent_skills

    assert (PRESETS_DIR / 'sql' / '03_create_table.sql').is_file()
    assert (DEPLOY_DIR / 'docker' / 'docker-compose.yml').is_file()
    assert 'using-foreach-subgraphs' in discover_agent_skills('workflow_builder', SKILLS_DIR)
