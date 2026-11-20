"""Verify the installed wheel can initialize an empty deployment directory offline."""

# ruff: noqa: INP001

from contextlib import chdir
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from bioflow_ai.cli.initialization import config, containers, tui, wizard
from bioflow_ai.cli.main import cli
from bioflow_ai.core.config import get_settings
from bioflow_ai.core.paths import BUILTIN_PRESETS_DIR, SKILLS_DIR
from bioflow_ai.services.initialization.containers import ComposeCommandResult, OperationResult
from bioflow_ai.services.initialization.ports import UpdatePort
from bioflow_ai.services.skills.definitions import discover_agent_skills


def check_deployment(runtime: str) -> None:
    runner = CliRunner()
    compose_command = [runtime, 'compose']
    get_settings.cache_clear()

    with (
        TemporaryDirectory() as deployment,
        chdir(deployment),
        patch.multiple(
            tui,
            prompt_confirm=lambda message, **_: message.startswith('Start project containers'),
            prompt_password=lambda _: '',
            prompt_path=lambda _, *, default: default,
            prompt_select=lambda message, *, default, **_: runtime if message == 'Container runtime' else default,
            prompt_text=lambda _, *, default='', **_options: default,
        ),
        patch.object(config, 'create_update_port', return_value=UpdatePort()),
        patch.object(containers, 'check_compose_command', return_value=ComposeCommandResult(command=compose_command)),
        patch.object(containers, 'start_project_services', return_value=CompletedProcess([], 0)) as start,
        patch.object(containers, 'wait_for_project_services', return_value=OperationResult(success=True)),
        patch.object(wizard, 'initialize_preset_database', return_value=False),
    ):
        assert not Path('config.toml').exists()
        result = runner.invoke(cli, ['auto-init'], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert Path('config.toml').is_file()
        start.assert_called_once_with([*compose_command, '-f', 'docker-compose.yml', 'up', '-d'])

        compose = yaml.safe_load(Path('docker-compose.yml').read_text())
        assert './presets/sql:/docker-entrypoint-initdb.d' in compose['services']['postgres-db']['volumes']
        sql_file = Path('presets/sql/03_create_table.sql')
        assert sql_file.read_bytes() == (BUILTIN_PRESETS_DIR / 'sql' / sql_file.name).read_bytes()

        assert discover_agent_skills('coding_agent') == {}
        skills = discover_agent_skills('workflow_builder')
        assert skills
        edited = next(iter(skills.values())).directory / 'SKILL.md'
        original = edited.read_text()
        edited.write_text(original + '\nLocal edit\n')
        sql_file.unlink()

        result = runner.invoke(cli, ['auto-init'], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert edited.read_text() == original + '\nLocal edit\n'
        assert sql_file.is_file()


if __name__ == '__main__':
    assert BUILTIN_PRESETS_DIR.parent.name == '_resources', BUILTIN_PRESETS_DIR
    assert Path('presets/skills') == SKILLS_DIR
    for language in ('bash', 'python', 'r'):
        assert (BUILTIN_PRESETS_DIR / 'code' / language / 'AGENTS.md').is_file()
    for container_runtime in ('docker', 'podman'):
        check_deployment(container_runtime)
