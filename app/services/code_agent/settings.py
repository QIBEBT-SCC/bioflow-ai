"""Persist administrator settings in the shared native Codex config file."""

import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from app.schemas.coding_agent import CodexAgentSettings
from app.services.code_agent.account import auth_home, ensure_auth_home


def read_settings() -> CodexAgentSettings:
    """Read site defaults without requiring a connected account."""
    path = auth_home() / 'config.toml'
    data: dict[str, Any] = tomllib.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    return CodexAgentSettings.model_validate(
        {
            'sandbox_mode': data.get('sandbox_mode', 'workspace-write'),
            'web_search': data.get('web_search', 'live'),
            'network_access': data.get('sandbox_workspace_write', {}).get('network_access', True),
        }
    )


def write_settings(settings: CodexAgentSettings) -> CodexAgentSettings:
    """Atomically replace native configuration while preserving account settings."""
    path = ensure_auth_home() / 'config.toml'
    data = tomllib.loads(path.read_text(encoding='utf-8'))
    data.update(settings.codex_config())
    owner = path.stat()
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8') as temporary:
        temporary.write(tomli_w.dumps(data))
        temporary_path = Path(temporary.name)
    try:
        if os.geteuid() == 0:
            os.chown(temporary_path, owner.st_uid, owner.st_gid)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return settings
