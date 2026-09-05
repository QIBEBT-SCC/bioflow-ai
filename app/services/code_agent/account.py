"""Dedicated provider credential directories, separate from host CLI accounts."""

import json
import os
import tempfile
from pathlib import Path

from app.core.config import get_settings
from app.schemas.coding_agent import CodingAgentProvider

CODEX_CONFIG = 'cli_auth_credentials_store = "file"\nforced_login_method = "chatgpt"\n'


def auth_home(provider: CodingAgentProvider = 'codex') -> Path:
    """Return a dedicated directory that cannot overlap another provider's credentials."""
    config = get_settings().code_agent
    paths = {'codex': config.AUTH_PATH, 'opencode': config.OPENCODE_AUTH_PATH}
    configured = Path(paths[provider]).expanduser().absolute()
    resolved = configured.resolve()
    others = [Path(value).expanduser().resolve() for key, value in paths.items() if key != provider]
    host_homes = [(Path.home() / name).resolve() for name in ('.codex', '.local/share/opencode')]
    if (
        configured == Path('/')
        or resolved == Path.home().resolve()
        or any(resolved == home or home in resolved.parents for home in host_homes)
        or configured.is_symlink()
        or any(resolved == other or resolved in other.parents or other in resolved.parents for other in others)
    ):
        message = 'code_agent.auth_path must be a dedicated non-symlink directory'
        raise RuntimeError(message)
    return resolved


def ensure_auth_home(provider: CodingAgentProvider = 'codex') -> Path:
    """Create the private Codex home and force file-backed ChatGPT credentials."""
    path = auth_home(provider)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    if provider != 'codex':
        return path
    config_path = path / 'config.toml'
    if config_path.is_symlink():
        message = 'Codex configuration path must not be a symbolic link'
        raise RuntimeError(message)
    if not config_path.exists():
        config_path.write_text(CODEX_CONFIG, encoding='utf-8')
    config_path.chmod(0o600)
    return path


def credentials_configured(provider: CodingAgentProvider = 'codex') -> bool:
    """Check for configured credentials; this does not validate them with the provider."""
    path = auth_home(provider) / 'auth.json'
    if not path.is_file() or path.is_symlink():
        return False
    if provider == 'codex':
        return True
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return isinstance(data, dict) and any(
            isinstance(value, dict)
            and (
                (value.get('type') == 'api' and bool(value.get('key')))
                or (value.get('type') == 'oauth' and bool(value.get('refresh')) and bool(value.get('access')))
            )
            for value in data.values()
        )
    except (OSError, ValueError, AttributeError):
        return False


def finalize_credentials() -> bool:
    """Validate and lock down the credential file written by Codex."""
    path = auth_home() / 'auth.json'
    if not path.is_file() or path.is_symlink():
        return False
    path.chmod(0o600)
    return True


def remove_credentials() -> None:
    """Remove a partial credential produced by a cancelled or failed login."""
    path = auth_home() / 'auth.json'
    if path.is_symlink():
        message = 'Refusing to remove a symbolic-link Codex credential file'
        raise RuntimeError(message)
    if path.exists():
        path.unlink()


def container_user() -> tuple[str, int | None]:
    """Choose a non-root container identity that can reach the private bind mount."""
    runtime = get_settings().docker.CONTAINER
    if runtime == 'podman':
        return '65532:65532', None
    uid = os.getuid()
    gid = os.getgid()
    if uid != 0:
        return f'{uid}:{gid}', None
    return '65532:65532', 65532


def prepare_auth_home_for_container(provider: CodingAgentProvider = 'codex') -> Path:
    """Prepare ownership immediately before mounting the credential directory."""
    path = ensure_auth_home(provider)
    _user, owner = container_user()
    if owner is not None:
        for item in (path, *path.iterdir()):
            os.chown(item, owner, owner, follow_symlinks=False)
    return path


def write_api_credentials(provider: CodingAgentProvider, api_key: str, model_provider: str) -> None:
    """Persist API credentials atomically without storing secrets in Redis."""
    if provider == 'codex':
        message = 'Use device login for Codex'
        raise ValueError(message)
    home = ensure_auth_home(provider)
    path = home / 'auth.json'
    if path.is_symlink():
        message = 'Credential file must not be a symbolic link'
        raise ValueError(message)
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(data, dict):
        message = 'Invalid OpenCode credential file'
        raise TypeError(message)
    data[model_provider] = {'type': 'api', 'key': api_key}
    with tempfile.NamedTemporaryFile(mode='w', dir=home, delete=False, encoding='utf-8') as temporary:
        temporary.write(json.dumps(data))
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
        prepare_auth_home_for_container(provider)
    finally:
        temporary_path.unlink(missing_ok=True)
