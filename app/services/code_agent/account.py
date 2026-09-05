"""Dedicated Codex account and credential-directory management."""

import os
from pathlib import Path

from app.core.config import get_settings

CODEX_CONFIG = 'cli_auth_credentials_store = "file"\nforced_login_method = "chatgpt"\n'


def auth_home() -> Path:
    """Return the dedicated site-wide Codex home."""
    configured = Path(get_settings().code_agent.AUTH_PATH).expanduser().absolute()
    resolved = configured.resolve()
    host_codex_home = (Path.home() / '.codex').resolve()
    if (
        configured == Path('/')
        or resolved == host_codex_home
        or host_codex_home in resolved.parents
        or configured.is_symlink()
    ):
        message = 'code_agent.auth_path must be a dedicated non-symlink directory'
        raise RuntimeError(message)
    return resolved


def ensure_auth_home() -> Path:
    """Create the private Codex home and force file-backed ChatGPT credentials."""
    path = auth_home()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    config_path = path / 'config.toml'
    if config_path.is_symlink():
        message = 'Codex configuration path must not be a symbolic link'
        raise RuntimeError(message)
    if not config_path.exists():
        config_path.write_text(CODEX_CONFIG, encoding='utf-8')
    config_path.chmod(0o600)
    return path


def credentials_configured() -> bool:
    """Return whether the dedicated Codex home contains a regular auth file."""
    path = auth_home() / 'auth.json'
    return path.is_file() and not path.is_symlink()


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


def prepare_auth_home_for_container() -> Path:
    """Prepare ownership immediately before mounting the credential directory."""
    path = ensure_auth_home()
    _user, owner = container_user()
    if owner is not None:
        for item in (path, *path.iterdir()):
            os.chown(item, owner, owner)
    return path
