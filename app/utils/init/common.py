import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import NetworkConfig


def resolve_path(path: str) -> Path:
    """Resolve a user-configured path relative to the current working directory."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    return target


def proxy_environment(network: 'NetworkConfig') -> dict[str, str]:
    """Build child-process environment variables from network proxy config."""
    env = os.environ.copy()
    if not network.USE_PROXY:
        return env

    proxy_url = network.get_proxy_url()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = proxy_url
    return env


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and capture stdout/stderr."""
    return subprocess.run(command, check=False, env=env, capture_output=True, text=True)  # noqa: S603
