import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bioflow_ai.core.config import NetworkConfig

PROXY_ENV_KEYS: tuple[str, ...] = (
    'HTTP_PROXY',
    'HTTPS_PROXY',
    'ALL_PROXY',
    'http_proxy',
    'https_proxy',
    'all_proxy',
)
ALL_PROXY_ENV_KEYS: tuple[str, ...] = (*PROXY_ENV_KEYS, 'FTP_PROXY', 'ftp_proxy')


def resolve_path(path: str) -> Path:
    """Resolve a user-configured path relative to the current working directory."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    return target


def detect_shell_proxy() -> str | None:
    """Return the first non-empty proxy URL already set in the environment."""
    for key in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy'):
        value = os.environ.get(key, '').strip()
        if value:
            return value
    return None


def proxy_environment(network: 'NetworkConfig') -> dict[str, str]:
    """Build child-process environment variables from network proxy config."""
    env = os.environ.copy()
    if not network.USE_PROXY:
        return env

    proxy_url = network.get_proxy_url()
    for key in PROXY_ENV_KEYS:
        env[key] = proxy_url
    return env


def no_proxy_environment() -> dict[str, str]:
    """Build a child-process environment with all proxy variables stripped."""
    env = os.environ.copy()
    for key in ALL_PROXY_ENV_KEYS:
        env.pop(key, None)
    return env


@contextmanager
def applied_proxy_environment(network: 'NetworkConfig') -> Iterator[None]:
    """Temporarily apply configured proxy env vars and reset huggingface_hub HTTP sessions."""
    if not network.USE_PROXY:
        yield
        return

    from huggingface_hub import close_session

    previous = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
    proxy_url = network.get_proxy_url()
    for key in PROXY_ENV_KEYS:
        os.environ[key] = proxy_url
    close_session()
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        close_session()


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and capture stdout/stderr."""
    return subprocess.run(command, check=False, env=env, capture_output=True, text=True)
