"""Dedicated provider credential directories, separate from host CLI accounts."""

import json
import os
import tempfile
from pathlib import Path

from app.core.config import get_settings
from app.schemas.coding_agent import CodingAgentProvider, OpenCodeModelProvider, OpenCodeSettings

CODEX_CONFIG = 'cli_auth_credentials_store = "file"\nforced_login_method = "chatgpt"\n'
OPENCODE_MODEL_PROVIDERS: tuple[OpenCodeModelProvider, ...] = (
    'opencode-go',
    'opencode',
    'anthropic',
    'openai',
    'google',
    'openrouter',
    'deepseek',
    'custom',
)


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
        return isinstance(data, dict) and any(_opencode_credential_configured(value) for value in data.values())
    except (OSError, ValueError, AttributeError):
        return False


def _opencode_credential_configured(value: object) -> bool:
    """Recognize native OpenCode API and OAuth credential entries."""
    return isinstance(value, dict) and (
        (value.get('type') == 'api' and bool(value.get('key')))
        or (value.get('type') == 'oauth' and bool(value.get('refresh')) and bool(value.get('access')))
    )


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


def read_opencode_settings() -> OpenCodeSettings:
    """Return the most recently saved provider without exposing its credential."""
    home = auth_home('opencode')
    auth_path = home / 'auth.json'
    model_provider: OpenCodeModelProvider = 'opencode-go'
    configured = False
    if auth_path.is_file() and not auth_path.is_symlink():
        try:
            credentials = json.loads(auth_path.read_text(encoding='utf-8'))
            if isinstance(credentials, dict):
                selected = next(
                    (
                        provider
                        for provider in reversed(credentials)
                        if provider in OPENCODE_MODEL_PROVIDERS
                        and _opencode_credential_configured(credentials[provider])
                    ),
                    None,
                )
                if selected is not None:
                    model_provider = selected
                    configured = True
        except (OSError, ValueError):
            pass

    if model_provider != 'custom':
        return OpenCodeSettings(configured=configured, model_provider=model_provider)

    config_path = home / 'opencode.json'
    if not config_path.is_file() or config_path.is_symlink():
        return OpenCodeSettings(configured=configured, model_provider='custom')
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
        custom = config.get('provider', {}).get('custom', {})
        base_url = custom.get('options', {}).get('baseURL')
        models = custom.get('models', {})
        model_id = next(iter(models), None) if isinstance(models, dict) else None
        return OpenCodeSettings(
            configured=configured,
            model_provider='custom',
            base_url=base_url if isinstance(base_url, str) else None,
            model_id=model_id if isinstance(model_id, str) else None,
        )
    except (AttributeError, OSError, ValueError):
        return OpenCodeSettings(configured=configured, model_provider='custom')


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    """Read a managed JSON object while rejecting symbolic links and invalid roots."""
    if path.is_symlink():
        message = f'{description} must not be a symbolic link'
        raise ValueError(message)
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(data, dict):
        message = f'Invalid {description}'
        raise TypeError(message)
    return data


def _write_json_object(path: Path, data: dict[str, object]) -> None:
    """Atomically replace one private JSON configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8') as temporary:
        json.dump(data, temporary)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
        path.chmod(0o600)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_opencode_credentials(
    api_key: str,
    model_provider: OpenCodeModelProvider,
    base_url: str | None = None,
    model_id: str | None = None,
) -> OpenCodeSettings:
    """Persist native OpenCode provider credentials without storing secrets in Redis."""
    home = ensure_auth_home('opencode')
    auth_path = home / 'auth.json'
    credentials = _read_json_object(auth_path, 'OpenCode credential file')

    if model_provider == 'custom':
        if base_url is None or model_id is None:
            message = 'Base URL and model ID are required for a custom OpenCode provider'
            raise ValueError(message)
        config_path = home / 'opencode.json'
        config = _read_json_object(config_path, 'OpenCode configuration file')
        providers = config.setdefault('provider', {})
        if not isinstance(providers, dict):
            message = 'Invalid OpenCode provider configuration'
            raise TypeError(message)
        providers['custom'] = {
            'npm': '@ai-sdk/openai-compatible',
            'name': 'Custom',
            'options': {'baseURL': base_url},
            'models': {model_id: {'name': model_id}},
        }
        config.setdefault('$schema', 'https://opencode.ai/config.json')
        _write_json_object(config_path, config)

    credentials.pop(model_provider, None)
    credentials[model_provider] = {'type': 'api', 'key': api_key}
    _write_json_object(auth_path, credentials)
    prepare_auth_home_for_container('opencode')
    return OpenCodeSettings(configured=True, model_provider=model_provider, base_url=base_url, model_id=model_id)
