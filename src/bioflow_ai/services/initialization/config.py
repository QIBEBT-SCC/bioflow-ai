from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w
from pydantic import AmqpDsn, PostgresDsn, RedisDsn, SecretStr

from bioflow_ai.services.initialization.ports import (
    UpdatePort,
)

if TYPE_CHECKING:
    from bioflow_ai.core.config import (
        Settings,
    )


def validate_int(text: str) -> bool | str:
    """Validate that the input is a positive integer."""
    if text.strip().isdigit() and int(text) > 0:
        return True
    return 'Enter a positive integer.'


def sanitize_config(value: Any) -> Any:
    """Convert settings values into TOML-serializable primitives."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, dict):
        return {k: sanitize_config(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [sanitize_config(v) for v in value]
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def write_settings(settings: Any, config_path: Path) -> None:
    """Write settings to a TOML config file."""
    config_dict = sanitize_config(settings.model_dump(by_alias=True, mode='python', exclude_none=True))
    with config_path.open('wb') as f:
        tomli_w.dump(config_dict, f)


def apply_update_port(settings: 'Settings', update_port: UpdatePort) -> None:
    """Apply automatically assigned dependency ports to settings."""
    if host_port := update_port.api:
        settings.network.PORT = host_port

    if postgres_port := update_port.postgres:
        settings.persistent.database.DB_URL = PostgresDsn(
            f'postgresql+psycopg://bioflow_ai:123456@127.0.0.1:{postgres_port}/bioflow_ai'
        )

    if redis_port := update_port.redis:
        settings.persistent.cache.LLM_CACHE = RedisDsn(f'redis://localhost:{redis_port}/0')
        settings.persistent.cache.MONITOR_CACHE = RedisDsn(f'redis://localhost:{redis_port}/2')
        settings.persistent.cache.API_CACHE = RedisDsn(f'redis://localhost:{redis_port}/2')
        settings.persistent.cache.NOTIFICATION_CACHE = RedisDsn(f'redis://localhost:{redis_port}/2')
        settings.celery.BACKEND = RedisDsn(f'redis://localhost:{redis_port}/1')
        settings.develop.CHECKPOINT_URL = RedisDsn(f'redis://localhost:{redis_port}/0')

    if rabbitmq_port := update_port.rabbitmq:
        settings.celery.BROKER = AmqpDsn(f'amqp://bioflow_ai:123456@localhost:{rabbitmq_port}//')

    if rustfs_port := update_port.rustfs:
        settings.persistent.storage.s3.ENDPOINT = f'http://localhost:{rustfs_port}'

    if milvus_port := update_port.milvus:
        settings.knowledge_base.MILVUS_URL = f'http://localhost:{milvus_port}'
