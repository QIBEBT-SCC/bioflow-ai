import socket

from pydantic import BaseModel, Field

DEFAULT_API_PORT = 8000
MAX_PORT = 65536
SERVICE_DEFAULT_PORTS = {
    'api': DEFAULT_API_PORT,
    'postgres': 5432,
    'redis': 6379,
    'rabbitmq': 5672,
    'rustfs': 9000,
    'milvus': 19530,
    'web_ui': 3000,
}


class UpdatePort(BaseModel):
    """Ports shifted during initialization; unset fields keep project defaults."""

    api: int | None = Field(default=None, description='API backend')
    postgres: int | None = Field(default=None, description='PostgreSQL database')
    redis: int | None = Field(default=None, description='Redis cache/result backend')
    rabbitmq: int | None = Field(default=None, description='RabbitMQ queue')
    rustfs: int | None = Field(default=None, description='RustFS/S3 storage')
    milvus: int | None = Field(default=None, description='Milvus vector database')
    web_ui: int | None = Field(default=None, description='Web UI')


def get_service_label(key: str) -> str:
    """Return a display label from the UpdatePort field description."""
    return UpdatePort.model_fields[key].description or key


def get_service_port(update_port: UpdatePort, key: str) -> int:
    """Return the assigned service port, falling back to the default."""
    port = getattr(update_port, key)
    return port if port is not None else SERVICE_DEFAULT_PORTS[key]


def assigned_service_ports(update_port: UpdatePort, *, exclude: set[str] | None = None) -> set[int]:
    """Return all assigned service ports, optionally excluding service keys."""
    exclude = exclude or set()
    return {get_service_port(update_port, key) for key in SERVICE_DEFAULT_PORTS if key not in exclude}


def is_port_available(port: int, host: str = '0.0.0.0') -> bool:  # noqa: S104
    """Check whether a host port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def find_available_port(
    start: int,
    host: str = '0.0.0.0',  # noqa: S104
    limit: int = 100,
    reserved_ports: set[int] | None = None,
) -> int:
    """Find the first available port from start, avoiding already assigned ports."""
    reserved_ports = reserved_ports or set()
    for port in range(start, min(start + limit, MAX_PORT)):
        if port not in reserved_ports and is_port_available(port, host):
            return port
    return start


def create_update_port() -> UpdatePort:
    """Preflight known default ports and store only shifted ports."""
    update_port = UpdatePort()
    reserved_ports: set[int] = set()

    for key, default_port in SERVICE_DEFAULT_PORTS.items():
        port = find_available_port(default_port, reserved_ports=reserved_ports)
        if port != default_port:
            setattr(update_port, key, port)
        reserved_ports.add(port)

    return update_port
