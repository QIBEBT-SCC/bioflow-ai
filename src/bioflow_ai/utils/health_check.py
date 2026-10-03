import subprocess
from pathlib import Path


def _check_service(
    compose_command: list[str],
    compose_file: Path,
    service: str,
    command: list[str],
) -> bool:
    result = subprocess.run(
        [*compose_command, '-f', str(compose_file), 'exec', '-T', service, *command],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def check_postgres(compose_command: list[str], compose_file: Path) -> bool:
    """Return whether PostgreSQL accepts TCP connections."""
    return _check_service(
        compose_command,
        compose_file,
        'postgres-db',
        ['pg_isready', '-h', '127.0.0.1', '-U', 'bioflow_ai', '-d', 'bioflow_ai'],
    )


def check_rustfs(compose_command: list[str], compose_file: Path) -> bool:
    """Return whether the RustFS health endpoint responds successfully."""
    return _check_service(
        compose_command,
        compose_file,
        'rustfs',
        ['curl', '-fsS', 'http://localhost:9000/health'],
    )


def check_milvus(compose_command: list[str], compose_file: Path) -> bool:
    """Return whether the Milvus health endpoint responds successfully."""
    return _check_service(
        compose_command,
        compose_file,
        'standalone',
        ['curl', '-fsS', 'http://localhost:9091/healthz'],
    )
