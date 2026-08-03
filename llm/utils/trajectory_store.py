import hashlib
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import orjson

from app.core.config import get_settings

TRAJECTORY_DIRECTORY = 'agent_trajectories'
SCHEMA_VERSION = 1
_PATH_COMPONENT_PATTERN = re.compile(r'[^A-Za-z0-9._-]+')


class TrajectoryStore(Protocol):
    """Storage interface for append-only agent trajectory events."""

    def append_event(self, agent_name: str, run_id: str, event: Mapping[str, Any]) -> Path:
        """Persist one immutable event and return its path."""
        ...


def get_trajectory_root() -> Path:
    """Return the configured persistent root for immutable agent trajectories."""
    storage_root = Path(get_settings().persistent.storage.STORAGE_PATH).expanduser().resolve()
    return storage_root / TRAJECTORY_DIRECTORY


def _safe_component(value: str, *, max_length: int = 80) -> str:
    normalized = _PATH_COMPONENT_PATTERN.sub('_', value).strip('._-') or 'unknown'
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f'{normalized[:max_length]}-{digest}'


def _sync_directory(path: Path) -> None:
    """Durably persist a rename when the platform supports directory fsync."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    descriptor = os.open(temp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
        _sync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


class LocalEventStore:
    """Store each trajectory event as an immutable, atomically committed JSON file."""

    def __init__(self, root_path: str | Path | None = None):
        self.root_path = Path(root_path).expanduser().resolve() if root_path is not None else get_trajectory_root()

    def append_event(self, agent_name: str, run_id: str, event: Mapping[str, Any]) -> Path:
        """Write an event without requiring a shared database or cross-process lock."""
        recorded_at_ns = time.time_ns()
        event_id = str(uuid.uuid4())
        event_record = {
            'schema_version': SCHEMA_VERSION,
            'event_id': event_id,
            'recorded_at': datetime.now(UTC).isoformat(),
            'recorded_at_ns': recorded_at_ns,
            'process_id': os.getpid(),
            **event,
        }
        event_dir = (
            self.root_path / f'agent={_safe_component(agent_name)}' / f'run={_safe_component(run_id)}' / 'events'
        )
        event_path = event_dir / f'{recorded_at_ns:020d}_{event_id}.json'
        content = orjson.dumps(event_record, option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE)
        _atomic_write(event_path, content)
        return event_path


def iter_trajectory_events(
    root_path: str | Path | None = None,
    *,
    agent_name: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate locally stored events in deterministic per-run order."""
    root = Path(root_path).expanduser().resolve() if root_path is not None else get_trajectory_root()
    agent_pattern = f'agent={_safe_component(agent_name)}' if agent_name else 'agent=*'
    for event_path in sorted(root.glob(f'{agent_pattern}/run=*/events/*.json')):
        yield orjson.loads(event_path.read_bytes())


def delete_trajectory_run(
    agent_name: str,
    run_id: str,
    root_path: str | Path | None = None,
) -> bool:
    """Delete one exact agent run and its immutable events."""
    root = Path(root_path).expanduser().resolve() if root_path is not None else get_trajectory_root()
    agent_directory = root / f'agent={_safe_component(agent_name)}'
    run_directory = agent_directory / f'run={_safe_component(run_id)}'
    if not run_directory.is_dir():
        return False

    shutil.rmtree(run_directory)
    with suppress(OSError):
        agent_directory.rmdir()
    return True
