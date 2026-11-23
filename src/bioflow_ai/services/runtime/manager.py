"""Manage background BioFlowAI runtime processes (API server and Celery workers)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import tomllib
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from bioflow_ai.core.config import CeleryConfig, NetworkConfig

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable, Iterator, Sequence

ServiceName = Literal['api', 'execution-worker', 'agent-worker']
ALL_SERVICES: tuple[ServiceName, ...] = ('api', 'execution-worker', 'agent-worker')

RUN_DIR_REL = Path('.bioflow') / 'run'
SERVICE_LOG_DIR_REL = Path('log') / 'services'
DEFAULT_START_TIMEOUT = 45.0
DEFAULT_STOP_TIMEOUT = 15.0
DEFAULT_POLL_INTERVAL = 0.25
MAX_SERVICE_LOG_BYTES = 10 * 1024 * 1024

_FAILURE_LOG_TAIL_LINES = 25
_PROC_STAT_STARTTIME_INDEX = 19
_PROC_STAT_PGRP_INDEX = 2
_SIGKILL_GRACE_SECONDS = 2.0
_UNREADY_TERM_GRACE_SECONDS = 3.0
_HEALTH_PROBE_TIMEOUT_SECONDS = 1.0
_WILDCARD_HOSTS = frozenset({'', '0.0.0.0', '::'})  # noqa: S104


class ServiceProcessState(BaseModel):
    """Persisted metadata for a managed background process."""

    model_config = ConfigDict(frozen=True)

    service: ServiceName
    pid: int
    pgid: int
    started_at: str
    process_start_ticks: int | None = None
    command: list[str]
    cwd: str
    log_file: str
    log_requests: bool = False


class ServiceStatusInfo(BaseModel):
    """Current runtime status for a BioFlowAI service."""

    model_config = ConfigDict(frozen=True)

    service: ServiceName
    status: Literal['running', 'stopped', 'exited']
    pid: int | None = None
    pgid: int | None = None
    started_at: str | None = None
    uptime_seconds: float | None = None
    log_file: Path
    log_requests: bool = False
    healthy: bool | None = None
    detail: str | None = None


class ServiceOperationResult(BaseModel):
    """Outcome of a start, stop, or restart operation on one service."""

    model_config = ConfigDict(frozen=True)

    service: ServiceName
    action: Literal[
        'started',
        'already_running',
        'stopped',
        'force_killed',
        'not_running',
        'cleaned_stale',
        'failed',
    ]
    success: bool
    pid: int | None = None
    log_file: Path
    message: str
    log_tail: list[str] = Field(default_factory=list)


class _RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str
    port: int
    execution_concurrency: int
    agent_concurrency: int


def _resolve_base_dir(base_dir: Path | None) -> Path:
    return (base_dir or Path.cwd()).resolve()


def _run_dir(base_dir: Path) -> Path:
    return base_dir / RUN_DIR_REL


def _state_file(base_dir: Path, service: ServiceName) -> Path:
    return _run_dir(base_dir) / f'{service}.json'


def _ready_file(base_dir: Path, service: ServiceName) -> Path:
    return _run_dir(base_dir) / f'{service}.ready'


def service_log_path(service: ServiceName, *, base_dir: Path | None = None) -> Path:
    """Return the service log path within the deployment directory."""
    root = _resolve_base_dir(base_dir)
    return root / SERVICE_LOG_DIR_REL / f'{service}.log'


def resolve_services(services: Sequence[ServiceName] | None) -> list[ServiceName]:
    """Normalize an optional service list, defaulting to all services in startup order."""
    if not services:
        return list(ALL_SERVICES)
    ordered: list[ServiceName] = []
    seen: set[ServiceName] = set()
    for service in services:
        if service not in seen:
            seen.add(service)
            ordered.append(service)
    return ordered


def _require_runtime_config(base_dir: Path) -> _RuntimeConfig:
    config_path = base_dir / 'config.toml'
    if not config_path.is_file():
        message = f'Configuration file "{config_path}" does not exist. Run `uv run bioflow-ai auto-init` first.'
        raise ValueError(message)

    try:
        with config_path.open('rb') as file:
            raw = tomllib.load(file)
        network = NetworkConfig.model_validate(raw.get('network', {}))
        celery = CeleryConfig.model_validate(raw.get('celery', {}))
    except (OSError, ValueError) as exc:
        message = f'Failed to read configuration from "{config_path}": {exc}'
        raise ValueError(message) from exc

    return _RuntimeConfig(
        host=network.HOST,
        port=network.PORT,
        execution_concurrency=celery.EXECUTION_CONCURRENCY,
        agent_concurrency=celery.AGENT_CONCURRENCY,
    )


def _load_optional_runtime_config(base_dir: Path) -> _RuntimeConfig | None:
    try:
        return _require_runtime_config(base_dir)
    except ValueError:
        return None


def _reap_child_nohang(pid: int) -> None:
    with suppress(ChildProcessError, OSError):
        os.waitpid(pid, os.WNOHANG)


def _read_proc_stat_fields(pid: int) -> list[str] | None:
    stat_path = Path('/proc') / str(pid) / 'stat'
    try:
        content = stat_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    rparen = content.rfind(')')
    if rparen == -1:
        return None
    fields = content[rparen + 2 :].split()
    if len(fields) <= _PROC_STAT_STARTTIME_INDEX:
        return None
    return fields


def _read_proc_stat(pid: int) -> tuple[str, int] | None:
    fields = _read_proc_stat_fields(pid)
    if fields is None:
        return None
    try:
        return fields[0], int(fields[_PROC_STAT_STARTTIME_INDEX])
    except ValueError:
        return None


def _is_managed_process_alive(state: ServiceProcessState) -> bool:
    _reap_child_nohang(state.pid)
    proc_root = Path('/proc')
    if proc_root.is_dir():
        stat_info = _read_proc_stat(state.pid)
        if stat_info is None:
            return False
        proc_state, starttime = stat_info
        if proc_state == 'Z':
            return False
        return state.process_start_ticks is None or starttime == state.process_start_ticks

    try:
        os.kill(state.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _can_signal_process_group(state: ServiceProcessState) -> bool:
    return state.pgid == state.pid and state.pgid > 1 and state.pgid != os.getpgrp()


def _has_live_pgrp_member(state: ServiceProcessState) -> bool:
    if not _can_signal_process_group(state):
        return False
    try:
        os.killpg(state.pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError:
        return False

    proc_root = Path('/proc')
    if not proc_root.is_dir():
        return True

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        _reap_child_nohang(pid)
        fields = _read_proc_stat_fields(pid)
        if fields is None or fields[0] == 'Z':
            continue
        try:
            pgrp = int(fields[_PROC_STAT_PGRP_INDEX])
        except ValueError:
            continue
        if pgrp == state.pgid:
            return True
    return False


def _is_service_completely_stopped(state: ServiceProcessState) -> bool:
    return not _is_managed_process_alive(state) and not _has_live_pgrp_member(state)


def _signal_process(pid: int, sig: signal.Signals) -> None:
    with suppress(ProcessLookupError, OSError):
        os.kill(pid, sig)


def _signal_service_group(state: ServiceProcessState, sig: signal.Signals) -> None:
    if _can_signal_process_group(state):
        with suppress(ProcessLookupError, OSError):
            os.killpg(state.pgid, sig)
            return
    _signal_process(state.pid, sig)


def _write_state_file(path: Path, state: ServiceProcessState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp_path.write_text(state.model_dump_json(indent=2), encoding='utf-8')
    tmp_path.replace(path)


def _read_state_file(path: Path) -> ServiceProcessState | None:
    if not path.is_file():
        return None
    try:
        return ServiceProcessState.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _cleanup_runtime_files(base_dir: Path, service: ServiceName) -> None:
    _state_file(base_dir, service).unlink(missing_ok=True)
    _ready_file(base_dir, service).unlink(missing_ok=True)


def _rotate_log_if_needed(log_path: Path) -> None:
    if not log_path.is_file():
        return
    try:
        size = log_path.stat().st_size
    except OSError:
        return
    if size >= MAX_SERVICE_LOG_BYTES:
        backup_path = log_path.with_name(f'{log_path.name}.1')
        with suppress(OSError):
            log_path.replace(backup_path)


def _default_command(service: ServiceName, *, log_requests: bool) -> list[str]:
    command = [sys.executable, '-m', 'bioflow_ai.runtime.entrypoint', service]
    if service == 'api' and log_requests:
        command.append('--log')
    return command


def _probe_host(host: str) -> str:
    return '127.0.0.1' if host in _WILDCARD_HOSTS else host


def _check_api_health(host: str, port: int, *, timeout: float = _HEALTH_PROBE_TIMEOUT_SECONDS) -> bool:
    url = f'http://{_probe_host(host)}:{port}/health'
    try:
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            response = client.get(url)
    except httpx.HTTPError:
        return False
    return response.status_code == HTTPStatus.OK


def _is_worker_ready(ready_file: Path, pid: int) -> bool:
    if not ready_file.is_file():
        return False
    try:
        return ready_file.read_text(encoding='utf-8').strip() == str(pid)
    except OSError:
        return False


def _terminate_unready_process(state: ServiceProcessState, *, poll_interval: float) -> None:
    _signal_process(state.pid, signal.SIGTERM)
    deadline = time.monotonic() + _UNREADY_TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        if _is_service_completely_stopped(state):
            return
        time.sleep(poll_interval)
    _signal_service_group(state, signal.SIGKILL)
    kill_deadline = time.monotonic() + _SIGKILL_GRACE_SECONDS
    while time.monotonic() < kill_deadline and not _is_service_completely_stopped(state):
        time.sleep(poll_interval)


def read_log_tail(log_file: Path, lines: int = 50) -> list[str]:
    """Return the last N lines of a log file."""
    if lines <= 0 or not log_file.is_file():
        return []
    tail: deque[str] = deque(maxlen=lines)
    try:
        with log_file.open('r', encoding='utf-8', errors='replace') as file:
            for line in file:
                tail.append(line.rstrip('\r\n'))
    except OSError:
        return []
    return list(tail)


def iter_follow_logs(
    log_files: dict[ServiceName, Path],
    *,
    lines: int = 50,
    poll_interval: float = 0.2,
    stop_event: threading.Event | None = None,
) -> Iterator[tuple[ServiceName, str]]:
    """Yield trailing log lines and continuously stream newly appended lines."""
    tracked: dict[ServiceName, tuple[int | None, int]] = {}

    for service, path in log_files.items():
        if path.is_file():
            try:
                with path.open('r', encoding='utf-8', errors='replace') as file:
                    tail = deque((line.rstrip('\r\n') for line in file), maxlen=lines)
                    inode = os.fstat(file.fileno()).st_ino
                    offset = file.tell()
                for line in tail:
                    yield service, line
                tracked[service] = (inode, offset)
                continue
            except OSError:
                pass
        tracked[service] = (None, 0)

    while stop_event is None or not stop_event.is_set():
        emitted = False
        for service, path in log_files.items():
            if not path.is_file():
                tracked[service] = (None, 0)
                continue

            tracked_inode, offset = tracked[service]
            try:
                stat_result = path.stat()
            except OSError:
                continue

            if tracked_inode is None or stat_result.st_ino != tracked_inode or stat_result.st_size < offset:
                offset = 0

            if stat_result.st_size > offset:
                try:
                    with path.open('r', encoding='utf-8', errors='replace') as file:
                        current_inode = os.fstat(file.fileno()).st_ino
                        if tracked_inode is not None and current_inode != tracked_inode:
                            offset = 0
                        file.seek(offset)
                        for line in file:
                            emitted = True
                            yield service, line.rstrip('\r\n')
                        tracked[service] = (current_inode, file.tell())
                except OSError:
                    continue
            else:
                tracked[service] = (stat_result.st_ino, offset)

        if not emitted:
            if stop_event is not None:
                stop_event.wait(poll_interval)
            else:
                time.sleep(poll_interval)


def start_service(
    service: ServiceName,
    *,
    base_dir: Path | None = None,
    log_requests: bool = False,
    ready_timeout: float = DEFAULT_START_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    command_factory: Callable[[ServiceName, bool], list[str]] | None = None,
    api_host: str | None = None,
    api_port: int | None = None,
) -> ServiceOperationResult:
    """Start a single service in the background and wait until it is ready."""
    root = _resolve_base_dir(base_dir)
    config = _require_runtime_config(root)

    log_path = service_log_path(service, base_dir=root)
    state_path = _state_file(root, service)
    ready_path = _ready_file(root, service)

    existing = _read_state_file(state_path)
    if existing is not None and _is_managed_process_alive(existing):
        return ServiceOperationResult(
            service=service,
            action='already_running',
            success=True,
            pid=existing.pid,
            log_file=Path(existing.log_file),
            message=f'{service} is already running (PID {existing.pid}).',
        )

    _cleanup_runtime_files(root, service)
    _run_dir(root).mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_log_if_needed(log_path)

    effective_log_requests = bool(log_requests and service == 'api')
    command = (
        command_factory(service, effective_log_requests)
        if command_factory is not None
        else _default_command(service, log_requests=effective_log_requests)
    )

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    if service != 'api':
        env['BIOFLOW_READY_FILE'] = str(ready_path)
    else:
        env.pop('BIOFLOW_READY_FILE', None)

    with log_path.open('ab') as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = pid

    stat_info = _read_proc_stat(pid)
    state = ServiceProcessState(
        service=service,
        pid=pid,
        pgid=pgid,
        started_at=datetime.now(UTC).isoformat(),
        process_start_ticks=stat_info[1] if stat_info is not None else None,
        command=command,
        cwd=str(root),
        log_file=str(log_path),
        log_requests=effective_log_requests,
    )
    _write_state_file(state_path, state)

    probe_host = config.host if api_host is None else api_host
    probe_port = config.port if api_port is None else api_port

    deadline = time.monotonic() + ready_timeout
    while True:
        exit_code = proc.poll()
        if exit_code is not None:
            _cleanup_runtime_files(root, service)
            return ServiceOperationResult(
                service=service,
                action='failed',
                success=False,
                log_file=log_path,
                message=f'{service} exited unexpectedly during startup (exit code {exit_code}).',
                log_tail=read_log_tail(log_path, lines=_FAILURE_LOG_TAIL_LINES),
            )

        is_ready = _check_api_health(probe_host, probe_port) if service == 'api' else _is_worker_ready(ready_path, pid)
        if is_ready:
            detail = (
                f'listening on http://{_probe_host(probe_host)}:{probe_port}'
                if service == 'api'
                else (
                    f'concurrency {config.execution_concurrency}'
                    if service == 'execution-worker'
                    else f'concurrency {config.agent_concurrency}'
                )
            )
            return ServiceOperationResult(
                service=service,
                action='started',
                success=True,
                pid=pid,
                log_file=log_path,
                message=f'{service} started ({detail}).',
            )

        if time.monotonic() >= deadline:
            _terminate_unready_process(state, poll_interval=poll_interval)
            _cleanup_runtime_files(root, service)
            return ServiceOperationResult(
                service=service,
                action='failed',
                success=False,
                log_file=log_path,
                message=f'Timed out waiting for {service} to become ready after {ready_timeout:.0f}s.',
                log_tail=read_log_tail(log_path, lines=_FAILURE_LOG_TAIL_LINES),
            )

        time.sleep(poll_interval)


def start_services(
    services: Sequence[ServiceName] | None = None,
    *,
    base_dir: Path | None = None,
    log_requests: bool = False,
    timeout: float = DEFAULT_START_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    command_factory: Callable[[ServiceName, bool], list[str]] | None = None,
    api_host: str | None = None,
    api_port: int | None = None,
    on_progress: Callable[[ServiceName, str], None] | None = None,
) -> list[ServiceOperationResult]:
    """Start one or more BioFlowAI services in the background."""
    root = _resolve_base_dir(base_dir)
    _require_runtime_config(root)

    targets = resolve_services(services)
    results: list[ServiceOperationResult] = []
    for service in targets:
        if on_progress is not None:
            on_progress(service, 'starting')
        result = start_service(
            service,
            base_dir=root,
            log_requests=log_requests,
            ready_timeout=timeout,
            poll_interval=poll_interval,
            command_factory=command_factory,
            api_host=api_host,
            api_port=api_port,
        )
        results.append(result)
        if on_progress is not None:
            on_progress(service, result.action)
    return results


def stop_services(
    services: Sequence[ServiceName] | None = None,
    *,
    base_dir: Path | None = None,
    timeout: float = DEFAULT_STOP_TIMEOUT,
    force: bool = False,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> list[ServiceOperationResult]:
    """Stop one or more managed BioFlowAI services gracefully (or forcefully)."""
    root = _resolve_base_dir(base_dir)
    targets = resolve_services(services)
    results_by_service: dict[ServiceName, ServiceOperationResult] = {}
    active: dict[ServiceName, ServiceProcessState] = {}

    for service in targets:
        state_path = _state_file(root, service)
        default_log = service_log_path(service, base_dir=root)
        state = _read_state_file(state_path)
        if state is None:
            _cleanup_runtime_files(root, service)
            results_by_service[service] = ServiceOperationResult(
                service=service,
                action='not_running',
                success=True,
                log_file=default_log,
                message=f'{service} is not running.',
            )
            continue

        if not _is_managed_process_alive(state):
            _cleanup_runtime_files(root, service)
            results_by_service[service] = ServiceOperationResult(
                service=service,
                action='cleaned_stale',
                success=True,
                pid=state.pid,
                log_file=Path(state.log_file),
                message=f'Cleaned stale state for {service} (PID {state.pid}).',
            )
            continue

        active[service] = state
        if force:
            _signal_service_group(state, signal.SIGKILL)
        else:
            _signal_process(state.pid, signal.SIGTERM)

    if active:
        wait_seconds = _SIGKILL_GRACE_SECONDS if force else timeout
        deadline = time.monotonic() + max(0.0, wait_seconds)
        pending = dict(active)

        while pending:
            finished = [svc for svc, st in pending.items() if _is_service_completely_stopped(st)]
            for svc in finished:
                st = pending.pop(svc)
                _cleanup_runtime_files(root, svc)
                action = 'force_killed' if force else 'stopped'
                msg = f'{svc} killed (PID {st.pid}).' if force else f'{svc} stopped (PID {st.pid}).'
                results_by_service[svc] = ServiceOperationResult(
                    service=svc,
                    action=action,
                    success=True,
                    pid=st.pid,
                    log_file=Path(st.log_file),
                    message=msg,
                )

            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(poll_interval)

        if pending and not force:
            for st in pending.values():
                _signal_service_group(st, signal.SIGKILL)

            kill_deadline = time.monotonic() + _SIGKILL_GRACE_SECONDS
            while pending:
                finished = [svc for svc, st in pending.items() if _is_service_completely_stopped(st)]
                for svc in finished:
                    st = pending.pop(svc)
                    _cleanup_runtime_files(root, svc)
                    results_by_service[svc] = ServiceOperationResult(
                        service=svc,
                        action='force_killed',
                        success=True,
                        pid=st.pid,
                        log_file=Path(st.log_file),
                        message=f'{svc} did not exit within {timeout:.0f}s; force killed (PID {st.pid}).',
                    )
                if not pending or time.monotonic() >= kill_deadline:
                    break
                time.sleep(poll_interval)

        for svc, st in pending.items():
            results_by_service[svc] = ServiceOperationResult(
                service=svc,
                action='failed',
                success=False,
                pid=st.pid,
                log_file=Path(st.log_file),
                message=f'Failed to stop {svc} (PID {st.pid}).',
            )

    return [results_by_service[service] for service in targets]


def restart_services(
    services: Sequence[ServiceName] | None = None,
    *,
    base_dir: Path | None = None,
    log_requests: bool | None = None,
    stop_timeout: float = DEFAULT_STOP_TIMEOUT,
    start_timeout: float = DEFAULT_START_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    command_factory: Callable[[ServiceName, bool], list[str]] | None = None,
    api_host: str | None = None,
    api_port: int | None = None,
    on_progress: Callable[[ServiceName, str], None] | None = None,
) -> list[ServiceOperationResult]:
    """Restart one or more BioFlowAI services in the background."""
    root = _resolve_base_dir(base_dir)
    _require_runtime_config(root)

    targets = resolve_services(services)
    if log_requests is None:
        api_state = _read_state_file(_state_file(root, 'api')) if 'api' in targets else None
        effective_log_requests = api_state.log_requests if api_state is not None else False
    else:
        effective_log_requests = log_requests

    stop_results = stop_services(
        targets,
        base_dir=root,
        timeout=stop_timeout,
        poll_interval=poll_interval,
    )

    stop_failures = {res.service: res for res in stop_results if not res.success}
    to_start = [service for service in targets if service not in stop_failures]
    started_results = (
        start_services(
            to_start,
            base_dir=root,
            log_requests=effective_log_requests,
            timeout=start_timeout,
            poll_interval=poll_interval,
            command_factory=command_factory,
            api_host=api_host,
            api_port=api_port,
            on_progress=on_progress,
        )
        if to_start
        else []
    )
    started_by_service = {res.service: res for res in started_results}

    return [stop_failures[service] if service in stop_failures else started_by_service[service] for service in targets]


def _service_detail(service: ServiceName, config: _RuntimeConfig | None) -> str | None:
    if config is None:
        return None
    if service == 'api':
        return f'port {config.port}'
    if service == 'execution-worker':
        return f'concurrency {config.execution_concurrency}'
    return f'concurrency {config.agent_concurrency}'


def _calculate_uptime_seconds(started_at: str) -> float | None:
    try:
        started_dt = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started_dt.tzinfo is None:
        started_dt = started_dt.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - started_dt).total_seconds())


def get_service_status(
    service: ServiceName,
    *,
    base_dir: Path | None = None,
    check_health: bool = False,
) -> ServiceStatusInfo:
    """Inspect the runtime status of a single BioFlowAI service."""
    root = _resolve_base_dir(base_dir)
    config = _load_optional_runtime_config(root)
    return _build_service_status(service, root=root, config=config, check_health=check_health)


def _build_service_status(
    service: ServiceName,
    *,
    root: Path,
    config: _RuntimeConfig | None,
    check_health: bool,
) -> ServiceStatusInfo:
    state = _read_state_file(_state_file(root, service))
    detail = _service_detail(service, config)
    default_log = service_log_path(service, base_dir=root)

    if state is None:
        return ServiceStatusInfo(
            service=service,
            status='stopped',
            log_file=default_log,
            detail=detail,
        )

    alive = _is_managed_process_alive(state)
    if alive:
        healthy: bool | None = None
        if check_health and service == 'api' and config is not None:
            healthy = _check_api_health(config.host, config.port)
        return ServiceStatusInfo(
            service=service,
            status='running',
            pid=state.pid,
            pgid=state.pgid,
            started_at=state.started_at,
            uptime_seconds=_calculate_uptime_seconds(state.started_at),
            log_file=Path(state.log_file),
            log_requests=state.log_requests,
            healthy=healthy,
            detail=detail,
        )

    return ServiceStatusInfo(
        service=service,
        status='exited',
        pid=state.pid,
        pgid=state.pgid,
        started_at=state.started_at,
        uptime_seconds=None,
        log_file=Path(state.log_file),
        log_requests=state.log_requests,
        healthy=False if (check_health and service == 'api') else None,
        detail=detail,
    )


def get_all_service_statuses(
    services: Sequence[ServiceName] | None = None,
    *,
    base_dir: Path | None = None,
    check_health: bool = False,
) -> list[ServiceStatusInfo]:
    """Inspect the runtime status of all or selected BioFlowAI services."""
    root = _resolve_base_dir(base_dir)
    config = _load_optional_runtime_config(root)
    targets = resolve_services(services)
    return [_build_service_status(service, root=root, config=config, check_health=check_health) for service in targets]


def has_running_services(*, base_dir: Path | None = None) -> bool:
    """Return whether any managed BioFlowAI backend service is currently running."""
    return any(status.status == 'running' for status in get_all_service_statuses(base_dir=base_dir))
