import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path as FilePath
from typing import Any, cast

import orjson
from loguru import logger

from bioflow_ai.services.development.database import TRAJECTORY_DATASET_DIRECTORY, TRAJECTORY_INSERT_BATCH_SIZE


def default_trajectory_dataset_path() -> str:
    """Return the configured persistent destination for compacted trajectories."""
    from bioflow_ai.core.config import get_settings

    storage_root = FilePath(get_settings().persistent.storage.STORAGE_PATH).expanduser().resolve()
    return str(storage_root / TRAJECTORY_DATASET_DIRECTORY)


def _load_trajectory_event(event_file: FilePath) -> dict[str, Any]:
    event = orjson.loads(event_file.read_bytes())
    if not isinstance(event, dict):
        msg = 'Top-level JSON value must be an object.'
        raise TypeError(msg)
    return event


def _trajectory_event_row(event: dict[str, Any], source_file: FilePath) -> tuple[Any, ...]:
    """Convert one trajectory event to the stable offline dataset schema."""
    required_fields = ('schema_version', 'event_id', 'agent', 'run_id', 'event_type', 'recorded_at_ns')
    missing_fields = [field for field in required_fields if event.get(field) is None]
    if missing_fields:
        msg = f'Missing required field(s): {", ".join(missing_fields)}'
        raise ValueError(msg)

    return (
        event['schema_version'],
        event['event_id'],
        event['agent'],
        event['run_id'],
        event.get('thread_id'),
        event.get('user_id'),
        event['event_type'],
        event.get('started_at'),
        event.get('ended_at'),
        event.get('recorded_at'),
        event['recorded_at_ns'],
        event.get('process_id'),
        event.get('duration_ms'),
        orjson.dumps(event.get('payload')).decode(),
        orjson.dumps(event.get('error')).decode() if event.get('error') is not None else None,
        str(source_file),
    )


def compact_trajectories(
    input_dir: str,
    output_dir: str,
    agents: tuple[str, ...],
    *,
    skip_invalid: bool,
    overwrite: bool,
) -> None:
    """Validate immutable trajectory JSON events and compact them into Parquet datasets."""
    import duckdb

    source_root = FilePath(input_dir).resolve()
    destination = FilePath(output_dir).resolve()
    if not source_root.is_dir():
        msg = f'Trajectory directory does not exist: {source_root}'
        raise ValueError(msg)
    if destination.exists() and not overwrite:
        msg = f'Output directory already exists: {destination}. Use --overwrite to replace it.'
        raise ValueError(msg)

    event_files = sorted(source_root.glob('agent=*/run=*/events/*.json'))
    if not event_files:
        msg = f'No trajectory events found: {source_root}'
        raise ValueError(msg)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_output = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp')
    temp_output.mkdir()
    con = duckdb.connect()
    invalid_files: list[dict[str, str]] = []
    selected_count = 0

    try:
        con.execute("""
            CREATE TABLE trajectory_events (
                schema_version INTEGER,
                event_id VARCHAR PRIMARY KEY,
                agent VARCHAR NOT NULL,
                run_id VARCHAR NOT NULL,
                thread_id VARCHAR,
                user_id BIGINT,
                event_type VARCHAR NOT NULL,
                started_at TIMESTAMPTZ,
                ended_at TIMESTAMPTZ,
                recorded_at TIMESTAMPTZ,
                recorded_at_ns UBIGINT NOT NULL,
                process_id BIGINT,
                duration_ms DOUBLE,
                payload JSON,
                error JSON,
                source_file VARCHAR NOT NULL
            )
        """)
        insert_sql = 'INSERT INTO trajectory_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
        batch: list[tuple[Any, ...]] = []
        for event_file in event_files:
            try:
                event = _load_trajectory_event(event_file)
                if agents and event.get('agent') not in agents:
                    continue
                batch.append(_trajectory_event_row(event, event_file.relative_to(source_root)))
                selected_count += 1
                if len(batch) >= TRAJECTORY_INSERT_BATCH_SIZE:
                    con.executemany(insert_sql, batch)
                    batch.clear()
            except Exception as error:
                invalid_files.append({'file': str(event_file), 'error': str(error)})
                if not skip_invalid:
                    msg = f'Invalid trajectory file {event_file}: {error}'
                    raise ValueError(msg) from error
        if batch:
            con.executemany(insert_sql, batch)

        if selected_count == 0:
            selected_agents = ', '.join(agents) if agents else '(all)'
            msg = f'No events match the agent filter: {selected_agents}'
            _raise_export_error(msg)

        events_path = temp_output / 'events.parquet'
        runs_path = temp_output / 'runs.parquet'
        con.execute(
            """
            COPY (
                SELECT *
                FROM trajectory_events
                ORDER BY agent, run_id, recorded_at_ns, event_id
            ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(events_path)],
        )
        con.execute(
            """
            COPY (
                SELECT
                    agent,
                    run_id,
                    any_value(thread_id) AS thread_id,
                    any_value(user_id) AS user_id,
                    min(recorded_at) AS started_at,
                    max(recorded_at) AS ended_at,
                    CASE
                        WHEN arg_max(event_type, recorded_at_ns) = 'agent_end' THEN 'completed'
                        WHEN count(*) FILTER (WHERE error IS NOT NULL) > 0 THEN 'failed'
                        ELSE 'incomplete'
                    END AS status,
                    count(*) AS event_count,
                    count(*) FILTER (WHERE event_type = 'agent_start') AS agent_start_count,
                    count(*) FILTER (WHERE event_type = 'model_call') AS model_call_count,
                    count(*) FILTER (WHERE event_type = 'tool_call') AS tool_call_count,
                    count(*) FILTER (WHERE event_type = 'agent_end') AS agent_end_count,
                    count(*) FILTER (WHERE error IS NOT NULL) AS error_count,
                    min(schema_version) AS min_schema_version,
                    max(schema_version) AS max_schema_version
                FROM trajectory_events
                GROUP BY agent, run_id
                ORDER BY agent, run_id
            ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(runs_path)],
        )

        run_count_row = con.execute('SELECT count(DISTINCT run_id) FROM trajectory_events').fetchone()
        run_count = cast('tuple[int]', run_count_row)[0]
        agent_counts = {
            row[0]: row[1]
            for row in con.execute(
                'SELECT agent, count(DISTINCT run_id) FROM trajectory_events GROUP BY agent ORDER BY agent'
            ).fetchall()
        }
        metadata = {
            'schema_version': 1,
            'generated_at': datetime.now(UTC).isoformat(),
            'source_root': str(source_root),
            'agent_filter': list(agents),
            'event_count': selected_count,
            'run_count': run_count,
            'run_count_by_agent': agent_counts,
            'invalid_file_count': len(invalid_files),
            'invalid_files': invalid_files,
        }
        (temp_output / 'metadata.json').write_bytes(
            orjson.dumps(metadata, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
        )
    except Exception:
        shutil.rmtree(temp_output, ignore_errors=True)
        raise
    finally:
        con.close()

    backup = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.backup')
    try:
        if destination.exists():
            destination.replace(backup)
        temp_output.replace(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)

    logger.success(
        f'Trajectory compaction complete: {selected_count} events, {run_count} runs, '
        f'{len(invalid_files)} invalid files; output directory: {destination}'
    )


def _raise_export_error(message: str) -> None:
    raise ValueError(message)
