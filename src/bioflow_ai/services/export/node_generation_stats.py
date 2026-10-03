"""Offline CSV statistics for recorded tool-node generation runs."""

import csv
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bioflow_ai.llm.observability.trajectories import iter_trajectory_events

INTERRUPTED_STATUS = 'INTERRUPTED'
NODE_STATS_COLUMNS = (
    'task_index',
    'tool_name',
    'thread_id',
    'tool_uid',
    'result',
    'source_status',
    'dialogue_rounds',
    'total_tokens',
    'max_input_tokens',
    'test_run_rounds',
    'test_run_tokens',
    'tool_generator_rounds',
    'tool_generator_tokens',
    'unmetered_rounds',
    'trajectory_event_count',
    'started_at',
    'finished_at',
    'last_error',
)


@dataclass
class _AgentRunStats:
    rounds: int = 0
    total_tokens: int | None = None
    max_input_tokens: int | None = None
    unmetered_rounds: int = 0
    event_count: int = 0


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _model_call_usage(event: dict[str, Any]) -> tuple[int | None, int | None]:
    """Read billing usage from the saved model response, never from repeated state messages."""
    payload = event.get('payload')
    response = payload.get('response') if isinstance(payload, dict) else None
    messages = response.get('messages') if isinstance(response, dict) else None
    if not isinstance(messages, list):
        return None, None

    inputs: list[int] = []
    totals: list[int] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        data = message.get('data')
        usage = data.get('usage_metadata') if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            continue
        input_tokens = _token_count(usage.get('input_tokens'))
        output_tokens = _token_count(usage.get('output_tokens'))
        total_tokens = _token_count(usage.get('total_tokens'))
        if input_tokens is not None:
            inputs.append(input_tokens)
        if total_tokens is not None:
            totals.append(total_tokens)
        elif input_tokens is not None and output_tokens is not None:
            totals.append(input_tokens + output_tokens)

    return (max(inputs) if inputs else None, sum(totals) if totals else None)


def _agent_run_stats(trajectory_root: Path, agent_name: str, thread_id: str) -> _AgentRunStats:
    run_id = f'{agent_name}:{thread_id}'
    stats = _AgentRunStats()
    for event in iter_trajectory_events(trajectory_root, agent_name=agent_name, run_id=run_id):
        if not isinstance(event, dict) or event.get('agent') != agent_name or event.get('run_id') != run_id:
            msg = f'Invalid {agent_name} trajectory event for {run_id}'
            raise ValueError(msg)
        stats.event_count += 1
        if event.get('event_type') != 'model_call':
            continue
        stats.rounds += 1
        input_tokens, total_tokens = _model_call_usage(event)
        if input_tokens is not None:
            stats.max_input_tokens = max(stats.max_input_tokens or 0, input_tokens)
        if total_tokens is None:
            stats.unmetered_rounds += 1
        else:
            stats.total_tokens = (stats.total_tokens or 0) + total_tokens
    return stats


def _load_records(db_path: Path) -> list[dict[str, Any]]:
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cursor = con.execute(
            'SELECT "index" AS task_index, tool_name, thread_id, tool_uid, status, started_at, finished_at, '
            'last_error FROM tool_functions WHERE thread_id IS NOT NULL ORDER BY "index"'
        )
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
    finally:
        con.close()


def export_node_stats(db_path: Path, trajectory_root: Path, output_path: Path, *, show_progress: bool = False) -> int:
    """Export current node-generation attempts and saved test-run/configuration model usage to CSV."""
    db_path = db_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if db_path == output_path:
        msg = 'The CSV output path cannot be the source DuckDB file.'
        raise ValueError(msg)
    if not db_path.is_file():
        msg = f'Node-generation database does not exist: {db_path}'
        raise FileNotFoundError(msg)

    records = _load_records(db_path)
    rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc='Exporting node stats', unit='run', dynamic_ncols=True, disable=not show_progress):
        thread_id = str(record['thread_id'])
        test_run = _agent_run_stats(trajectory_root, 'test_run', thread_id)
        generator = _agent_run_stats(trajectory_root, 'tool_generator', thread_id)
        measured = [value for value in (test_run.total_tokens, generator.total_tokens) if value is not None]
        max_inputs = [value for value in (test_run.max_input_tokens, generator.max_input_tokens) if value is not None]
        result = (
            'success' if record['tool_uid'] else 'interrupted' if record['status'] == INTERRUPTED_STATUS else 'error'
        )
        rows.append(
            {
                **record,
                'tool_uid': record['tool_uid'] or '',
                'result': result,
                'source_status': record['status'],
                'dialogue_rounds': test_run.rounds + generator.rounds,
                'total_tokens': sum(measured) if measured else None,
                'max_input_tokens': max(max_inputs) if max_inputs else None,
                'test_run_rounds': test_run.rounds,
                'test_run_tokens': test_run.total_tokens,
                'tool_generator_rounds': generator.rounds,
                'tool_generator_tokens': generator.total_tokens,
                'unmetered_rounds': test_run.unmetered_rounds + generator.unmetered_rounds,
                'trajectory_event_count': test_run.event_count + generator.event_count,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f'.{output_path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temp_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=NODE_STATS_COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return len(rows)
