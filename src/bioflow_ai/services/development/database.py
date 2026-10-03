from typing import Any

DB_PATH = 'presets/build/develop.db'
TRAJECTORY_DATASET_DIRECTORY = 'agent_datasets/trajectories'
TRAJECTORY_INSERT_BATCH_SIZE = 500
TOOL_STATUS_PENDING = 'PENDING'
TOOL_STATUS_RUNNING = 'RUNNING'
TOOL_STATUS_INTERRUPTED = 'INTERRUPTED'
TOOL_STATUS_FAILED = 'FAILED'
TOOL_STATUS_COMPLETED = 'COMPLETED'
TOOL_INCOMPLETE_STATUSES = (TOOL_STATUS_RUNNING, TOOL_STATUS_INTERRUPTED, TOOL_STATUS_FAILED)
TOOL_TRAJECTORY_AGENTS = ('test_run', 'tool_generator')


def row_to_dict(con: Any, query: str, params: list | None = None) -> dict[str, Any] | None:
    """Read one development database row by column name."""
    cur = con.execute(query, params or [])
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], row, strict=False))


def rows_to_dicts(con: Any, query: str, params: list | None = None) -> list[dict[str, Any]]:
    """Read development database rows by column name."""
    cur = con.execute(query, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
