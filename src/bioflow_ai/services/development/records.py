from typing import Any

from bioflow_ai.services.development.database import (
    DB_PATH,
    TOOL_STATUS_COMPLETED,
    TOOL_STATUS_FAILED,
    TOOL_STATUS_INTERRUPTED,
    TOOL_STATUS_PENDING,
    TOOL_STATUS_RUNNING,
    row_to_dict,
)


def search_tools(tool_name: str) -> list[dict[str, Any]]:
    """搜索数据库中指定名称的工具原始数据."""
    import duckdb

    con = duckdb.connect(DB_PATH)
    result = con.execute('SELECT * FROM tool_functions WHERE tool_name LIKE ?', [f'%{tool_name}%']).pl()

    rows = result.to_dicts()
    con.close()
    return rows


def update_tool_requirement(index: int, requirement: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update a tool build requirement and show the record before and after the change."""
    import duckdb

    if not requirement.strip():
        msg = 'Requirement must not be empty.'
        raise ValueError(msg)

    con = duckdb.connect(DB_PATH)
    try:
        previous = row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
        if previous is None:
            msg = f'No data found for index {index}'
            raise ValueError(msg)

        con.execute(
            """UPDATE tool_functions
               SET function_description = ?, updated_at = current_timestamp
               WHERE index = ?""",
            [requirement, index],
        )
        updated = row_to_dict(con, 'SELECT * FROM tool_functions WHERE index = ?', [index])
    finally:
        con.close()

    if updated is None:
        msg = f'Failed to reload the updated record for index {index}'
        raise ValueError(msg)

    return previous, updated


def statistic() -> dict[str, Any]:
    """统计生成情况."""
    import duckdb

    con = duckdb.connect(DB_PATH, read_only=True)

    total = (con.execute('SELECT COUNT(*) FROM tool_functions').fetchone() or (0,))[0]
    pending = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_PENDING]).fetchone() or (0,)
    )[0]
    processed = total - pending
    success = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_COMPLETED]).fetchone() or (0,)
    )[0]
    unique_nodes = (
        con.execute('SELECT COUNT(DISTINCT tool_uid) FROM tool_functions WHERE tool_uid IS NOT NULL').fetchone() or (0,)
    )[0]
    failed = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_FAILED]).fetchone() or (0,)
    )[0]
    interrupted = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_INTERRUPTED]).fetchone()
        or (0,)
    )[0]
    running = (
        con.execute('SELECT COUNT(*) FROM tool_functions WHERE status = ?', [TOOL_STATUS_RUNNING]).fetchone() or (0,)
    )[0]

    latest = row_to_dict(
        con,
        'SELECT * FROM tool_functions WHERE status <> ? ORDER BY started_at DESC NULLS LAST LIMIT 1',
        [TOOL_STATUS_PENDING],
    )

    latest_interrupt = row_to_dict(
        con,
        'SELECT * FROM tool_functions WHERE status = ? ORDER BY finished_at DESC NULLS LAST LIMIT 1',
        [TOOL_STATUS_INTERRUPTED],
    )

    con.close()
    return {
        'total': total,
        'pending': pending,
        'processed': processed,
        'success': success,
        'unique_nodes': unique_nodes,
        'failed': failed,
        'interrupted': interrupted,
        'running': running,
        'latest': latest,
        'latest_interrupt': latest_interrupt,
    }
