"""Export a searchable tool-to-test-run-report lookup table."""

import csv
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from llm.utils.trajectory_store import iter_trajectory_events

TEST_RUN_AGENT = 'test_run'
TEST_RUN_REPORT_COLUMNS = ('run_id', 'report')
REPORT_TEMPLATE = """working_dir: {working_dir}
help_command: |-
{help_command}
running_command: |-
{running_command}
parameters:{parameters}
input_files:{input_files}
output_files:{output_files}
note:{note}
"""


@dataclass(frozen=True)
class TestRunReportExportResult:
    """Summary of one test-run report export."""

    output: Path
    event_count: int
    run_count: int
    source_report_count: int
    lookup_count: int
    resolved_reference_count: int
    missing_lookup_count: int


@dataclass(frozen=True)
class _ToolRecord:
    lookup_id: str
    run_id: str | None
    tool_uid: str


def _final_report(event: dict[str, Any]) -> Mapping[str, Any] | None:
    """Return a completed agent's structured report, if this event contains one."""
    if event.get('event_type') != 'agent_end':
        return None
    payload = event.get('payload')
    state = payload.get('state') if isinstance(payload, dict) else None
    report = state.get('test_run_report') if isinstance(state, dict) else None
    if report is None:
        return None
    if not isinstance(report, Mapping):
        msg = f'Invalid test-run report in trajectory event {event.get("event_id", "unknown")}: expected an object.'
        raise TypeError(msg)
    return report


def _quoted(value: Any) -> str:
    """Render a JSON string scalar, which is also valid YAML."""
    return json.dumps(value, ensure_ascii=False)


def _block(value: Any) -> str:
    if not isinstance(value, str):
        msg = f'Expected report text, got {type(value).__name__}.'
        raise TypeError(msg)
    return '\n'.join(f'  {line}' for line in value.splitlines() or [''])


def _object_list(value: Any, fields: tuple[str, ...], label: str) -> str:
    if not isinstance(value, list):
        msg = f'Expected {label} to be a list.'
        raise TypeError(msg)
    if not value:
        return ' []'

    lines: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            msg = f'Expected each {label} item to be an object.'
            raise TypeError(msg)
        for index, field in enumerate(fields):
            if field not in item:
                msg = f'Missing {field} in {label} item.'
                raise ValueError(msg)
            prefix = '  - ' if index == 0 else '    '
            lines.append(f'{prefix}{field}: {_quoted(item[field])}')
    return f'\n{"\n".join(lines)}'


def _optional_note(value: Any) -> str:
    if value is None:
        return ' null'
    return f' |-\n{_block(value)}'


def _report_text(report: Mapping[str, Any]) -> str:
    """Render a structured report with a stable, human-oriented field order."""
    required_fields = ('working_dir', 'help_command', 'running_command', 'parameters', 'input_files', 'output_files')
    missing_fields = [field for field in required_fields if field not in report]
    if missing_fields:
        msg = f'Missing required test-run report field(s): {", ".join(missing_fields)}.'
        raise ValueError(msg)
    return REPORT_TEMPLATE.format(
        working_dir=_quoted(report['working_dir']),
        help_command=_block(report['help_command']),
        running_command=_block(report['running_command']),
        parameters=_object_list(report['parameters'], ('parameter', 'description'), 'parameters'),
        input_files=_object_list(
            report['input_files'],
            ('name', 'path', 'file_type', 'description'),
            'input_files',
        ),
        output_files=_object_list(
            report['output_files'],
            ('name', 'path', 'file_type', 'description'),
            'output_files',
        ),
        note=_optional_note(report.get('note')),
    )


def _load_tool_records(db_path: Path) -> list[_ToolRecord]:
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            'SELECT "index", tool_name, thread_id, tool_uid FROM tool_functions '
            'WHERE tool_uid IS NOT NULL ORDER BY "index"'
        ).fetchall()
    finally:
        con.close()
    return [
        _ToolRecord(
            lookup_id=f'tool_{task_index}_{tool_name}',
            run_id=f'test_run:{thread_id}' if thread_id is not None else None,
            tool_uid=str(tool_uid),
        )
        for task_index, tool_name, thread_id, tool_uid in rows
    ]


def _lookup_rows(
    reports: Mapping[str, Mapping[str, Any]],
    records: list[_ToolRecord],
) -> tuple[list[dict[str, str]], int, int]:
    """Build one lookup row per tool record, resolving reused nodes through their tool UID."""
    source_report_by_tool: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for record in records:
        if record.run_id is None or record.run_id not in reports:
            continue
        source = (record.run_id, reports[record.run_id])
        previous_source = source_report_by_tool.setdefault(record.tool_uid, source)
        if previous_source[0] != record.run_id:
            msg = f'Tool {record.tool_uid} has multiple test-run report sources: {previous_source[0]}, {record.run_id}.'
            raise ValueError(msg)

    rows: list[dict[str, str]] = []
    resolved_reference_count = 0
    missing_lookup_count = 0
    for record in records:
        report = reports.get(record.run_id) if record.run_id is not None else None
        if report is None:
            source = source_report_by_tool.get(record.tool_uid)
            report = source[1] if source is not None else None
        if report is None:
            missing_lookup_count += 1
            continue
        if record.run_id not in reports:
            resolved_reference_count += 1
        rows.append({'run_id': record.lookup_id, 'report': _report_text(report)})
    return rows, resolved_reference_count, missing_lookup_count


def export_test_run_reports(
    trajectory_root: Path,
    db_path: Path,
    output_path: Path,
    *,
    show_progress: bool = False,
) -> TestRunReportExportResult:
    """Export tool lookup IDs mapped to their own or referenced structured test-run report."""
    trajectory_root = trajectory_root.expanduser().resolve()
    db_path = db_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not trajectory_root.is_dir():
        msg = f'Agent trajectory directory does not exist: {trajectory_root}'
        raise FileNotFoundError(msg)
    if not db_path.is_file():
        msg = f'Node-generation database does not exist: {db_path}'
        raise FileNotFoundError(msg)
    if db_path == output_path:
        msg = 'The CSV output path cannot be the source DuckDB file.'
        raise ValueError(msg)

    reports: dict[str, Mapping[str, Any]] = {}
    run_ids: set[str] = set()
    event_count = 0
    events = iter_trajectory_events(trajectory_root, agent_name=TEST_RUN_AGENT)
    for event in tqdm(
        events,
        desc='Exporting test-run reports',
        unit='event',
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        if not isinstance(event, dict) or event.get('agent') != TEST_RUN_AGENT:
            msg = 'Invalid test-run trajectory event.'
            raise ValueError(msg)
        run_id = event.get('run_id')
        if not isinstance(run_id, str) or not run_id:
            msg = f'Invalid run ID in test-run trajectory event {event.get("event_id", "unknown")}.'
            raise ValueError(msg)
        event_count += 1
        run_ids.add(run_id)
        report = _final_report(event)
        if report is not None:
            reports[run_id] = report

    if event_count == 0:
        msg = f'No test-run trajectory events found: {trajectory_root}'
        raise ValueError(msg)

    rows, resolved_reference_count, missing_lookup_count = _lookup_rows(
        reports,
        _load_tool_records(db_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f'.{output_path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temp_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=TEST_RUN_REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return TestRunReportExportResult(
        output=output_path,
        event_count=event_count,
        run_count=len(run_ids),
        source_report_count=len(reports),
        lookup_count=len(rows),
        resolved_reference_count=resolved_reference_count,
        missing_lookup_count=missing_lookup_count,
    )
