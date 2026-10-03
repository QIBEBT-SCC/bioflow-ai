"""Export a searchable tool-to-Markdown-test-run-report lookup table."""

import csv
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from bioflow_ai.llm.observability.trajectories import iter_trajectory_events

TEST_RUN_AGENT = 'test_run'
WRITE_REPORT_TOOL = 'write_tool_report'
TEST_RUN_REPORT_COLUMNS = ('run_id', 'with_validation', 'without_validation', 'first_pass')
ERROR_PREVIEW_LIMIT = 5


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
    first_pass_count: int
    validation_retry_count: int


@dataclass(frozen=True)
class _ToolRecord:
    lookup_id: str
    run_id: str | None
    tool_uid: str


@dataclass(frozen=True)
class _ReportAttempt:
    report: Any
    accepted: bool


@dataclass(frozen=True)
class _ReportVersions:
    with_validation: Mapping[str, Any]
    without_validation: Any
    first_pass: bool


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


def _report_attempt(event: dict[str, Any]) -> _ReportAttempt | None:
    """Return a submitted report and whether that exact submission passed validation."""
    if event.get('event_type') != 'tool_call':
        return None
    payload = event.get('payload')
    if not isinstance(payload, Mapping):
        return None
    request = payload.get('request')
    if not isinstance(request, Mapping) or request.get('name') != WRITE_REPORT_TOOL:
        return None

    args = request.get('args')
    report = args['report'] if isinstance(args, Mapping) and 'report' in args else args

    response = payload.get('response')
    update = response.get('update') if isinstance(response, Mapping) else None
    accepted = isinstance(update, Mapping) and 'test_run_report' in update
    return _ReportAttempt(report=report, accepted=accepted)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        msg = f'Expected {label} to be text, got {type(value).__name__}.'
        raise TypeError(msg)
    return value


def _objects(value: Any, fields: tuple[str, ...], label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        msg = f'Expected {label} to be a list.'
        raise TypeError(msg)

    items: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            msg = f'Expected each {label} item to be an object.'
            raise TypeError(msg)
        missing_fields = [field for field in fields if field not in item]
        if missing_fields:
            msg = f'Missing {", ".join(missing_fields)} in {label} item.'
            raise ValueError(msg)
        items.append(item)
    return items


def _parameter_lines(value: Any) -> str:
    items = _objects(value, ('parameter', 'description'), 'parameters')
    if not items:
        return '- None'
    return '\n'.join(
        f'- `{_text(item["parameter"], "parameters parameter")}`: '
        f'{_text(item["description"], "parameters description")}'
        for item in items
    )


def _output_file_lines(value: Any) -> str:
    items = _objects(value, ('path', 'file_type', 'description'), 'output_files')
    if not items:
        return '- None'
    return '\n'.join(
        f'- `{_text(item["path"], "output_files path")}`: '
        f'[{_text(item["file_type"], "output_files file_type")}] '
        f'{_text(item["description"], "output_files description")}'
        for item in items
    )


def _report_text(report: Mapping[str, Any]) -> str:
    """Render a structured report as compact Markdown for evaluation and display."""
    required_fields = ('working_dir', 'help_command', 'running_command', 'parameters', 'input_files', 'output_files')
    missing_fields = [field for field in required_fields if field not in report]
    if missing_fields:
        msg = f'Missing required test-run report field(s): {", ".join(missing_fields)}.'
        raise ValueError(msg)

    working_dir = _text(report['working_dir'], 'working_dir')
    _text(report['help_command'], 'help_command')
    running_command = _text(report['running_command'], 'running_command')
    parameters = _parameter_lines(report['parameters'])
    _objects(report['input_files'], ('name', 'path', 'file_type', 'description'), 'input_files')
    output_files = _output_file_lines(report['output_files'])
    sections = [
        '## Test Run Report',
        f'working dir: {working_dir}',
        f'```bash\n{running_command}\n```',
        f'**Command Explain**\n\n{parameters}',
        f'**Output Files**\n\n{output_files}',
    ]
    note = report.get('note')
    if note is not None:
        sections.append(f'**Note**\n{_text(note, "note")}')
    return '\n\n'.join(sections)


def _first_report_text(report: Any) -> str:
    """Render a first submission, retaining malformed tool arguments as Markdown-wrapped JSON."""
    if isinstance(report, Mapping):
        try:
            return _report_text(report)
        except (TypeError, ValueError):
            pass
    raw_report = json.dumps(report, ensure_ascii=False, indent=2)
    return f'## Test Run Report\n\n**Raw first submission (invalid report structure)**\n\n```json\n{raw_report}\n```'


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
    reports: Mapping[str, _ReportVersions],
    records: list[_ToolRecord],
) -> tuple[list[dict[str, Any]], int, int]:
    """Build one lookup row per tool record, resolving reused nodes through their tool UID."""
    source_report_by_tool: dict[str, tuple[str, _ReportVersions]] = {}
    for record in records:
        if record.run_id is None or record.run_id not in reports:
            continue
        source = (record.run_id, reports[record.run_id])
        previous_source = source_report_by_tool.setdefault(record.tool_uid, source)
        if previous_source[0] != record.run_id:
            msg = f'Tool {record.tool_uid} has multiple test-run report sources: {previous_source[0]}, {record.run_id}.'
            raise ValueError(msg)

    rows: list[dict[str, Any]] = []
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
        rows.append(
            {
                'run_id': record.lookup_id,
                'with_validation': _report_text(report.with_validation),
                'without_validation': _first_report_text(report.without_validation),
                'first_pass': report.first_pass,
            }
        )
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

    final_reports: dict[str, Mapping[str, Any]] = {}
    first_attempts: dict[str, _ReportAttempt] = {}
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
        attempt = _report_attempt(event)
        if attempt is not None:
            first_attempts.setdefault(run_id, attempt)
        report = _final_report(event)
        if report is not None:
            final_reports[run_id] = report

    if event_count == 0:
        msg = f'No test-run trajectory events found: {trajectory_root}'
        raise ValueError(msg)

    missing_first_attempts = sorted(set(final_reports) - set(first_attempts))
    if missing_first_attempts:
        preview = ', '.join(missing_first_attempts[:ERROR_PREVIEW_LIMIT])
        suffix = ' ...' if len(missing_first_attempts) > ERROR_PREVIEW_LIMIT else ''
        msg = f'Missing first {WRITE_REPORT_TOOL} attempt for completed test-run report(s): {preview}{suffix}'
        raise ValueError(msg)

    reports = {
        run_id: _ReportVersions(
            with_validation=report,
            without_validation=first_attempts[run_id].report,
            first_pass=first_attempts[run_id].accepted,
        )
        for run_id, report in final_reports.items()
    }

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
        first_pass_count=sum(report.first_pass for report in reports.values()),
        validation_retry_count=sum(not report.first_pass for report in reports.values()),
    )
