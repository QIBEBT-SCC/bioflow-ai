"""Export final structured test-run reports as YAML text in a compact CSV table."""

import csv
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from llm.utils.trajectory_store import iter_trajectory_events

TEST_RUN_AGENT = 'test_run'
TEST_RUN_REPORT_COLUMNS = ('run_id', 'report')


@dataclass(frozen=True)
class TestRunReportExportResult:
    """Summary of one test-run report export."""

    output: Path
    event_count: int
    run_count: int
    report_count: int


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


def _report_yaml(report: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(report), allow_unicode=True, sort_keys=False, default_flow_style=False)


def export_test_run_reports(
    trajectory_root: Path,
    output_path: Path,
    *,
    show_progress: bool = False,
) -> TestRunReportExportResult:
    """Export the last completed structured report saved for each test-run trajectory."""
    trajectory_root = trajectory_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not trajectory_root.is_dir():
        msg = f'Agent trajectory directory does not exist: {trajectory_root}'
        raise FileNotFoundError(msg)

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

    rows = ({'run_id': run_id, 'report': _report_yaml(reports[run_id])} for run_id in sorted(reports))
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
        report_count=len(reports),
    )
