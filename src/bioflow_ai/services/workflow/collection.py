"""Versioned file collections used by the durable foreach coordinator."""

import hashlib
import json
import re
from pathlib import Path

from bioflow_ai.schemas.react_flow import TaskInput
from bioflow_ai.schemas.workflow_config import ToolOutput
from bioflow_ai.services.workflow.execution_context import get_node_output

COLLECTION_VERSION = 1


def freeze_file_collection(*, folder_value: str, pattern: str, output_dir: str) -> Path:
    """Persist one deterministic glob snapshot for a directory."""
    folder = Path(folder_value).resolve(strict=True)
    if not folder.is_dir():
        message = f'Not a directory: {folder}'
        raise ValueError(message)
    if not isinstance(pattern, str) or not pattern.strip():
        message = 'A file glob pattern is required'
        raise ValueError(message)
    pattern = pattern.strip()
    candidate = Path(pattern)
    if candidate.is_absolute() or '..' in candidate.parts or '\\' in pattern:
        message = f'Pattern must stay inside the input directory: {pattern}'
        raise ValueError(message)
    matches: dict[str, Path] = {}
    for path in folder.glob(pattern):
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_relative_to(folder) or not resolved.is_file():
            continue
        relative = path.relative_to(folder).as_posix()
        matches[relative] = resolved
    if not matches:
        message = f'No files match {pattern!r} in {folder}'
        raise ValueError(message)
    items = [
        {
            'key': hashlib.sha256(relative.encode()).hexdigest()[:24],
            'name': Path(relative).name,
            'relative_path': relative,
            'path': str(matches[relative]),
        }
        for relative in sorted(matches)
    ]
    target = Path(output_dir).absolute()
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / 'collection.json'
    manifest.write_text(
        json.dumps({'version': COLLECTION_VERSION, 'items': items}, ensure_ascii=False), encoding='utf-8'
    )
    return manifest


def collect_file_collection(*, inputs: dict[str, TaskInput], pattern: str, output_dir: str) -> ToolOutput:
    """Freeze a deterministic set of regular files beneath one upstream folder."""
    source = inputs.get('folder_path')
    if source is None:
        message = 'collect_file_collection requires folder_path'
        raise ValueError(message)
    folder_value, sample_name = get_node_output(source.source_uid, source.source_k)
    manifest = freeze_file_collection(folder_value=folder_value, pattern=pattern, output_dir=output_dir)
    return ToolOutput(result={'collection': manifest}, sample_name=sample_name)


def read_collection(path: str) -> list[dict[str, str]]:
    """Validate a frozen manifest before creating child runs."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if raw.get('version') != COLLECTION_VERSION or not isinstance(raw.get('items'), list):
        message = 'Unsupported file collection manifest'
        raise ValueError(message)
    seen: set[str] = set()
    items = raw['items']
    for item in items:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ('key', 'name', 'path')):
            message = 'Invalid collection item'
            raise ValueError(message)
        if re.fullmatch(r'[0-9a-f]{24}', item['key']) is None:
            message = 'Invalid collection key'
            raise ValueError(message)
        if item['key'] in seen:
            message = f'Duplicate collection key: {item["key"]}'
            raise ValueError(message)
        seen.add(item['key'])
    if not items:
        message = 'Empty file collection'
        raise ValueError(message)
    return items
