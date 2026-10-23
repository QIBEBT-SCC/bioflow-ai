import base64
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from bioflow_ai.services.export import PREBUILD_BUCKET_ID

SCHEMA_VERSION = 1
TOOL_ID_LENGTH = 22
BUCKET_PATH = 'tools.duckdb'


class MCPExportResult(BaseModel):
    """Summary of a completed MCP catalog export."""

    model_config = ConfigDict(frozen=True)

    output: Path
    tool_count: int
    remote_path: str | None


def _normalize_search_text(value: str) -> str:
    """Normalize public catalog text for deterministic local search."""
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def _compact_tool_id(value: uuid.UUID) -> str:
    """Encode a UUID as a stable, URL-safe 22-character public identifier."""
    return base64.urlsafe_b64encode(value.bytes).decode().rstrip('=')


def _collect_snapshot() -> list[dict[str, Any]]:
    """Read a consistent PostgreSQL snapshot for the standalone MCP catalog."""
    from sqlalchemy import select

    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.tools import ToolDefine, ToolHelpDoc, ToolImage
    from bioflow_ai.schemas.tools import ToolDefinitionImport

    with (
        engine.connect().execution_options(isolation_level='REPEATABLE READ') as connection,
        connection.begin(),
        Session(bind=connection) as session,
    ):
        images = list(session.scalars(select(ToolImage).order_by(ToolImage.uid)))
        docs = list(session.scalars(select(ToolHelpDoc).order_by(ToolHelpDoc.uid)))
        tools = list(session.scalars(select(ToolDefine).order_by(ToolDefine.uid)))

        image_by_uid = {image.uid: image for image in images}
        doc_by_uid = {doc.uid: doc for doc in docs}
        tool_rows: list[dict[str, Any]] = []
        for tool in tools:
            validated = ToolDefinitionImport.model_validate(
                {
                    'uid': tool.uid,
                    'name': tool.name,
                    'image_uid': tool.image_uid,
                    'description': tool.description,
                    'tool_type': tool.tool_type,
                    'group_id': tool.group_id,
                    'help_doc_uid': tool.help_doc_uid,
                    'command_template': tool.command_template,
                    'dynamic_params': tool.dynamic_params,
                    'immutable_static_params': tool.immutable_static_params,
                    'modifiable_static_params': tool.modifiable_static_params,
                    'file_mounts': tool.file_mounts,
                }
            )
            if validated.uid is None:
                msg = f'Tool has no UUID: {validated.name}'
                raise ValueError(msg)

            image = image_by_uid.get(validated.image_uid) if validated.image_uid is not None else None
            if validated.image_uid is not None and image is None:
                msg = f'Tool {validated.uid} references missing image {validated.image_uid}.'
                raise ValueError(msg)

            doc = doc_by_uid.get(validated.help_doc_uid) if validated.help_doc_uid is not None else None
            if validated.help_doc_uid is not None and doc is None:
                msg = f'Tool {validated.uid} references missing documentation {validated.help_doc_uid}.'
                raise ValueError(msg)

            aliases = [image.name, *image.aliases] if image is not None else []
            normalized_aliases = [_normalize_search_text(alias) for alias in aliases]
            normalized_name = _normalize_search_text(validated.name)
            search_text = ' '.join(
                [
                    normalized_name,
                    *normalized_aliases,
                    _normalize_search_text(validated.description),
                ]
            )
            mounts = [value.model_dump(mode='json') for value in validated.file_mounts]
            tool_rows.append(
                {
                    'schema_version': SCHEMA_VERSION,
                    'id': _compact_tool_id(validated.uid),
                    'name': validated.name,
                    'description': validated.description,
                    'aliases': aliases,
                    'image': image.image.get_image() if image is not None and image.image is not None else None,
                    'complete_command': tool.complete_command,
                    'parameters': [value.model_dump(mode='json') for value in validated.dynamic_params],
                    'inputs': [mount for mount in mounts if mount['file_type'] == 'INPUT'],
                    'outputs': [mount for mount in mounts if mount['file_type'] == 'OUTPUT'],
                    'documentation': doc.content if doc is not None else None,
                    'normalized_name': normalized_name,
                    'normalized_aliases': normalized_aliases,
                    'search_text': search_text,
                }
            )

    return tool_rows


def _write_database(output: Path, tools: list[dict[str, Any]]) -> None:
    """Write a catalog snapshot to a DuckDB database file."""
    import duckdb

    connection = duckdb.connect(str(output))
    try:
        connection.execute(
            """
            CREATE TABLE tools (
                schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                id VARCHAR PRIMARY KEY CHECK (length(id) = 22),
                name VARCHAR NOT NULL,
                description VARCHAR NOT NULL,
                aliases VARCHAR[] NOT NULL,
                image VARCHAR,
                complete_command VARCHAR NOT NULL,
                parameters JSON NOT NULL,
                inputs JSON NOT NULL,
                outputs JSON NOT NULL,
                documentation VARCHAR,
                normalized_name VARCHAR NOT NULL,
                normalized_aliases VARCHAR[] NOT NULL,
                search_text VARCHAR NOT NULL
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO tools VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?::JSON, ?::JSON, ?::JSON, ?, ?, ?, ?
            )
            """,
            [
                (
                    row['schema_version'],
                    row['id'],
                    row['name'],
                    row['description'],
                    row['aliases'],
                    row['image'],
                    row['complete_command'],
                    orjson.dumps(row['parameters']).decode(),
                    orjson.dumps(row['inputs']).decode(),
                    orjson.dumps(row['outputs']).decode(),
                    row['documentation'],
                    row['normalized_name'],
                    row['normalized_aliases'],
                    row['search_text'],
                )
                for row in tools
            ],
        )
        connection.execute('CHECKPOINT')
    finally:
        connection.close()


def _validate_database(output: Path, tools: list[dict[str, Any]]) -> None:
    """Verify the generated database through a fresh read-only connection."""
    import duckdb

    connection = duckdb.connect(str(output), read_only=True)
    try:
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        if tables != [('tools',)]:
            msg = 'Generated MCP database must contain only the tools table.'
            raise ValueError(msg)

        summary = connection.execute(
            """
            SELECT count(*), count(DISTINCT id), min(schema_version), max(schema_version),
                   count(*) FILTER (WHERE NOT regexp_full_match(id, '[A-Za-z0-9_-]{22}'))
            FROM tools
            """
        ).fetchone()
        expected = (len(tools), len(tools), SCHEMA_VERSION, SCHEMA_VERSION, 0)
        if summary != expected:
            msg = 'Generated MCP database count, schema version, or public tool identifiers are invalid.'
            raise ValueError(msg)
    finally:
        connection.close()


def export_mcp_database(output: Path, *, upload: bool = False) -> MCPExportResult:
    """Export the standalone BioFlowAI tool catalog used by the local MCP server."""
    destination = output.expanduser().absolute()
    if destination.is_symlink():
        destination.unlink()
    elif destination.is_dir():
        msg = f'Output path is a directory: {destination}'
        raise ValueError(msg)
    elif destination.exists():
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)

    tools = _collect_snapshot()
    if not tools:
        msg = 'Cannot export an empty MCP tool catalog.'
        raise ValueError(msg)
    try:
        _write_database(destination, tools)
        _validate_database(destination, tools)
    except Exception:
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        raise

    remote_path = None
    if upload:
        from huggingface_hub import batch_bucket_files

        batch_bucket_files(
            PREBUILD_BUCKET_ID,
            add=[(destination, BUCKET_PATH)],
        )
        remote_path = f'{PREBUILD_BUCKET_ID}/{BUCKET_PATH}'

    return MCPExportResult(
        output=destination,
        tool_count=len(tools),
        remote_path=remote_path,
    )
