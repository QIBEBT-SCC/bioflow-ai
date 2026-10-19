from pathlib import Path

from pydantic import BaseModel, ConfigDict

from bioflow_ai.services.export import PREBUILD_BUCKET_ID

IMAGES_OUTPUT = Path('presets/images.parquet')
DOCS_OUTPUT = Path('presets/tool_docs.parquet')
TOOLS_OUTPUT = Path('presets/tools.parquet')


class DatabaseExportResult(BaseModel):
    """Summary of one exported preset database dataset."""

    model_config = ConfigDict(frozen=True)

    name: str
    output: Path
    record_count: int
    remote_path: str | None


def _database_uri() -> str:
    """Return a ConnectorX-compatible URL for the configured database."""
    from bioflow_ai.core.config import get_settings

    db_url = get_settings().persistent.database.DB_URL
    return db_url.unicode_string().replace(db_url.scheme, db_url.scheme.split('+')[0])


def _export_dataset(
    name: str,
    query: str,
    output: Path,
    *,
    upload: bool,
) -> DatabaseExportResult:
    """Export one PostgreSQL query to a sorted Parquet file and optionally upload it."""
    import polars as pl

    data = pl.read_database_uri(
        query,
        uri=_database_uri(),
        engine='connectorx',
    ).sort('uid')
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output, compression='snappy')

    remote_path = None
    if upload:
        from huggingface_hub import batch_bucket_files

        batch_bucket_files(
            PREBUILD_BUCKET_ID,
            add=[(output, output.name)],
        )
        remote_path = f'{PREBUILD_BUCKET_ID}/{output.name}'

    return DatabaseExportResult(
        name=name,
        output=output,
        record_count=data.height,
        remote_path=remote_path,
    )


def export_images(*, upload: bool = False) -> DatabaseExportResult:
    """Export the tool-image preset dataset."""
    return _export_dataset(
        'images',
        'SELECT * FROM images',
        IMAGES_OUTPUT,
        upload=upload,
    )


def export_docs(*, upload: bool = False) -> DatabaseExportResult:
    """Export the tool-documentation preset dataset."""
    return _export_dataset(
        'docs',
        'SELECT * FROM tool_docs',
        DOCS_OUTPUT,
        upload=upload,
    )


def export_tools(*, upload: bool = False) -> DatabaseExportResult:
    """Export the tool-definition preset dataset."""
    return _export_dataset(
        'tools',
        """
        SELECT t.*,
               COALESCE((
                   SELECT jsonb_agg(jsonb_build_object('id', tt.id, 'name', tt.name) ORDER BY tt.id)
                   FROM tool_tag_link ttl
                   JOIN tool_tags tt ON tt.id = ttl.tag_id
                   WHERE ttl.tool_uid = t.uid
               ), '[]'::jsonb)::text AS tags
        FROM tools t
        """,
        TOOLS_OUTPUT,
        upload=upload,
    )
