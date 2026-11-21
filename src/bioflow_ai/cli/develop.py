import asyncio
from pathlib import Path as FilePath
from typing import cast

import click
from loguru import logger

from bioflow_ai.cli.development_render import render_statistics, render_tools, render_updated_requirement
from bioflow_ai.cli.errors import handle_service_errors
from bioflow_ai.cli.render import create_iterator_renderer, create_progress_renderer
from bioflow_ai.services.development.database import DB_PATH
from bioflow_ai.services.development.records import search_tools, statistic, update_tool_requirement
from bioflow_ai.services.export.trajectories import compact_trajectories, default_trajectory_dataset_path
from bioflow_ai.utils.observability.execution_logging import agent_logging_to_console


@click.group()
def cli() -> None:
    """Developer tools."""
    from bioflow_ai.infrastructure.logging import init_logging

    init_logging()


@cli.command()
@click.option('--name', '-n', type=str, default=None, help='Tool name to search for.')
@handle_service_errors
def search_cmd(name: str) -> None:
    """Search for tools by name."""
    render_tools(search_tools(tool_name=name))


@cli.command(name='update-tool-requirement')
@click.argument('index', type=int)
@click.argument('requirement', type=str)
@handle_service_errors
def update_tool_requirement_cmd(index: int, requirement: str) -> None:
    """Update the original build requirement for one tool record."""
    if not requirement.strip():
        message = 'Requirement must not be empty.'
        raise click.BadParameter(message, param_hint='requirement')
    previous, updated = update_tool_requirement(index, requirement)
    render_updated_requirement(index, previous, updated)


@cli.command()
@click.option('--index', type=int, default=None, help='Data index to process (default: all incomplete records).')
@click.option('--name', '-n', help='Tool name; switches to manual creation mode when provided.')
@click.option('--help-command', '-c', help='Command used to retrieve tool help (required in manual mode).')
@click.option('--function', '-f', help='Description of the tool function to generate (required in manual mode).')
@click.option('--thread-id', '-t', help='Checkpoint thread ID (required in manual mode).')
@click.option(
    '--drop-old', is_flag=True, default=False, help='Delete the old checkpoint for this thread before running.'
)
@click.option(
    '--langsmith-tracing', is_flag=True, default=False, help='Enable LangSmith tracing (disabled by default).'
)
@handle_service_errors
@agent_logging_to_console(enabled=False)
def create_tools_cmd(
    index: int | None,
    name: str | None,
    help_command: str | None,
    function: str | None,
    thread_id: str | None,
    *,
    drop_old: bool,
    langsmith_tracing: bool,
) -> None:
    """Create tools automatically, or create one manually with --name."""
    from bioflow_ai.services.development.tool_generation import create_tool_manually, init_tools

    manual_values = {
        '--name': name,
        '--help-command': help_command,
        '--function': function,
        '--thread-id': thread_id,
    }
    if any(value is not None for value in manual_values.values()):
        missing = [option for option, value in manual_values.items() if value is None]
        if missing:
            message = f'Missing options for manual creation mode: {", ".join(missing)}'
            raise click.UsageError(message)
        if index is not None:
            message = '--index cannot be used with manual creation options'
            raise click.UsageError(message)
        # The completeness check above narrows these values at runtime.
        asyncio.run(
            create_tool_manually(
                cast('str', name),
                cast('str', help_command),
                cast('str', function),
                cast('str', thread_id),
                drop_old=drop_old,
                langsmith_tracing=langsmith_tracing,
                on_event=create_progress_renderer(),
            )
        )
        return

    if drop_old:
        message = '--drop-old can only be used in manual creation mode'
        raise click.UsageError(message)
    asyncio.run(init_tools(index, langsmith_tracing=langsmith_tracing, on_event=create_progress_renderer()))


@cli.command()
@click.argument('index', type=int)
@handle_service_errors
@agent_logging_to_console(enabled=False)
def read_interrupt_cmd(index: int) -> None:
    """Read tool interruption details."""
    from bioflow_ai.services.development.tool_generation import read_interrupt

    asyncio.run(read_interrupt(index, on_event=create_progress_renderer()))


@cli.command()
@click.argument('index', type=int)
@click.option('--message', '-m', type=str, default=None, help='Message to send when resuming the run.')
@click.option('--approve', is_flag=True, default=False, help='Approve the interrupted action.')
@click.option(
    '--langsmith-tracing', is_flag=True, default=False, help='Enable LangSmith tracing (disabled by default).'
)
@handle_service_errors
@agent_logging_to_console(enabled=False)
def continue_run_cmd(index: int, message: str | None, *, approve: bool, langsmith_tracing: bool) -> None:
    """Resume the tool creation task at the specified index."""
    from bioflow_ai.services.development.tool_generation import continue_run

    asyncio.run(
        continue_run(
            index, message, approve=approve, langsmith_tracing=langsmith_tracing, on_event=create_progress_renderer()
        )
    )


@cli.command()
@click.argument('indices', type=int, nargs=-1, required=True)
@handle_service_errors
def delete_prebuild_tool_cmd(indices: tuple[int, ...]) -> None:
    """Delete prebuilt tools by index (multiple indices are supported, e.g. 61 62 63 65)."""
    from bioflow_ai.services.development.tool_generation import delete_prebuild_tool

    asyncio.run(delete_prebuild_tool(list(indices)))


@cli.command()
@click.option('--upload', is_flag=True, default=False, help='Upload the exported data to Hugging Face.')
@click.option(
    '--db',
    type=click.Choice(['images', 'docs', 'tools', 'tool_memories']),
    multiple=True,
    default=['images', 'docs', 'tools', 'tool_memories'],
    show_default=True,
    help='Select one or more datasets to export.',
)
@handle_service_errors
def export(*, upload: bool, db: tuple[str, ...]) -> None:
    """Export database data."""
    from bioflow_ai.services.export.database import export_docs, export_images, export_tool_memories, export_tools

    db_map = {
        'images': export_images,
        'docs': export_docs,
        'tools': export_tools,
        'tool_memories': export_tool_memories,
    }
    for name in db:
        logger.info(f'Exporting {name}...')
        result = db_map[name](upload=upload)
        logger.success('Exported {} records to {}', result.record_count, result.output)
        if result.remote_path is not None:
            logger.success('Uploaded {} to {}', result.name, result.remote_path)


@cli.command(name='export-mcp')
@click.option('--upload', is_flag=True, default=False, help='Upload tools.duckdb to the Hugging Face data bucket.')
@click.option(
    '--output',
    required=True,
    type=click.Path(dir_okay=False, path_type=FilePath),
    help='Destination DuckDB file for the standalone MCP server.',
)
@handle_service_errors
def export_mcp_cmd(output: FilePath, *, upload: bool) -> None:
    """Export the local MCP tool knowledge database."""
    from bioflow_ai.services.export.mcp import export_mcp_database

    try:
        result = export_mcp_database(output, upload=upload)
    except Exception as error:
        raise click.ClickException(str(error)) from error

    logger.success('Exported MCP database to {} ({} tools)', result.output, result.tool_count)
    if result.remote_path is not None:
        logger.success('Uploaded MCP database to {}', result.remote_path)


@cli.command()
@handle_service_errors
def statistic_cmd() -> None:
    """Show node generation statistics."""
    render_statistics(statistic())


@cli.command(name='export-node-stats')
@click.option(
    '--db-path',
    type=click.Path(dir_okay=False, path_type=FilePath),
    default=DB_PATH,
    show_default=True,
    help='DuckDB file containing tool_functions run records.',
)
@click.option(
    '--input-dir',
    type=click.Path(file_okay=False, path_type=FilePath),
    default=None,
    help='Saved agent trajectories (default: persistent.storage.storage_path/agent_trajectories).',
)
@click.option(
    '--output',
    type=click.Path(dir_okay=False, path_type=FilePath),
    default=FilePath('presets/build/node_generation_stats.csv'),
    show_default=True,
    help='Destination CSV file.',
)
@handle_service_errors
def export_node_stats_cmd(db_path: FilePath, input_dir: FilePath | None, output: FilePath) -> None:
    """Export one CSV row per locally recorded node-generation run.

    Token counts cover saved test_run and tool_generator model calls. Missing usage is blank.
    """
    from bioflow_ai.services.export.node_generation_stats import export_node_stats
    from bioflow_ai.utils.observability.trajectories import get_trajectory_root

    trajectory_root = input_dir or get_trajectory_root()
    try:
        count = export_node_stats(
            db_path,
            trajectory_root,
            output,
            progress=create_iterator_renderer(description='Exporting node stats', unit='run', dynamic_ncols=True),
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    logger.success('Exported {} node-generation runs to {}', count, output.expanduser().resolve())


@cli.command(name='export-test-run-reports')
@click.option(
    '--db-path',
    type=click.Path(dir_okay=False, path_type=FilePath),
    default=DB_PATH,
    show_default=True,
    help='DuckDB file containing tool_functions run-to-tool references.',
)
@click.option(
    '--input-dir',
    type=click.Path(file_okay=False, path_type=FilePath),
    default=None,
    help='Saved agent trajectories (default: persistent.storage.storage_path/agent_trajectories).',
)
@click.option(
    '--output',
    type=click.Path(dir_okay=False, path_type=FilePath),
    default=FilePath('presets/build/test_run_reports.csv'),
    show_default=True,
    help='Destination CSV containing first-submission and validated reports plus first-pass flags.',
)
@handle_service_errors
def export_test_run_reports_cmd(db_path: FilePath, input_dir: FilePath | None, output: FilePath) -> None:
    """Export first-submission and validated report text for recorded tool entries."""
    from bioflow_ai.services.export.test_run_reports import export_test_run_reports
    from bioflow_ai.utils.observability.trajectories import get_trajectory_root

    trajectory_root = input_dir or get_trajectory_root()
    try:
        result = export_test_run_reports(
            trajectory_root,
            db_path,
            output,
            progress=create_iterator_renderer(
                description='Exporting test-run reports', unit='event', dynamic_ncols=True
            ),
        )
    except (OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    logger.success(
        'Exported {} report lookups from {} saved reports ({} trajectory runs, {} events) to {}',
        result.lookup_count,
        result.source_report_count,
        result.run_count,
        result.event_count,
        result.output,
    )
    logger.info(
        '{} of {} source reports passed validation on the first submission; {} passed after validation feedback.',
        result.first_pass_count,
        result.source_report_count,
        result.validation_retry_count,
    )
    logger.info(
        'Resolved {} no-test-run references; skipped {} tool records without a recorded source report.',
        result.resolved_reference_count,
        result.missing_lookup_count,
    )


@cli.command(name='compact-trajectories')
@click.option(
    '--input-dir',
    type=click.Path(exists=True, file_okay=False, path_type=str),
    default=None,
    help='Immutable JSON trajectory root (default: persistent.storage.storage_path/agent_trajectories).',
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help='Parquet output directory (default: persistent.storage.storage_path/agent_datasets/trajectories).',
)
@click.option('--agent', 'agents', multiple=True, help='Include only the specified agent; may be repeated.')
@click.option('--skip-invalid', is_flag=True, help='Skip invalid JSON files and record them in metadata.json.')
@click.option('--overwrite', is_flag=True, help='Atomically replace an existing output directory.')
@handle_service_errors
def compact_trajectories_cmd(
    input_dir: str | None,
    output_dir: str | None,
    agents: tuple[str, ...],
    *,
    skip_invalid: bool,
    overwrite: bool,
) -> None:
    """Compact raw agent JSON events into events/runs Parquet datasets."""
    from bioflow_ai.utils.observability.trajectories import get_trajectory_root

    resolved_input = input_dir or str(get_trajectory_root())
    resolved_output = output_dir or default_trajectory_dataset_path()
    compact_trajectories(resolved_input, resolved_output, agents, skip_invalid=skip_invalid, overwrite=overwrite)


if __name__ == '__main__':
    cli()
