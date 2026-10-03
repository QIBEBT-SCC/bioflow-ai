import asyncio

import click

from bioflow_ai.cli.render import create_progress_renderer
from bioflow_ai.services.evaluation.workflow import (
    EvaluationUsageError,
    WorkflowResumeRequiredError,
    WorkflowResumeUnavailableError,
    wf_eval,
)


@click.group()
def cli() -> None:
    """Command for agent evaluate."""
    from bioflow_ai.infrastructure.logging import init_logging

    init_logging()


@cli.command()
@click.argument('task_file', type=click.Path(exists=True))
@click.option('--paper', '-p', 'paper_file', default=None, type=click.Path(exists=True), help='PDF paper file path.')
@click.option('--thread-id', '-t', default=None, help='Checkpoint thread ID.')
@click.option('--db-index', '-d', default=None, type=click.IntRange(min=1), help='Deprecated; ignored.')
@click.option('--drop-old', is_flag=True, default=False, help='Deprecated alias for --clean-checkpoint.')
@click.option(
    '--clean-checkpoint',
    is_flag=True,
    default=False,
    help='Drop the workflow builder checkpoint and remove workflows imported into the evaluate project.',
)
@click.option(
    '--clean-all',
    is_flag=True,
    default=False,
    help='Clear plans, samples, workflows, runs, and project files while retaining the existing project ID.',
)
@click.option(
    '--langsmith-tracing', is_flag=True, default=False, help='Enable LangSmith tracing (disabled by default).'
)
@click.option(
    '--resume',
    type=str,
    default=None,
    metavar='MESSAGE',
    help='Continue the workflow builder checkpoint and respond to any pending interrupt with MESSAGE.',
)
def wf_cmd(
    task_file: str,
    paper_file: str | None,
    thread_id: str | None,
    db_index: int | None,
    drop_old: bool,  # noqa: FBT001
    clean_checkpoint: bool,  # noqa: FBT001
    clean_all: bool,  # noqa: FBT001
    langsmith_tracing: bool,  # noqa: FBT001
    resume: str | None,
) -> None:
    """Run the workflow design evaluation on TASK_FILE."""
    try:
        asyncio.run(
            wf_eval(
                task_file,
                paper_file,
                thread_id,
                drop_old=drop_old,
                clean_checkpoint=clean_checkpoint,
                clean_all=clean_all,
                resume=resume,
                db_index=db_index,
                langsmith_tracing=langsmith_tracing,
                on_event=create_progress_renderer(),
            )
        )
    except EvaluationUsageError as exc:
        raise click.UsageError(str(exc)) from exc
    except (WorkflowResumeRequiredError, WorkflowResumeUnavailableError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == '__main__':
    cli()
