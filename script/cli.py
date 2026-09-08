import click

from script.develop import cli as dev_cli
from script.evaluate import cli as eval_cli
from script.init import cli as init_cli


@click.group()
def cli() -> None:
    """BioFlowAI management CLI."""


@cli.command()
@click.option('--log', 'enable_request_log', is_flag=True, help='Enable request body logging middleware.')
def api(*, enable_request_log: bool) -> None:
    """Start the FastAPI API server."""
    from app.utils.logger import init_logging
    from script.server import main

    init_logging()
    main(log_requests=enable_request_log)


@cli.command()
def worker() -> None:
    """Start the Celery worker."""
    from app.utils.logger import init_logging
    from script.worker import main

    init_logging()
    main()


@cli.command(name='agent-worker')
@click.option('--concurrency', type=click.IntRange(min=1), default=1, show_default=True)
def agent_worker(*, concurrency: int) -> None:
    """Start the shared isolated Celery worker for web agent and ACP runs."""
    from app.utils.logger import init_logging
    from script.worker import agent_main

    init_logging()
    agent_main(concurrency=concurrency)


@cli.command(name='workflow-v2-worker')
@click.option('--concurrency', type=click.IntRange(min=1), default=4, show_default=True)
def workflow_v2_worker(*, concurrency: int) -> None:
    """Start the isolated Celery worker for V2 DAG nodes."""
    from app.utils.logger import init_logging
    from script.worker import workflow_v2_main

    init_logging()
    workflow_v2_main(concurrency=concurrency)


@cli.command(name='workflow-v2-dispatcher')
@click.option('--stale-after', type=click.IntRange(min=30), default=300, show_default=True)
def workflow_v2_dispatcher(*, stale_after: int) -> None:
    """Start the durable V2 Outbox dispatcher and recovery scanner."""
    from app.utils.logger import init_logging
    from script.workflow_v2_dispatcher import main

    init_logging()
    main(stale_after_seconds=stale_after)


@cli.command(name='auto-init')
def auto_init() -> None:
    """Run the interactive system initialization wizard."""
    from script.auto_init import run_auto_init

    run_auto_init()


cli.add_command(init_cli, name='init')
cli.add_command(dev_cli, name='dev')
cli.add_command(eval_cli, name='eval')

if __name__ == '__main__':
    cli()
