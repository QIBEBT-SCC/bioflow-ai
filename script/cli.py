import click

from script.develop import cli as dev_cli
from script.evaluate import cli as eval_cli
from script.init import cli as init_cli


@click.group()
def cli() -> None:
    """BioFlowAI management CLI."""


@cli.command()
def api() -> None:
    """Start the FastAPI API server."""
    from app.utils.logger import init_logging
    from script.server import main

    init_logging()
    main()


@cli.command()
def worker() -> None:
    """Start the Celery worker."""
    from app.utils.logger import init_logging
    from script.worker import main

    init_logging()
    main()


@cli.command(name='auto-init')
def auto_init() -> None:
    """Run the interactive system initialization wizard (config + database)."""
    from script.auto_init import run_auto_init

    run_auto_init()


cli.add_command(init_cli, name='init')
cli.add_command(dev_cli, name='dev')
cli.add_command(eval_cli, name='eval')

if __name__ == '__main__':
    cli()
