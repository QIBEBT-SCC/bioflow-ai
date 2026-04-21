import click

from script.develop import cli as dev_cli
from script.evaluate import cli as eval_cli
from script.init_db import cli as init_cli


@click.group()
def cli() -> None:
    """BioFlowAI management CLI."""


@cli.command()
def api() -> None:
    """Start the FastAPI API server."""
    from script.server import main

    main()


@cli.command()
def worker() -> None:
    """Start the Celery worker."""
    from script.worker import main

    main()


cli.add_command(init_cli, name='init')
cli.add_command(dev_cli, name='dev')
cli.add_command(eval_cli, name='eval')

if __name__ == '__main__':
    cli()
