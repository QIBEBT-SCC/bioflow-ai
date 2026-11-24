import click

from bioflow_ai.cli.develop import cli as dev_cli
from bioflow_ai.cli.evaluate import cli as eval_cli
from bioflow_ai.cli.init import cli as init_cli
from bioflow_ai.cli.services import logs, restart, start, status, stop


@click.group()
def cli() -> None:
    """BioFlowAI management CLI."""


@cli.command(name='auto-init')
def auto_init() -> None:
    """Run the interactive system initialization wizard."""
    from bioflow_ai.cli.initialization.wizard import run_auto_init

    run_auto_init()


cli.add_command(start)
cli.add_command(stop)
cli.add_command(restart)
cli.add_command(status)
cli.add_command(logs)
cli.add_command(init_cli, name='init')
cli.add_command(dev_cli, name='dev')
cli.add_command(eval_cli, name='eval')

if __name__ == '__main__':
    cli()
