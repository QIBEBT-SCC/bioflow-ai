"""Internal background runtime entry point for managed backend services."""

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Initialize logging and start the requested runtime service."""
    parser = argparse.ArgumentParser(prog='bioflow_ai.runtime.entrypoint')
    parser.add_argument('service', choices=('api', 'execution-worker', 'agent-worker'))
    parser.add_argument('--log', action='store_true', help='Enable request body logging middleware for the API.')
    args = parser.parse_args(argv)

    from bioflow_ai.infrastructure.logging import init_logging

    init_logging()

    if args.service == 'api':
        from bioflow_ai.runtime.server import main as api_main

        api_main(log_requests=args.log)
    elif args.service == 'execution-worker':
        from bioflow_ai.runtime.worker import execution_main

        execution_main()
    else:
        from bioflow_ai.runtime.worker import agent_main

        agent_main()


if __name__ == '__main__':
    main()
