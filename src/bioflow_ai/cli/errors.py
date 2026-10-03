"""Translate service validation errors at the command-line boundary."""

from collections.abc import Callable
from functools import wraps
from typing import Any

import click


def handle_service_errors(function: Callable[..., Any]) -> Callable[..., Any]:
    """Keep reusable services independent of Click's exception classes."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except ValueError as error:
            raise click.ClickException(str(error)) from error

    return wrapped
