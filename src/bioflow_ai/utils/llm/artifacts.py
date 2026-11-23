"""Helpers for reading structured payloads from ``ToolMessage.artifact``."""

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage


def latest_tool_artifact(
    messages: Sequence[BaseMessage],
    *required_keys: str,
) -> Mapping[str, Any] | None:
    """Return the most recent mapping ``ToolMessage.artifact`` containing all ``required_keys``."""
    for message in reversed(messages):
        if (
            isinstance(message, ToolMessage)
            and isinstance(message.artifact, Mapping)
            and all(key in message.artifact for key in required_keys)
        ):
            return message.artifact
    return None
