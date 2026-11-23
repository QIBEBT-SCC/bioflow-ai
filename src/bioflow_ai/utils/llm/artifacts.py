"""Helpers for reading structured payloads from ``ToolMessage.artifact``."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

END_LOOP_ARTIFACT_KEY = 'end_loop'
PARENT_HANDOFF_ARTIFACT_KEY = 'parent_request'


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


def trailing_tool_artifacts(messages: Sequence[BaseMessage]) -> Iterator[Mapping[str, Any]]:
    """Yield mapping artifacts from the trailing consecutive ``ToolMessage`` turn."""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        if isinstance(message.artifact, Mapping):
            yield message.artifact


def parent_handoff_request(messages: Sequence[BaseMessage]) -> str | None:
    """Return the handoff request if the trailing tool turn triggered ``return_to_parent``."""
    for artifact in trailing_tool_artifacts(messages):
        request = artifact.get(PARENT_HANDOFF_ARTIFACT_KEY)
        if isinstance(request, str):
            return request
    return None
