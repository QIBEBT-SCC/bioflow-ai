"""BioFlowAI event callbacks for the official ACP Python SDK."""

from collections.abc import Awaitable, Callable
from typing import Any

from acp import RequestError
from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    ElicitationUrlRequestMode,
    ElicitationUrlSessionMode,
    RequestPermissionResponse,
)

UpdateHandler = Callable[[dict[str, Any]], Awaitable[None]]
ElicitationHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class CodeAgentClient:
    """Adapt SDK events to the application's SSE payloads."""

    def __init__(self, on_update: UpdateHandler, on_elicitation: ElicitationHandler | None) -> None:
        self.on_update = on_update
        self.on_elicitation = on_elicitation

    async def session_update(self, session_id: str, update: Any, **_kwargs: Any) -> None:
        """Forward typed updates using the frontend event field names."""
        await self.on_update(
            {'sessionId': session_id, 'update': update.model_dump(mode='json', by_alias=True, exclude_unset=True)}
        )

    async def request_permission(
        self, session_id: str, tool_call: Any, options: list[Any], **_kwargs: Any
    ) -> RequestPermissionResponse:
        """Approve agent requests; Codex and the container enforce execution boundaries."""
        for option in options:
            if option.kind in {'allow_always', 'allow_once'}:
                return RequestPermissionResponse(outcome=AllowedOutcome(optionId=option.option_id, outcome='selected'))
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome='cancelled'))

    async def create_elicitation(self, message: str, mode: Any, **_kwargs: Any) -> Any:
        """Expose URL login requests without granting form or filesystem capabilities."""
        if self.on_elicitation is None or not isinstance(mode, (ElicitationUrlRequestMode, ElicitationUrlSessionMode)):
            method = 'elicitation/create'
            raise RequestError.method_not_found(method)
        return await self.on_elicitation(
            {'message': message, 'mode': 'url', **mode.model_dump(mode='json', by_alias=True, exclude_none=True)}
        )
