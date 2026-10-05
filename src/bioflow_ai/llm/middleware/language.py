from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import SystemMessage


class UserLanguageMiddleware(AgentMiddleware):
    """Append an agent-specific language prompt using the runtime context."""

    def __init__(self, language_prompt: str) -> None:
        self.language_prompt = language_prompt

    def wrap_model_call(  # noqa: D102
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        language = getattr(request.runtime.context, 'language', None)
        if not language or request.system_message is None:
            return handler(request)

        language_prompt = self.language_prompt.format(language)
        new_content = [
            *list(request.system_message.content_blocks),
            {'type': 'text', 'text': language_prompt},
        ]
        new_system_message = SystemMessage(content=new_content)  # ty: ignore[no-matching-overload]
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(  # noqa: D102
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        language = getattr(request.runtime.context, 'language', None)
        if not language or request.system_message is None:
            return await handler(request)

        language_prompt = self.language_prompt.format(language)
        new_content = [
            *list(request.system_message.content_blocks),
            {'type': 'text', 'text': language_prompt},
        ]
        new_system_message = SystemMessage(content=new_content)  # ty: ignore[no-matching-overload]
        return await handler(request.override(system_message=new_system_message))
