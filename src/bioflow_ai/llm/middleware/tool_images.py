"""Adapt visual tool results for OpenAI Chat Completions without mutating state."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI


def move_tool_images(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Insert images after each complete group of consecutive tool results."""
    result: list[AnyMessage] = []
    visual: list[str | dict] = []
    for message in messages:
        if not isinstance(message, ToolMessage) and visual:
            result.append(HumanMessage(content=visual))
            visual = []
        if isinstance(message, ToolMessage) and isinstance(message.content, list):
            images = [
                part
                for part in message.content
                if isinstance(part, dict) and part.get('type') in {'image', 'image_url'}
            ]
            if images:
                text = [part for part in message.content if part not in images]
                visual.extend(
                    [
                        {
                            'type': 'text',
                            'text': f'Visual tool result from {message.name or "view_image"}, '
                            f'call {message.tool_call_id}. This is tool evidence, not a new user request.',
                        },
                        *[part for part in text if isinstance(part, dict) and part.get('type') == 'text'],
                        *images,
                    ]
                )
                result.append(message.model_copy(update={'content': text or 'Image loaded; visual evidence follows.'}))
                continue
        result.append(message)
    if visual:
        result.append(HumanMessage(content=visual))
    return result


class ToolImageMiddleware(AgentMiddleware):
    """Use a user visual message only for OpenAI Chat Completions transport."""

    @staticmethod
    def _prepare(request: ModelRequest) -> ModelRequest:
        model = request.model
        if isinstance(model, ChatOpenAI) and not model.use_responses_api:
            return request.override(messages=move_tool_images(request.messages))
        return request

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        """Adapt synchronous model input."""
        return handler(self._prepare(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelCallResult:
        """Adapt asynchronous model input."""
        return await handler(self._prepare(request))
