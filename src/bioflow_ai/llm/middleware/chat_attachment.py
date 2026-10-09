"""Prepare chat attachments for model transport without changing graph state."""

import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from bioflow_ai.llm.utils.multimodal import rebuild_multimodal_messages


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


class ChatAttachmentMiddleware(AgentMiddleware):
    """Hydrate references and adapt tool images in an isolated request copy."""

    @staticmethod
    def _prepare(request: ModelRequest) -> ModelRequest:
        messages = rebuild_multimodal_messages(request.messages)
        if isinstance(request.model, ChatOpenAI) and not request.model.use_responses_api:
            messages = move_tool_images(messages)
        return request.override(messages=messages)

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        """Prepare an isolated synchronous request."""
        return handler(self._prepare(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelCallResult:
        """Read object storage outside the event loop."""
        prepared = await asyncio.to_thread(self._prepare, request)
        return await handler(prepared)
