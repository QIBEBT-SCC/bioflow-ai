import json
from collections.abc import Callable, Sequence
from typing import Any, Literal, Optional

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _DictOrPydantic, _DictOrPydanticClass
from pydantic import BaseModel


class ChatOpenAIReasoning(ChatOpenAI):
    provider: str = 'openai'

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)

        if (choices := chunk.get('choices')) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                if (reasoning_content := top.get('delta', {}).get('reasoning_content')) is not None:
                    generation_chunk.message.additional_kwargs['reasoning_content'] = reasoning_content
                # Handle use via OpenRouter
                elif (reasoning := top.get('delta', {}).get('reasoning')) is not None:
                    generation_chunk.message.additional_kwargs['reasoning_content'] = reasoning

        if (
            (citations := chunk.get('citations'))
            and generation_chunk
            and isinstance(generation_chunk.message, AIMessageChunk)
        ):
            generation_chunk.message.additional_kwargs['citations'] = citations

        return generation_chunk

    def _create_chat_result(
        self,
        response: dict | BaseModel,
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        for generation in rtn.generations:
            generation.message.response_metadata['model_provider'] = self.provider

        if not isinstance(response, BaseModel):
            return rtn

        choices = getattr(response, 'choices', None)
        if choices and hasattr(choices[0].message, 'reasoning_content'):
            rtn.generations[0].message.additional_kwargs['reasoning_content'] = choices[0].message.reasoning_content
        # Handle use via OpenRouter
        elif choices and hasattr(choices[0].message, 'model_extra'):
            model_extra = choices[0].message.model_extra
            if isinstance(model_extra, dict) and (reasoning := model_extra.get('reasoning')):
                rtn.generations[0].message.additional_kwargs['reasoning_content'] = reasoning

        if hasattr(response, 'citations'):
            rtn.generations[0].message.additional_kwargs['citations'] = response.citations

        return rtn


class ChatDeepseekReasoning(ChatOpenAIReasoning):
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # Ensure assistant messages include reasoning_content field
        for i, message in enumerate(payload['messages']):
            if message['role'] == 'assistant':
                # DeepSeek API expects assistant content to be a string, not a list
                if isinstance(message['content'], list):
                    text_parts = [
                        block.get('text', '')
                        for block in message['content']
                        if isinstance(block, dict) and block.get('type') == 'text'
                    ]
                    message['content'] = ''.join(text_parts) if text_parts else ''

                # Ensure each assistant message includes reasoning_content field
                # Get reasoning content from additional_kwargs, or set to empty string if not present
                if 'reasoning_content' not in message:
                    # Try to get reasoning content from original message
                    if isinstance(input_, list) and i < len(input_):
                        original_msg = input_[i]
                        if hasattr(original_msg, 'additional_kwargs'):
                            reasoning = original_msg.additional_kwargs.get('reasoning_content', '')
                            message['reasoning_content'] = reasoning
                        else:
                            message['reasoning_content'] = ''
                    else:
                        message['reasoning_content'] = ''

            # Handle tool messages
            elif message['role'] == 'tool' and isinstance(message['content'], list):
                message['content'] = json.dumps(message['content'])

        return payload

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: _DictOrPydanticClass | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        tool_choice = 'required' if tool_choice == 'any' else tool_choice
        return super().bind_tools(
            tools,
            tool_choice=tool_choice,
            strict=strict,
            parallel_tool_calls=parallel_tool_calls,
            response_format=response_format,
            **kwargs,
        )

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal['function_calling', 'json_mode', 'json_schema'] = 'json_schema',
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        method = 'json_mode' if method == 'json_schema' else method
        return super().with_structured_output(schema, method=method, include_raw=include_raw, strict=strict, **kwargs)
