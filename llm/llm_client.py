from typing import Optional, Union

import openai
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI


class ChatOpenAIReasoning(ChatOpenAI):
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

        return generation_chunk

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        choices = getattr(response, 'choices', None)
        if choices and hasattr(choices[0].message, 'reasoning_content'):
            rtn.generations[0].message.additional_kwargs['reasoning_content'] = choices[0].message.reasoning_content
        # Handle use via OpenRouter
        elif choices and hasattr(choices[0].message, 'model_extra'):
            model_extra = choices[0].message.model_extra
            if isinstance(model_extra, dict) and (reasoning := model_extra.get('reasoning')):
                rtn.generations[0].message.additional_kwargs['reasoning_content'] = reasoning

        return rtn
