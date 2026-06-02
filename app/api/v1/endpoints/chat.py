import asyncio
import json
import sys
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.stream.transformers import CustomTransformer
from langgraph.types import Command
from loguru import logger
from sqlmodel import Session
from uuid_extensions import uuid7

import app.crud.chat as chat_crud
from app.core.config import get_settings
from app.core.session import SessionDep
from app.models.chat import ChatSessions
from app.schemas.chat import ChatResponse, ChatSessionPage, ChatSessionPublic, ChatSessionUpdate
from app.utils.security import UserDep
from llm.agents.workflow_agent import DesignContext, WorkflowDesignAgent
from llm.utils.ai_sdk.stream_protocol import (
    data_part,
    done_marker,
    error_part,
    finish_message,
    finish_step,
    reasoning_delta,
    reasoning_end,
    reasoning_start,
    start_message,
    start_step,
    text_delta,
    text_end,
    text_start,
)
from llm.utils.ai_sdk.stream_translator import AISDKStreamTranslator
from llm.utils.ai_sdk.ui_message import UIMessage
from llm.utils.lc.memory import PostgresMessageHistory
from llm.utils.lc.messages import InterruptMessage
from llm.utils.multimodal import (
    filter_multimedia_message,
    langchain_messages_to_ui_messages,
    rebuild_multimodal_messages,
    ui_input_to_human_message,
)

router = APIRouter()


@router.post('', description='Create new chat session')
async def new_chat_session(session: SessionDep, current_user: UserDep) -> ChatSessionPublic:
    """Create new chat session."""
    chat = ChatSessions(owner_id=current_user.id)
    # pyrefly: ignore [bad-return]
    return chat_crud.insert(session, chat)


@router.get('', description='获取历史对话列表')
async def get_chats_sessions(
    session: SessionDep,
    current_user: UserDep,
    cursor: datetime | None = None,
    limit: Annotated[int, Query(le=20)] = 8,
) -> ChatSessionPage:
    """Retrieve the user's chat session list for scroll-based lazy loading.

    Args:
        cursor: The ``update_time`` of the last item from the previous page; omit for the first page.
        limit: The number of items per page, maximum of 20 (default is 8)

    Returns:
        a page of sessions plus ``next_cursor`` and a flag indicating whether more data is available.
    """
    chats = chat_crud.get_page(session, current_user.id, cursor, limit)
    has_more = len(chats) > limit
    chats = chats[:limit]

    return ChatSessionPage(
        limit=limit,
        has_more=has_more,
        next_cursor=chats[-1].update_time if chats else None,
        data=[ChatSessionPublic.model_validate(chat) for chat in chats],
    )


@router.delete('/{session_id}', description='删除对话')
async def delete_chat_session(session: SessionDep, session_id: str, _current_user: UserDep):
    chat = chat_crud.get(session, session_id)
    if not chat:
        raise HTTPException(status_code=404, detail='对话不存在')
    chat_crud.delete(session, chat)

    history = PostgresMessageHistory(session, session_id)
    await history.aclear()

    return {'message': '对话已删除'}


@router.put('/{session_id}', description='更新对话信息')
async def update_chat_history(session: SessionDep, session_id: str, description: str, current_user: UserDep):
    chat = ChatSessionUpdate(uid=uuid.UUID(session_id), description=description, update_time=datetime.now(UTC))
    chat = chat_crud.update(session, chat)
    return chat


@router.get('/{session_id}', response_model=ChatSessionPublic)
async def get_chat_session(session: SessionDep, session_id: str, current_user: UserDep):
    chat = chat_crud.get(session, session_id)
    if not chat:
        raise HTTPException(status_code=404, detail='对话不存在')
    return chat


@router.get('/{session_id}/history', response_model=list[UIMessage], description='获取对话历史消息')
async def get_history(session: SessionDep, session_id: str, current_user: UserDep):
    history = PostgresMessageHistory(session, session_id)
    messages = await history.aget_messages()
    messages = rebuild_multimodal_messages(messages)
    return langchain_messages_to_ui_messages(messages)


@router.post('/{session_id}/completions', description='流式生成对话')
async def chat_completions(
    session_id: str,
    response: ChatResponse,
    session: SessionDep,
    current_user: UserDep,
):
    # 验证session_id
    if not session_id:
        raise HTTPException(status_code=400, detail='会话ID不能为空')

    assert current_user.id is not None
    if response.rebuild:
        # 针对构建工作流任务
        streaming_response = StreamingResponse(
            generate_reproduction_stream(response, session_id, session, current_user.id),
        )
    else:
        # 针对普通对话
        streaming_response = StreamingResponse(
            generate_test_stream(),
        )

    # 设置 AI SDK 所需的响应头
    streaming_response.headers['x-vercel-ai-data-stream'] = 'v1'
    return streaming_response


async def generate_reproduction_stream(
    response: ChatResponse, session_id: str, session: Session, user_id: int
) -> AsyncGenerator[str, None]:
    history = PostgresMessageHistory(session, session_id)
    chat = chat_crud.get(session, session_id)

    prelude: list[str] = []
    if response.resume and response.approved:
        agent_input = Command(resume=True)
    else:
        prelude.append(data_part('info', {'message': 'Parsing user file...'}))
        human_input = ui_input_to_human_message(response.messages[-1])
        human_input = await filter_multimedia_message(human_input)

        if response.resume:
            agent_input = Command(resume=human_input.content)
        else:
            history_messages = await history.aget_messages()
            await history.aadd_messages([human_input])
            agent_input = {'messages': [*history_messages, human_input]}

    if response.resume:
        thread_id = chat.checkpoint_uid
    else:
        thread_id = uuid7()
        chat_crud.update(session, ChatSessionUpdate(uid=chat.uid, checkpoint_uid=thread_id))

    async with AsyncRedisSaver.from_conn_string(
        get_settings().persistent.cache.LLM_CACHE.unicode_string()
    ) as checkpoint:
        agent = WorkflowDesignAgent(memory=checkpoint).build()
        config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=sys.maxsize)
        translator = AISDKStreamTranslator(
            text_nodes={'generate_workflow'},
            exclude_nodes={'intent_recognition'},
        )
        try:
            run_stream = await agent.astream_events(
                agent_input,
                config,
                version='v3',
                context=DesignContext(user_id=user_id),
                transformers=[CustomTransformer],
            )
            async for line in translator.run(run_stream, prelude=prelude):
                yield line
        except Exception as e:
            logger.exception(f'生成聊天响应时发生错误: {e!s}')
            yield start_message()
            yield error_part(f'生成响应时发生错误: {e!s}')
            yield finish_message()
            yield done_marker()
            return

        # 恢复运行时，先把上一条中断消息标记为已处理并清除中断标志
        if response.resume:
            history.update_last_interrupt('approved' if response.approved else 'rejected')
            chat_crud.update(session, ChatSessionUpdate(uid=chat.uid, interrupted=False))

        if translator.is_interrupt:
            await history.aadd_messages(
                [InterruptMessage(content=json.dumps(translator.interrupt_value, ensure_ascii=False))]
            )
            chat_crud.update(session, ChatSessionUpdate(uid=chat.uid, interrupted=True))
        else:
            state = await agent.aget_state(config)
            final_response = state.values['messages'][-1]
            await history.aadd_messages([final_response])
            await checkpoint.adelete_thread(thread_id)


async def generate_test_stream() -> AsyncGenerator[str, None]:
    yield start_message()
    await asyncio.sleep(1)
    yield start_step()
    yield reasoning_start('111')
    yield reasoning_delta('111', '这是一个思考的内容。')
    await asyncio.sleep(1)
    yield reasoning_end('111')

    yield data_part('test', {'mes': '这是一个测试数据部分。'})
    await asyncio.sleep(1)

    yield reasoning_start('222')
    yield reasoning_delta('222', '这是另一个思考的内容。')
    await asyncio.sleep(1)
    yield reasoning_end('222')

    yield text_start('333')
    yield text_delta('333', '这是生成的文本内容。')
    await asyncio.sleep(1)
    yield text_end('333')
    yield finish_step()

    yield start_step()
    yield data_part('test', {'mes': '这是一个测试数据部分。'})
    await asyncio.sleep(1)

    yield reasoning_start('444')
    yield reasoning_delta('444', '这是另一个思考的内容。')
    await asyncio.sleep(1)
    yield reasoning_end('444')

    yield text_start('555')
    yield text_delta('555', '这是生成的文本内容。')
    await asyncio.sleep(1)
    yield text_end('555')
    yield finish_step()

    yield start_step()
    yield data_part('link', {'to': '连接转发'})
    await asyncio.sleep(1)
    yield finish_step()

    yield finish_message()
    yield done_marker()
