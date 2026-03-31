import asyncio
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessageChunk,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.types import Command
from loguru import logger
from sqlmodel import Session
from uuid_extensions import uuid7

import app.crud.chat as chat_crud
from app.core.config import get_settings
from app.core.session import SessionDep
from app.models.chat import ChatSessions
from app.schemas.chat import ChatResponse, ChatSessionPublic, ChatSessionUpdate, PaginatedChatSessions
from app.utils.security import UserDep
from llm.agents.reproduction.design_workflow import WorkflowDesigner
from llm.states import UserContext
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


@router.post('', response_model=ChatSessionPublic, description='创建新对话')
async def new_chat_session(session: SessionDep, current_user: UserDep):
    chat_id = uuid7()
    chat = ChatSessions(uid=chat_id, owner_id=current_user.id)
    chat = chat_crud.insert(session, chat)
    return chat


@router.get('', response_model=PaginatedChatSessions, description='获取历史对话列表（支持分页懒加载）')
async def get_chats_sessions(
    session: SessionDep, current_user: UserDep, offset: int = 0, limit: Annotated[int, Query(le=20)] = 8
):
    """获取用户的聊天会话列表，支持分页懒加载

    - **offset**: 偏移量，用于分页（默认0）
    - **limit**: 每页数量，最大20（默认8）

    返回包含总数和是否有更多数据的分页信息
    """
    chats = chat_crud.get_all(session, current_user.id, offset, limit)
    total = chat_crud.count_all(session, current_user.id)
    has_more = offset + limit < total

    return PaginatedChatSessions(
        total=total,
        offset=offset,
        limit=limit,
        has_more=has_more,
        data=[ChatSessionPublic.model_validate(chat.model_dump()) for chat in chats],
    )


@router.delete('/{session_id}', description='删除对话')
async def delete_chat_session(session: SessionDep, session_id: str, current_user: UserDep):
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
    yield start_message()

    history = PostgresMessageHistory(session, session_id)

    if response.resume and response.approved:
        agent_input = Command(resume=True)
    else:
        yield data_part('task', {'msg': 'Parsing user file...'})
        human_input = ui_input_to_human_message(response.messages[-1])
        human_input = await filter_multimedia_message(human_input)

        if response.resume:
            agent_input = Command(resume=human_input.content)
        else:
            history_messages = await history.aget_messages()
            await history.aadd_messages([human_input])
            agent_input = {'messages': [*history_messages, human_input]}

    chat = chat_crud.get(session, session_id)
    if response.resume:
        thread_id = chat.checkpoint_uid
    else:
        thread_id = uuid7()
        chat_crud.update(session, ChatSessionUpdate(uid=chat.uid, checkpoint_uid=thread_id))

    async with AsyncRedisSaver.from_conn_string(get_settings().persistent.cache.LLM_CACHE) as checkpoint:
        try:
            agent = WorkflowDesigner(history=history).build(memory=checkpoint)
            config = RunnableConfig(configurable={'thread_id': thread_id}, recursion_limit=500)

            current_state: Optional[Literal['thinking', 'text', 'tool']] = None
            wait_to_end: Optional[str] = None
            exclude_nodes = {'intent_recognition'}
            is_interrupt = False
            async for source, stream_mode, chunk in agent.astream(
                agent_input,
                config,
                context=UserContext(user_id=user_id),
                stream_mode=['messages', 'updates', 'custom'],
                subgraphs=True,
            ):
                match stream_mode:
                    case 'messages':
                        chunk_data, chunk_meta = chunk

                        now_node = (
                            f'{source[-1].split(":")[0]}:{chunk_meta["langgraph_node"]}'
                            if source
                            else chunk_meta['langgraph_node']
                        )
                        if now_node in exclude_nodes:
                            continue

                        # 处理 AIMessageChunk
                        logger.debug(now_node)
                        logger.info(chunk)
                        if isinstance(chunk_data, AIMessageChunk):
                            # 处理思索内容
                            if 'reasoning_content' in chunk_data.additional_kwargs:
                                if reasoning_content := chunk_data.additional_kwargs['reasoning_content']:
                                    if current_state != 'thinking':
                                        current_state = 'thinking'
                                        if wait_to_end:
                                            yield wait_to_end
                                            # yield finish_step()

                                        wait_to_end = reasoning_end(chunk_data.id)
                                        # yield start_step()
                                        yield reasoning_start(chunk_data.id)

                                    yield reasoning_delta(chunk_data.id, reasoning_content)

                            # 处理正常文本内容
                            if chunk_data.content and now_node not in exclude_nodes:
                                if current_state != 'text':
                                    current_state = 'text'
                                    if wait_to_end:
                                        yield wait_to_end

                                    wait_to_end = text_end(chunk_data.id)
                                    yield text_start(chunk_data.id)

                                yield text_delta(chunk_data.id, chunk_data.content)
                    case 'custom':
                        logger.info(chunk)
                        if 'info' in chunk:
                            yield data_part('info', {'message': chunk['info']})
                        elif 'link' in chunk:
                            # 可选择映射为 source-url；这里保留为自定义数据
                            yield data_part('link', chunk)
                    case 'updates':
                        yield finish_step()
                        yield start_step()
                        logger.debug(chunk)
                        if '__interrupt__' in chunk:
                            is_interrupt = True
                    case _:
                        logger.debug(chunk)

            if wait_to_end:
                yield wait_to_end
                yield finish_step()

            response = await agent.aget_state(config)
            if not is_interrupt:
                final_response = response.values['messages'][-1]
                await history.aadd_messages([final_response])
                await checkpoint.adelete_thread(thread_id)
            else:
                interrupt_value = response.interrupts[0].value
                yield data_part('interrupt', interrupt_value)
                await history.aadd_messages([InterruptMessage(content=interrupt_value)])
                chat_crud.update(session, ChatSessionUpdate(uid=chat.uid, interrupted=True))

        except Exception as e:
            logger.exception(f'生成聊天响应时发生错误: {e!s}')
            yield error_part(f'生成响应时发生错误: {e!s}')
        finally:
            yield finish_message()
            yield done_marker()


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
