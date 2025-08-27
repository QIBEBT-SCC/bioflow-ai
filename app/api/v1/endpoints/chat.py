import asyncio
from datetime import datetime, UTC
from typing import AsyncGenerator, Optional, Annotated

import psycopg
from fastapi import APIRouter, HTTPException, Query, Form
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage, filter_messages, BaseMessage, HumanMessage, AIMessage
from langchain_postgres import PostgresChatMessageHistory
from loguru import logger
from sqlmodel import Session
from uuid_extensions import uuid7

from app.crud import chat as chat_crud
from app.models.chat import ChatSessions
from app.schemas.chat import ChatSessionUpdate, ChatSessionPublic
from app.api.sse_base import SSEMessage, EventType
from app.core.config import get_settings
from app.core.session import SessionDep
from app.utils.security import UserDep
from llm.prebuild.chains import conclude_chain
from llm.prebuild.simple_agent import simple_agent

router = APIRouter()


@router.post('', response_model=ChatSessionPublic, description='创建新对话')
async def new_chat_session(session: SessionDep, current_user: UserDep):
    chat_id = uuid7()
    chat = ChatSessions(uid=chat_id, owner_id=current_user.id)
    chat = chat_crud.insert(session, chat)
    return chat


@router.get('', response_model=list[ChatSessionPublic], description='获取历史对话列表')
async def get_chats_sessions(
    session: SessionDep, current_user: UserDep, offset: int = 0, limit: Annotated[int, Query(le=8)] = 8
):
    chats = chat_crud.get_all(session, current_user.id, offset, limit)
    return chats


@router.post('/completions', description='流式生成对话')
async def chat_completions(
    message: Annotated[str, Form()], session_id: Annotated[str, Form()], session: SessionDep, current_user: UserDep
):
    if not message:
        raise HTTPException(status_code=400, detail='消息不能为空')

    logger.info(f'用户 {current_user.username} 发起聊天请求')

    # 流式响应
    return StreamingResponse(
        generate_chat_stream(message, session_id, session),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',  # 禁用nginx缓冲
        },
    )


@router.delete('/{session_id}', description='删除对话')
async def delete_chat_session(session: SessionDep, session_id: str, current_user: UserDep):
    chat = chat_crud.get(session, session_id)
    if not chat:
        raise HTTPException(status_code=404, detail='对话不存在')
    chat_crud.delete(session, chat)

    return {'message': '对话已删除'}


@router.put('/{session_id}', description='更新对话信息')
async def update_chat_history(session: SessionDep, session_id: str, description: str, current_user: UserDep):
    chat = ChatSessionUpdate(uid=session_id, description=description, update_time=datetime.now(UTC))
    chat = chat_crud.update(session, chat)
    return chat


@router.get('/{session_id}/history', response_model=list[BaseMessage], description='获取对话历史消息')
async def get_history(session_id: str, current_user: UserDep):
    msg_db = get_settings().persistent.CHECKPOINT_URL
    conn = await psycopg.AsyncConnection.connect(msg_db)
    try:
        history = PostgresChatMessageHistory('simple_history', session_id, async_connection=conn)
        messages = await history.aget_messages()
        return messages
    finally:
        await conn.close()


async def generate_chat_stream(message: str, session_id: Optional[str], session: Session) -> AsyncGenerator[str, None]:
    """生成流式聊天响应"""
    msg_db = get_settings().persistent.CHECKPOINT_URL
    conn = await psycopg.AsyncConnection.connect(msg_db)
    try:
        history = PostgresChatMessageHistory('simple_history', session_id, async_connection=conn)
        messages = await history.aget_messages()
        messages = filter_messages(messages, exclude_types=['info'])

        try:
            # 发送开始生成消息
            yield SSEMessage(event=EventType.LOADING, data={'message': '初始化智能体...'}).to_sse()

            # 获取LLM实例
            agent = simple_agent

            yield SSEMessage(event=EventType.GENERATING, data={'message': '模型加载完成，开始生成响应...'}).to_sse()

            # 流式生成响应
            temp_response = ''
            final_response = ''
            tool_name = ''
            tool_call_id = ''
            is_tool_call = False
            async for stream_mode, chunk in agent.astream(
                {'messages': [*messages, ('human', message)]}, stream_mode=['messages', 'custom']
            ):
                match stream_mode:
                    case 'messages':
                        chunk_data, _ = chunk

                        if isinstance(chunk_data, AIMessageChunk):
                            if 'tool_calls' in chunk_data.additional_kwargs and not is_tool_call:
                                # 工具调用
                                tool_name = chunk_data.additional_kwargs['tool_calls'][0]['function']['name']
                                tool_call_id = chunk_data.additional_kwargs['tool_calls'][0]['id']
                                is_tool_call = True

                            if chunk_data.usage_metadata:
                                # 生成结束
                                if is_tool_call:
                                    yield SSEMessage(
                                        event=EventType.TOOL_CALL,
                                        data={'id': tool_call_id, 'tool_name': tool_name, 'status': 'begin'},
                                    ).to_sse()
                                    is_tool_call = False
                                else:
                                    yield SSEMessage(
                                        event=EventType.SUCCESS,
                                        data={
                                            'id': chunk_data.id,
                                            'message': {'role': 'assistant', 'content': temp_response},
                                        },
                                    ).to_sse()
                                    final_response = temp_response
                                    temp_response = ''
                                logger.debug(chunk_data.usage_metadata)

                            if chunk_data.content:
                                temp_response += chunk_data.content

                                yield SSEMessage(
                                    event=EventType.GENERATING,
                                    data={'id': chunk_data.id, 'delta': chunk_data.content, 'full_text': temp_response},
                                ).to_sse()

                                # 添加小延迟，避免客户端处理过快
                                await asyncio.sleep(0.01)

                        elif isinstance(chunk_data, ToolMessage):
                            yield SSEMessage(
                                event=EventType.TOOL_CALL,
                                data={'id': chunk_data.tool_call_id, 'tool_name': tool_name, 'status': 'end'},
                            ).to_sse()
                        else:
                            logger.debug(chunk)
                    case _:
                        pass

            chat_update = ChatSessionUpdate(uid=session_id, update_time=datetime.now(UTC))
            if len(messages) < 6:
                conclude_response = await conclude_chain().ainvoke(
                    {'history': [('human', message), ('ai', final_response)]}
                )
                chat_update.description = conclude_response.content
            chat_crud.update(session, chat_update)

            await history.aadd_messages([HumanMessage(content=message), AIMessage(content=final_response)])

        except Exception as e:
            logger.exception(f'生成聊天响应时发生错误: {str(e)}')
            yield SSEMessage(event=EventType.ERROR, data={'error': '生成响应时发生错误', 'detail': str(e)}).to_sse()
    finally:
        await conn.close()
