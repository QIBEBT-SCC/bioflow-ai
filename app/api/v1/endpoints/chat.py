from datetime import UTC, datetime
from typing import Annotated, AsyncGenerator, List, Literal, Optional

import aiosqlite
import psycopg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_postgres import PostgresChatMessageHistory
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from loguru import logger
from sqlmodel import Session
from uuid_extensions import uuid7

from app.core.config import get_settings
from app.core.session import SessionDep
from app.crud import chat as chat_crud
from app.models.chat import ChatSessions
from app.schemas.chat import ChatResponse, ChatSessionPublic, ChatSessionUpdate
from app.schemas.file import StoredFileInfo
from app.utils.multimodal_builder import build_multimodal_message
from app.utils.security import UserDep
from llm.agents.reproduction.reproduction import ReproductionAgent
from llm.prebuild.chains import conclude_chain
from llm.states import ReproductionState
from llm.utils.ai_sdk.message import convert_to_langchain_message
from llm.utils.ai_sdk.stream_protocol import (
    data_part,
    done_marker,
    error_part,
    finish_message,
    reasoning_delta,
    reasoning_end,
    reasoning_start,
    start_message,
    text_delta,
    text_end,
    text_start,
    tool_input_start,
)

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


@router.post('/{session_id}/completions', description='流式生成对话')
async def chat_completions(
    session_id: str,
    response: ChatResponse,
    session: SessionDep,
    current_user: UserDep,
):
    """
    流式生成多模态聊天响应
    支持纯文本、图片、文档等多种输入类型
    """
    # 验证session_id
    if not session_id:
        raise HTTPException(status_code=400, detail='会话ID不能为空')

    human_input = convert_to_langchain_message(response.messages[-1])

    # 创建流式响应
    streaming_response = StreamingResponse(
        generate_chat_stream(human_input, response.model, response.webSearch),
    )

    # 设置 AI SDK 所需的响应头
    streaming_response.headers['x-vercel-ai-data-stream'] = 'v1'
    return streaming_response


async def generate_chat_stream(
    text_message: str, stored_files: List[StoredFileInfo], resume: bool, session_id: str, session: Session
) -> AsyncGenerator[str, None]:
    msg_db = get_settings().persistent.CHECKPOINT_URL
    async with await psycopg.AsyncConnection.connect(msg_db) as history_conn:
        try:
            yield start_message()
            history = PostgresChatMessageHistory('simple_history', session_id, async_connection=history_conn)
            messages = await history.aget_messages()
            messages = filter_messages(messages, exclude_types=['info', 'link'])

            async with aiosqlite.connect('checkpoints.db') as conn:
                memory = AsyncSqliteSaver(conn)
                agent = ReproductionAgent().build(memory=memory)

                # 可以向前端发送自定义数据片段，例如状态提示
                yield data_part('status', {'message': '模型加载完成，开始生成响应...'})

                config = RunnableConfig(configurable={'thread_id': session_id}, recursion_limit=500)
                if resume:
                    chat_input = Command(resume=text_message) if text_message else Command(resume=True)
                else:
                    # 构建多模态消息内容
                    if stored_files:
                        human_message = await build_multimodal_message(text_message, stored_files)
                    else:
                        human_message = HumanMessage(content=text_message)
                    chat_input = ReproductionState(messages=[*messages, human_message])

                current_state: Optional[Literal['thinking', 'text', 'tool']] = None
                wait_to_end: Optional[str] = None
                exclude_nodes = {'paper_extract:summary_paper'}
                is_interrupt = False
                async for source, stream_mode, chunk in agent.astream(
                    chat_input, config, stream_mode=['messages', 'updates', 'custom'], subgraphs=True
                ):
                    match stream_mode:
                        case 'messages':
                            chunk_data, chunk_meta = chunk

                            now_node = (
                                f'{source[-1].split(":")[0]}:{chunk_meta["langgraph_node"]}'
                                if source
                                else chunk_meta['langgraph_node']
                            )
                            if now_node == 'supervisor':
                                continue

                            # 处理 AIMessageChunk
                            if isinstance(chunk_data, AIMessageChunk):
                                # 处理思索内容
                                if 'reasoning_content' in chunk_data.additional_kwargs:
                                    if reasoning_content := chunk_data.additional_kwargs['reasoning_content']:
                                        if current_state != 'thinking':
                                            current_state = 'thinking'
                                            if wait_to_end:
                                                yield wait_to_end

                                            wait_to_end = reasoning_end(chunk_data.id)
                                            yield reasoning_start(chunk_data.id)

                                        yield reasoning_delta(chunk_data.id, reasoning_content)

                                # 处理工具调用
                                if 'tool_calls' in chunk_data.additional_kwargs:
                                    if wait_to_end:
                                        logger.info(wait_to_end)
                                        wait_to_end = None

                                    for tool_call in chunk_data.additional_kwargs['tool_calls']:
                                        if tool_call_id := tool_call['id']:
                                            yield tool_input_start(tool_call_id, tool_call['function']['name'])

                                # 处理正常文本内容
                                if chunk_data.content and now_node not in exclude_nodes:
                                    if current_state != 'text':
                                        current_state = 'text'
                                        if wait_to_end:
                                            logger.info(wait_to_end)

                                        wait_to_end = text_end(chunk_data.id)
                                        logger.debug(now_node)
                                        logger.info(text_start(chunk_data.id))

                                    yield text_delta(chunk_data.id, chunk_data.content)
                        case 'custom':
                            logger.info(chunk)
                            if 'info' in chunk:
                                yield data_part('info', {'message': chunk['info']})
                            elif 'link' in chunk:
                                # 可选择映射为 source-url；这里保留为自定义数据
                                yield data_part('link', chunk)
                        case 'updates':
                            logger.debug(chunk)
                            if '__interrupt__' in chunk:
                                yield data_part('interrupt', chunk['__interrupt__'][0].value)
                                is_interrupt = True
                        case _:
                            pass

                update_messages = []
                if stored_files:
                    update_messages.append(
                        HumanMessage(content=f'I upload files at: {",".join([f.file_path for f in stored_files])}')
                    )

                message_content = text_message if text_message else 'Yes' if resume else None
                if message_content:
                    update_messages.append(HumanMessage(content=message_content))

                response = await agent.aget_state(config)
                if not is_interrupt:
                    final_response = response.values['messages'][-1].content
                else:
                    interrupt_value = response.interrupts[0].value
                    final_response = (
                        interrupt_value['question'] if 'question' in interrupt_value else interrupt_value['confirm']
                    )
                update_messages.extend([AIMessage(content=final_response)])

                await history.aadd_messages(update_messages)

                chat_update = ChatSessionUpdate(uid=session_id, update_time=datetime.now(UTC))
                if len(messages) <= 4:
                    conclude_response = await conclude_chain().ainvoke({'history': [*messages, *update_messages]})
                    chat_update.description = conclude_response.content
                chat_crud.update(session, chat_update)

                if not is_interrupt:
                    await memory.adelete_thread(session_id)

                yield finish_message()
        except Exception as e:
            logger.exception(f'生成聊天响应时发生错误: {str(e)}')
            yield error_part(f'生成响应时发生错误: {str(e)}')
            yield finish_message()
        finally:
            yield done_marker()
