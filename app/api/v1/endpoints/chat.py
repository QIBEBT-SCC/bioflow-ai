import asyncio
from datetime import datetime, UTC
from typing import AsyncGenerator, Optional, Annotated, List

import aiosqlite
import psycopg
from fastapi import APIRouter, HTTPException, Query, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, filter_messages, BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_postgres import PostgresChatMessageHistory
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from loguru import logger
from sqlmodel import Session
from uuid_extensions import uuid7

from app.crud import chat as chat_crud
from app.models.chat import ChatSessions
from app.schemas.chat import ChatSessionUpdate, ChatSessionPublic
from app.schemas.file import StoredFileInfo
from app.api.sse_base import SSEMessage, EventType
from app.core.config import get_settings
from app.core.session import SessionDep
from app.utils.security import UserDep
from app.utils.file.file_storage import file_storage
from app.utils.file.file_validator import file_validator
from app.utils.multimodal_builder import build_multimodal_message
from llm.prebuild.chains import conclude_chain
from llm.reproduction.reproduction import ReproductionAgent
from llm.states import ReproductionState

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


@router.post('/completions', description='流式生成对话（支持多模态输入）')
async def chat_completions(
    session: SessionDep,
    current_user: UserDep,
    message: Annotated[str, Form(description='用户消息')],
    session_id: Annotated[str, Form(description='会话ID')],
    resume: Annotated[bool, Form(description='是否是一条中断信息')],
    files: List[UploadFile] = File(default=[], description='要处理的文件列表'),
):
    """
    流式生成多模态聊天响应
    支持纯文本、图片、文档等多种输入类型
    """
    # 验证session_id
    if not session_id:
        raise HTTPException(status_code=400, detail='会话ID不能为空')

    # 处理上传的文件
    stored_files = []

    for upload_file in files:
        try:
            # 验证文件安全性
            is_valid, error_msg, validated_info = await file_validator.validate_upload_file(upload_file)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)

            # 保存文件
            stored_info = await file_storage.save_file(upload_file, current_user.username, validated_info)
            stored_files.append(stored_info)

            logger.info(f'用户 {current_user.username} 上传并处理文件: {upload_file.filename}')

        except Exception as e:
            logger.error(f'处理上传文件失败 {upload_file.filename}: {str(e)}')
            raise HTTPException(status_code=400, detail=f'处理文件失败: {upload_file.filename}')

    logger.info(f'用户 {current_user.username} 发起聊天请求，文件数量: {len(stored_files)}')

    # 流式响应
    return StreamingResponse(
        generate_chat_stream(message, stored_files, resume, session_id, session),
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


async def generate_chat_stream(
    text_message: str, stored_files: List[StoredFileInfo], resume: bool, session_id: Optional[str], session: Session
) -> AsyncGenerator[str, None]:
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
            async with aiosqlite.connect('checkpoints.db') as conn:
                memory = AsyncSqliteSaver(conn)
                agent = ReproductionAgent().build(memory=memory)

                yield SSEMessage(event=EventType.LOADING, data={'message': '模型加载完成，开始生成响应...'}).to_sse()

                config = RunnableConfig(configurable={'thread_id': '1'}, recursion_limit=500)
                if resume:
                    chat_input = Command(resume=text_message) if text_message else Command(resume=True)
                else:
                    # 构建多模态消息内容
                    if stored_files:
                        human_message = await build_multimodal_message(text_message, stored_files)
                    else:
                        human_message = HumanMessage(content=text_message)
                    chat_input = ReproductionState(messages=[*messages, human_message])

                exclude_nodes = ['extract_tools_node']
                # 流式生成响应
                current_id = ''
                temp_response = ''
                is_tool_call = {}
                is_interrupt = False
                async for source, stream_mode, chunk in agent.astream(
                    chat_input, config, stream_mode=['messages', 'updates', 'custom'], subgraphs=True
                ):
                    match stream_mode:
                        case 'messages':
                            message_chunk, chunk_meta = chunk

                            if isinstance(message_chunk, AIMessageChunk):
                                if message_chunk.id != current_id:
                                    current_id = message_chunk.id
                                    is_tool_call[source] = False
                                    temp_response = ''

                                if 'tool_calls' in message_chunk.additional_kwargs and not is_tool_call[source]:
                                    # 工具调用
                                    tool_name = message_chunk.additional_kwargs['tool_calls'][0]['function']['name']
                                    yield SSEMessage(
                                        event=EventType.LOADING, data={'message': f'call tool {tool_name}...'}
                                    ).to_sse()

                                if message_chunk.content and chunk_meta['langgraph_node'] not in exclude_nodes:
                                    temp_response += message_chunk.content

                                    yield SSEMessage(
                                        event=EventType.GENERATING,
                                        data={
                                            'id': current_id,
                                            'delta': message_chunk.content,
                                            'full_text': temp_response,
                                        },
                                    ).to_sse()

                                    await asyncio.sleep(0.01)
                        case 'custom':
                            logger.info(chunk['info'])
                            yield SSEMessage(event=EventType.LOADING, data={'message': chunk['info']}).to_sse()
                        case 'updates':
                            logger.debug(chunk)
                            if '__interrupt__' in chunk:
                                yield SSEMessage(
                                    event=EventType.INTERRUPT, data={'message': chunk['__interrupt__'][0].value}
                                ).to_sse()
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
                    final_response = response.interrupts[0].value
                update_messages.extend([AIMessage(content=final_response)])

                await history.aadd_messages(update_messages)

                chat_update = ChatSessionUpdate(uid=session_id, update_time=datetime.now(UTC))
                if len(messages) <= 4:
                    conclude_response = await conclude_chain().ainvoke({'history': [*messages, *update_messages]})
                    chat_update.description = conclude_response.content
                chat_crud.update(session, chat_update)

                if not is_interrupt:
                    await memory.adelete_thread('1')

        except Exception as e:
            logger.exception(f'生成聊天响应时发生错误: {str(e)}')
            yield SSEMessage(event=EventType.ERROR, data={'error': '生成响应时发生错误', 'detail': str(e)}).to_sse()
    finally:
        await conn.close()
