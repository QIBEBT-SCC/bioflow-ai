import asyncio
import sys
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sqlalchemy.orm import Session
from starlette.status import HTTP_404_NOT_FOUND
from uuid_extensions import uuid7str

import app.crud.chat as chat_crud
from app.core.config import get_settings
from app.core.session import SessionDep
from app.models.chat import ChatSessions
from app.schemas.chat import ChatResponse, ChatSessionPage, ChatSessionPublic, ChatSessionUpdate
from app.services import chat as chat_service
from app.utils.security import UserDep
from llm.context import CreateToolContext, ProjectContext
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
from llm.utils.multimodal import (
    filter_multimedia_message,
    langchain_messages_to_ui_messages,
    rebuild_multimodal_messages,
    ui_input_to_human_message,
)

router = APIRouter()
SUPPORTED_CHAT_ROUTES = frozenset({'workflow-builder', 'sample-manager', 'tool-generator'})
PROJECT_CHAT_ROUTES = frozenset({'workflow-builder', 'sample-manager'})


def _get_chat_session(session: Session, session_uid: str) -> ChatSessions:
    """Translate a missing chat session into the existing HTTP response."""
    try:
        return chat_service.get_session(session, session_uid)
    except chat_service.ChatSessionNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Chat session not found') from error


@router.post('', description='Create new chat session')
async def new_chat_session(session: SessionDep, current_user: UserDep) -> ChatSessionPublic:
    """Create new chat session."""
    session_uid = uuid7str()
    chat = ChatSessions(uid=uuid.UUID(session_uid), owner_id=current_user.id)
    return ChatSessionPublic.model_validate(chat_service.create_session(session, chat))


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
async def delete_chat_session(session: SessionDep, session_id: str, _current_user: UserDep) -> dict[str, str]:
    """Delete chat session."""
    try:
        chat_service.delete_session(session, session_id)
    except chat_service.ChatSessionNotFoundError as error:
        raise HTTPException(status_code=404, detail='对话不存在') from error

    history = PostgresMessageHistory(session, session_id)
    await history.aclear()

    return {'message': '对话已删除'}


@router.put('/{session_id}', description='更新对话信息')
async def update_chat_history(
    session: SessionDep,
    session_id: str,
    description: str,
    _current_user: UserDep,
) -> ChatSessionPublic:
    """Update chat description."""
    chat = ChatSessionUpdate(uid=session_id, description=description, update_time=datetime.now(UTC))
    try:
        return ChatSessionPublic.model_validate(chat_service.update_session(session, chat))
    except chat_service.ChatSessionNotFoundError as error:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Chat session not found') from error


@router.get('/{session_id}')
async def get_chat_session(
    session: SessionDep,
    session_id: str,
    _current_user: UserDep,
) -> ChatSessionPublic:
    """Get chat session."""
    return ChatSessionPublic.model_validate(_get_chat_session(session, session_id))


@router.get('/{session_id}/history', description='获取对话历史消息')
async def get_history(
    session: SessionDep,
    session_id: str,
    _current_user: UserDep,
) -> list[UIMessage]:
    """Get history messages for chat session."""
    history = PostgresMessageHistory(session, session_id)
    messages = await history.aget_messages()
    messages = rebuild_multimodal_messages(messages)
    return langchain_messages_to_ui_messages(messages)


@router.post('/{session_id}/completions', description='流式生成对话')
async def chat_completions(
    request: Request,
    session_id: str,
    response: ChatResponse,
    session: SessionDep,
    current_user: UserDep,
) -> StreamingResponse:
    """Generate chat stream."""
    # 验证session_id
    if not session_id:
        raise HTTPException(status_code=400, detail='会话ID不能为空')
    route = None
    if response.resume:
        chat_session = _get_chat_session(session, session_id)
        route = chat_session.route
    else:
        route, _ = _parse_slash_command(_latest_text(response.messages[-1]))

    if route not in SUPPORTED_CHAT_ROUTES:
        return StreamingResponse(
            generate_test_stream(),
            media_type='text/event-stream',
            headers={'x-vercel-ai-data-stream': 'v1'},
        )
    if route in PROJECT_CHAT_ROUTES and response.project_id is None:
        raise HTTPException(status_code=400, detail=f'{route} requires a project_id')

    streaming_response = StreamingResponse(
        _generate_chat_stream(request, response, session_id, session, current_user.id, response.project_id),
        media_type='text/event-stream',
    )

    # 设置 AI SDK 所需的响应头
    streaming_response.headers['x-vercel-ai-data-stream'] = 'v1'
    return streaming_response


def _latest_text(message: UIMessage) -> str:
    """Concatenate the text parts of the latest user message for slash-command parsing."""
    parts = getattr(message, 'parts', None) or []
    texts = [getattr(p, 'text', '') for p in parts if getattr(p, 'type', None) == 'text']
    return ''.join(texts).strip()


def _parse_slash_command(text: str) -> tuple[str | None, str]:
    """Return a supported leading slash command and the remaining user text."""
    stripped = text.lstrip()
    if not stripped.startswith('/'):
        return None, text

    parts = stripped[1:].split(maxsplit=1)
    route = parts[0].lower()
    if route not in SUPPORTED_CHAT_ROUTES:
        return None, text
    return route, parts[1] if len(parts) > 1 else ''


def _replace_human_text(message: HumanMessage, text: str) -> HumanMessage:
    """Replace command-bearing user text while preserving uploaded-file placeholders."""
    if isinstance(message.content, str):
        return message.model_copy(update={'content': text})

    content = list(message.content)
    for index, part in enumerate(content):
        if isinstance(part, dict) and part.get('type') == 'text' and not str(part.get('text', '')).startswith('<'):
            content[index] = {**part, 'text': text}
            break
    return message.model_copy(update={'content': content})


def _custom_event_data(payload: object) -> dict[str, object]:
    """Normalize LangGraph custom stream payloads for an AI SDK data part."""
    if isinstance(payload, dict):
        return payload
    return {'message': str(payload)}


async def _generate_chat_stream(
    request: Request,
    response: ChatResponse,
    session_id: str,
    session: Session,
    user_id: int,
    project_id: int | None,
) -> AsyncGenerator[str, None]:
    """Dispatch a chat turn to the cached top-level agent and stream its output.

    Threading model (see project notes): the durable conversation lives in
    ``PostgresMessageHistory`` (only the initial user input and final agent output are
    persisted). A fresh turn mints a new ``thread_id``; the Redis checkpointer bound to the
    cached agent is per-thread scratch used only for the interrupt -> resume cycle within a
    single turn.
    """
    history = PostgresMessageHistory(session, session_id)
    chat_session = _get_chat_session(session, session_id)

    # ------------------------------------------------------------------ resolve route + input
    if response.resume:
        # A resume continues the SAME thread, so it must dispatch to the SAME agent that
        # started it — read the route persisted when the thread was created.
        thread_id = chat_session.checkpoint_uid
        if thread_id is None:
            raise HTTPException(HTTP_404_NOT_FOUND, 'Checkpoint does not exist.')
        if chat_session.route is None:
            raise HTTPException(HTTP_404_NOT_FOUND, 'Session route is missing; cannot resume.')
        route = chat_session.route

        if response.approved:
            agent_input: Command | dict = Command(resume=True)
        else:
            # A rejected interrupt resumes the graph with the user's replacement input.
            human_input = await filter_multimedia_message(ui_input_to_human_message(response.messages[-1]))
            agent_input = Command(resume=human_input.content)
    else:
        human_input = await filter_multimedia_message(ui_input_to_human_message(response.messages[-1]))
        history_messages = await history.aget_messages()

        route, query = _parse_slash_command(_latest_text(response.messages[-1]))
        if route is None:
            return
        history_input = _replace_human_text(human_input, query)
        await history.aadd_messages([history_input])

        agent_text = f'<Require>\n{query}\n</Require>' if route == 'tool-generator' else query
        human_input = _replace_human_text(human_input, agent_text)

        messages = [human_input] if route == 'tool-generator' else [*history_messages, human_input]
        agent_input = {'messages': messages}
        thread_id = uuid7str()
        # Persist the route so a later resume turn reuses the same agent.
        chat_service.update_session(
            session,
            ChatSessionUpdate(uid=str(chat_session.uid), checkpoint_uid=uuid.UUID(thread_id), route=route),
        )

    # ------------------------------------------------------------------ select cached agent
    agents = request.app.state.chat_agents
    agent = agents[route]
    if route == 'tool-generator':
        context = CreateToolContext(
            user_id=user_id,
            skip_search=True,
            auto_save=True,
            test_run_id=str(thread_id),
        )
    else:
        if project_id is None:
            return
        project_path = Path(get_settings().persistent.storage.STORAGE_PATH) / 'proj' / str(project_id)
        context = ProjectContext(
            user_id=user_id,
            project_id=project_id,
            workdir=str(project_path.resolve().expanduser()),
        )
    config = RunnableConfig(configurable={'thread_id': str(thread_id)}, recursion_limit=sys.maxsize)

    # ------------------------------------------------------------------ stream
    yield start_message()
    try:
        async for chunk in agent.astream(
            agent_input,
            config,
            context=context,  # type: ignore[arg-type]
            stream_mode=['updates', 'custom'],
            version='v2',
            subgraphs=True,
        ):
            if chunk['type'] == 'custom':
                yield data_part('info', _custom_event_data(chunk['data']))
    except Exception as exc:  # surface any agent failure to the client stream
        yield error_part(str(exc))
        yield finish_message()
        yield done_marker()
        return

    # Record whether the turn paused on an interrupt so the next request knows to resume.
    state = await agent.aget_state(config)
    interrupted = bool(state.interrupts)
    chat_service.update_session(
        session,
        ChatSessionUpdate(uid=str(chat_session.uid), interrupted=interrupted),
    )

    if interrupted:
        for interrupt in state.interrupts:
            yield data_part('interrupt', {'value': interrupt.value})
    else:
        messages = state.values.get('messages', [])
        if route == 'tool-generator' and messages:
            final_response = AIMessage(content=messages[-1].content)
        else:
            final_response = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
        if final_response is not None and final_response.content:
            text_id = f'msg_{final_response.id or uuid.uuid4().hex}'
            yield text_start(text_id)
            yield text_delta(text_id, str(final_response.content))
            yield text_end(text_id)
            await history.aadd_messages([final_response])

    yield finish_message()
    yield done_marker()


async def generate_test_stream() -> AsyncGenerator[str, None]:
    """Emit a canned AI SDK stream for frontend protocol testing."""
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
