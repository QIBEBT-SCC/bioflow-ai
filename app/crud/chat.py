from collections.abc import Sequence
from datetime import datetime

from fastapi import HTTPException
from sqlmodel import Session, col, select
from starlette.status import HTTP_404_NOT_FOUND

from app.models.chat import ChatSessions
from app.schemas.chat import ChatSessionUpdate


def insert(session: Session, chat: ChatSessions) -> ChatSessions:
    """Insert a new chat session into the database."""
    session.add(chat)
    session.commit()
    session.refresh(chat)

    return chat


def delete(session: Session, chat: ChatSessions) -> None:
    """Delete a chat session from the database."""
    session.delete(chat)
    session.commit()


def update(session: Session, chat_update: ChatSessionUpdate) -> ChatSessions:
    """Update an existing chat session with new data."""
    db_chat = get(session, chat_update.uid)
    if db_chat is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Chat session not found')

    update_data = chat_update.model_dump(exclude_unset=True)
    db_chat.sqlmodel_update(update_data)

    session.add(db_chat)
    session.commit()
    session.refresh(db_chat)
    return db_chat


def get(session: Session, uid: str) -> ChatSessions:
    """Return a chat session by its uid, or None if not found."""
    query = select(ChatSessions).where(ChatSessions.uid == uid)
    chat = session.exec(query).one_or_none()
    if chat is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Chat session not found')
    return chat


def get_page(
    session: Session,
    owner_id: int,
    cursor: datetime | None = None,
    limit: int = 10,
) -> Sequence[ChatSessions]:
    """Return a cursor-paginated slice of a user's chat sessions, newest first.

    Uses ``update_time`` as the cursor so that scroll-based lazy loading stays stable
    even when a session's ``update_time`` changes mid-scroll (offset pagination would
    duplicate or skip rows in that case). Fetches ``limit + 1`` rows so the caller can
    tell whether more data is available without a separate count query.
    """
    query = select(ChatSessions).where(ChatSessions.owner_id == owner_id)
    if cursor is not None:
        query = query.where(col(ChatSessions.update_time) < cursor)
    query = query.order_by(col(ChatSessions.update_time).desc()).limit(limit + 1)
    return session.exec(query).all()
