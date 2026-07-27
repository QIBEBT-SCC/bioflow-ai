from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat import ChatMessages, ChatSessions
from app.schemas.chat import ChatSessionUpdate
from app.utils.identifiers import UUIDLike, as_uuid


def insert(session: Session, chat: ChatSessions) -> ChatSessions:
    """Stage a new chat session."""
    session.add(chat)
    session.flush()
    session.refresh(chat)
    return chat


def delete(session: Session, chat: ChatSessions) -> None:
    """Stage deletion of a chat session."""
    session.delete(chat)
    session.flush()


def update(session: Session, chat: ChatSessions, chat_update: ChatSessionUpdate) -> ChatSessions:
    """Stage updates to an existing chat session."""
    update_data = chat_update.model_dump(exclude={'uid'}, exclude_unset=True)
    for field, value in update_data.items():
        setattr(chat, field, value)
    session.flush()
    session.refresh(chat)
    return chat


def get(session: Session, uid: UUIDLike) -> ChatSessions | None:
    """Return a chat session by UID, or ``None`` when absent."""
    return session.scalar(select(ChatSessions).where(ChatSessions.uid == as_uuid(uid)))


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
        query = query.where(ChatSessions.update_time < cursor)
    query = query.order_by(ChatSessions.update_time.desc()).limit(limit + 1)
    return session.scalars(query).all()


def list_messages(session: Session, session_id: str) -> Sequence[ChatMessages]:
    """Return persisted messages for a chat session in insertion order."""
    statement = select(ChatMessages).where(ChatMessages.session_id == session_id).order_by(ChatMessages.id.asc())
    return session.scalars(statement).all()


def insert_messages(session: Session, messages: Sequence[ChatMessages]) -> None:
    """Stage a batch of persisted chat messages."""
    session.add_all(messages)
    session.flush()


def delete_messages(session: Session, session_id: str) -> None:
    """Stage deletion of every persisted message for a chat session."""
    session.execute(sa_delete(ChatMessages).where(ChatMessages.session_id == session_id))
