from typing import Optional

from sqlmodel import Session, col, func, select

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
    if not db_chat:
        msg = f'Chat with uid {chat_update.uid} not found'
        raise ValueError(msg)

    update_data = chat_update.model_dump(exclude_unset=True)
    db_chat.sqlmodel_update(update_data)

    session.add(db_chat)
    session.commit()
    session.refresh(db_chat)
    return db_chat


def get_all(session: Session, owner_id: int, offset: int = 0, limit: int = 10) -> list[ChatSessions]:
    """Return a paginated list of chat sessions for a user."""
    query = (
        select(ChatSessions)
        .where(ChatSessions.owner_id == owner_id)
        .offset(offset)
        .limit(limit)
        .order_by(col(ChatSessions.update_time).desc())
    )
    return session.exec(query).all()


def count_all(session: Session, owner_id: int) -> int:
    """获取用户的聊天会话总数."""
    query = select(func.count(ChatSessions.uid)).where(ChatSessions.owner_id == owner_id)
    return session.exec(query).one()


def get(session: Session, uid: str) -> Optional[ChatSessions]:
    """Return a chat session by its uid, or None if not found."""
    query = select(ChatSessions).where(ChatSessions.uid == uid)
    return session.exec(query).one_or_none()
