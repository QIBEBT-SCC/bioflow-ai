"""Application services for chat sessions and persisted message history."""

from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.crud import chat as chat_crud
from app.models.chat import ChatMessages, ChatSessions
from app.schemas.chat import ChatSessionUpdate
from app.services._transaction import committing
from app.utils.identifiers import UUIDLike


class ChatServiceError(ValueError):
    """Base error for chat application operations."""


class ChatSessionNotFoundError(ChatServiceError):
    """Raised when a requested chat session does not exist."""


def get_session(session: Session, uid: UUIDLike) -> ChatSessions:
    """Return one chat session or raise a domain error."""
    chat = chat_crud.get(session, uid)
    if chat is None:
        raise ChatSessionNotFoundError
    return chat


def create_session(session: Session, chat: ChatSessions) -> ChatSessions:
    """Create and commit a chat session."""
    with committing(session):
        chat_crud.insert(session, chat)
    session.refresh(chat)
    return chat


def update_session(session: Session, data: ChatSessionUpdate) -> ChatSessions:
    """Update and commit a chat session."""
    with committing(session):
        chat = chat_crud.get(session, data.uid)
        if chat is None:
            raise ChatSessionNotFoundError
        chat_crud.update(session, chat, data)
    session.refresh(chat)
    return chat


def delete_session(session: Session, uid: UUIDLike) -> None:
    """Delete and commit a chat session."""
    with committing(session):
        chat = chat_crud.get(session, uid)
        if chat is None:
            raise ChatSessionNotFoundError
        chat_crud.delete(session, chat)


def list_messages(session: Session, session_id: str) -> Sequence[ChatMessages]:
    """Return a chat session's persisted messages."""
    return chat_crud.list_messages(session, session_id)


def add_message(session: Session, message: ChatMessages) -> None:
    """Persist one chat message."""
    add_messages(session, [message])


def add_messages(session: Session, messages: Sequence[ChatMessages]) -> None:
    """Persist a batch of chat messages atomically."""
    with committing(session):
        chat_crud.insert_messages(session, messages)


def clear_messages(session: Session, session_id: str) -> None:
    """Delete all persisted messages for a chat session."""
    with committing(session):
        chat_crud.delete_messages(session, session_id)
