from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from sqlalchemy.orm import Session

from app.models.chat import ChatMessages
from app.services import chat as chat_service


class PostgresMessageHistory(BaseChatMessageHistory):
    def __init__(self, session: Session, session_id: str):
        self.session = session
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        """Return the persisted chat messages as LangChain messages."""
        messages_data = chat_service.list_messages(self.session, self.session_id)
        if messages_data:
            return messages_from_dict([m.message for m in messages_data])
        return []

    def add_message(self, message: BaseMessage) -> None:
        """Persist one LangChain message."""
        new_message = ChatMessages(session_id=self.session_id, message=message_to_dict(message))
        chat_service.add_message(self.session, new_message)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist multiple LangChain messages atomically."""
        new_messages = [
            ChatMessages(session_id=self.session_id, message=message_to_dict(message)) for message in messages
        ]
        chat_service.add_messages(self.session, new_messages)

    def clear(self) -> None:
        """Delete all persisted messages for this chat session."""
        chat_service.clear_messages(self.session, self.session_id)
