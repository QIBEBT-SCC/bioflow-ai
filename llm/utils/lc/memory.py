from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from sqlalchemy import Select, delete
from sqlmodel import Session, col, select

from app.models.chat import ChatMessages


class PostgresMessageHistory(BaseChatMessageHistory):
    def __init__(self, session: Session, session_id: str):
        self.session = session
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        query: Select = (
            select(ChatMessages).where(ChatMessages.session_id == self.session_id).order_by(col(ChatMessages.id).asc())
        )
        messages_data = self.session.exec(query).all()
        if messages_data:
            return messages_from_dict([m.message for m in messages_data])
        else:
            return []

    def add_message(self, message: BaseMessage) -> None:
        new_message = ChatMessages(session_id=self.session_id, message=message_to_dict(message))
        self.session.add(new_message)
        self.session.commit()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        try:
            new_objs = [ChatMessages(session_id=self.session_id, message=message_to_dict(m)) for m in messages]
            self.session.add_all(new_objs)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def clear(self) -> None:
        statement = delete(ChatMessages).where(col(ChatMessages.session_id) == self.session_id)
        self.session.exec(statement)
        self.session.commit()
