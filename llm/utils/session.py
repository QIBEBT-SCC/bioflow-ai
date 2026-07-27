from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.orm import Session
from uuid_extensions import uuid7str

from app.core.session import engine
from app.services import llm_setting as setting_service


async def delete_checkpoint(ckpt: BaseCheckpointSaver, thread_id: str) -> None:
    """清除checkpoint, 但保留token统计."""
    await ckpt.adelete_thread(thread_id)

    new_id = uuid7str()
    with Session(engine) as session:
        setting_service.refresh_statistic_thread_id(session, thread_id, new_id)
