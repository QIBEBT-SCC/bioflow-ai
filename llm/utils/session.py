from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlmodel import Session
from uuid_extensions import uuid7str

import app.crud.setting.llm_statistic as stat_crud
from app.core.session import engine


async def delete_checkpoint(ckpt: BaseCheckpointSaver, thread_id: str) -> None:
    """清除checkpoint, 但保留token统计."""
    await ckpt.adelete_thread(thread_id)

    new_id = uuid7str()
    with Session(engine) as session:
        stat_crud.refresh_thread_id(session, thread_id, new_id)
