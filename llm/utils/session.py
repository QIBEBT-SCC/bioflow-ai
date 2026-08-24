"""LangGraph checkpoint and runtime-session helpers."""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.orm import Session
from uuid_extensions import uuid7str

from app.core.session import engine
from app.services import llm_setting as setting_service


def workflow_thread_id(config: RunnableConfig) -> str:
    """Extract the workflow checkpoint identity used to correlate manager calls."""
    thread_id = config.get('configurable', {}).get('thread_id')
    if thread_id is None or not str(thread_id).strip():
        msg = 'Catalog generation requires a LangGraph thread_id'
        raise RuntimeError(msg)
    return str(thread_id)


async def delete_checkpoint(ckpt: BaseCheckpointSaver, thread_id: str) -> None:
    """清除checkpoint, 但保留token统计."""
    await ckpt.adelete_thread(thread_id)

    new_id = uuid7str()
    with Session(engine) as session:
        setting_service.refresh_statistic_thread_id(session, thread_id, new_id)
