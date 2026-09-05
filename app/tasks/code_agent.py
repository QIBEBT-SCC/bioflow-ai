"""Celery entry points for isolated ACP sessions."""

import asyncio
import uuid

from script.worker import celery_app


@celery_app.task
def execute_code_agent_session(session_id: str) -> None:
    """Own one long-lived ACP adapter on the isolated queue."""
    from app.services.code_agent.runner import run_session

    asyncio.run(run_session(uuid.UUID(session_id)))


@celery_app.task
def login_codex_account(login_id: str) -> None:
    """Own one administrator device-code login on the shared ACP queue."""
    from app.services.code_agent.runner import run_codex_login

    asyncio.run(run_codex_login(uuid.UUID(login_id)))
