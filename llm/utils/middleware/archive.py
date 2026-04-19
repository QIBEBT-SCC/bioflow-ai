from typing import Any

import duckdb
from langchain.agents.middleware import AgentMiddleware
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime
from loguru import logger

from llm.states import TestRunContext

DB_PATH = 'presets/build/develop.db'


class SaveMessagesMiddleware(AgentMiddleware):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def after_agent(self, state: MessagesState, runtime: Runtime[TestRunContext]) -> dict[str, Any] | None:  # type: ignore  # noqa: D102
        if runtime.context.save_message:
            message_dicts = [m.model_dump_json() for m in state['messages']]
            con = duckdb.connect(DB_PATH)
            con.execute("""
                CREATE TABLE IF NOT EXISTS test_run (
                    question VARCHAR,
                    answer   VARCHAR,
                    agent    VARCHAR,
                    test_run_id VARCHAR,
                    messages VARCHAR[]
                )
            """)
            con.execute(
                'INSERT INTO test_run VALUES (?, ?, ?, ?, ?)',
                [
                    state['messages'][0].content,
                    state['messages'][-1].content,
                    self.agent_name,
                    runtime.context.test_run_id,
                    message_dicts,
                ],
            )
            con.close()
            logger.debug(f'Saved new conversation record, containing {len(message_dicts)} conversations.')

    async def aafter_agent(self, state: MessagesState, runtime: Runtime[TestRunContext]) -> dict[str, Any] | None:  # type: ignore  # noqa: D102
        return self.after_agent(state, runtime)
