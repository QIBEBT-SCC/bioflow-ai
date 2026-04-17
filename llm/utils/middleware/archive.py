from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime
from loguru import logger

from llm.states import TestRunContext


class SaveMessagesMiddleware(AgentMiddleware):
    def __init__(self, agent_name: str, dataset_file: str = 'presets/build/test_run.parquet'):
        self.agent_name = agent_name
        self.dataset_file = dataset_file

    def after_agent(self, state: MessagesState, runtime: Runtime[TestRunContext]) -> dict[str, Any] | None:  # type: ignore
        if runtime.context.save_message:
            import polars as pl

            message_dicts = [m.model_dump_json() for m in state['messages']]
            new_data = {
                'question': state['messages'][0].content,
                'answer': state['messages'][-1].content,
                'agent': self.agent_name,
                'test_run_id': runtime.context.test_run_id,
                'messages': message_dicts,
            }
            df_new = pl.DataFrame([new_data])

            if Path(self.dataset_file).exists():
                df_existing = pl.read_parquet(self.dataset_file)
                df_existing = df_existing.vstack(df_new)
                df_existing.write_parquet(self.dataset_file, compression='snappy')
            else:
                df_new.write_parquet(self.dataset_file, compression='snappy')

            logger.debug(f'Saved new conversation record, containing {len(message_dicts)} conversations.')

    async def aafter_agent(self, state: MessagesState, runtime: Runtime[TestRunContext]) -> dict[str, Any] | None:  # type: ignore
        return self.after_agent(state, runtime)
