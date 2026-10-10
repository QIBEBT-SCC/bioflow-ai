"""Summarize oversized history before sampling, using the todo message-reset pattern."""

from typing import Any, NotRequired

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

# Verbatim Codex templates: codex-rs/prompts/templates/compact/.
COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.
"""
SUMMARY_PREFIX = (
    'Another language model started to solve this problem and produced a summary of its thinking process. '
    'You also have access to the state of the tools that were used by that language model. '
    'Use this to build on the work that has already been done and avoid duplicating work. '
    'Here is the summary produced by the other language model, use the information in this summary '
    'to assist with your own analysis:'
)


class ContextCompactionState(AgentState):
    msg_to_keep: NotRequired[int]


class ContextCompactionMiddleware(AgentMiddleware[ContextCompactionState]):
    """Check history size, summarize once, then replace messages in graph state.

    Pass the agent's unbound model to build the summary chain. The default trigger
    is 90% of its profile's input window, following Codex; no window or configured
    limit means no automatic compaction. Like todo.py, retain the ``msg_to_keep``
    initial messages (default 1). This adapter uses Codex's prompt and prefix;
    history cleanup follows this repository's todo convention.
    """

    state_schema = ContextCompactionState

    def __init__(
        self,
        model: BaseChatModel,
        *,
        model_context_window: int | None = None,
        model_auto_compact_token_limit: int | None = None,
    ) -> None:
        super().__init__()
        window = model_context_window
        if window is None:
            window = (model.profile or {}).get('max_input_tokens')
        limit = model_auto_compact_token_limit
        if window is not None:
            default = window * 9 // 10
            limit = default if limit is None else min(limit, default)
        self.token_limit = limit
        prompt = ChatPromptTemplate.from_messages(
            [MessagesPlaceholder('messages'), HumanMessage(content=COMPACTION_PROMPT)]
        )
        self.summary_chain = prompt | model | StrOutputParser()

    def _should_compact(self, state: ContextCompactionState) -> bool:
        return (
            self.token_limit is not None
            and len(state['messages']) > state.get('msg_to_keep', 1)
            and count_tokens_approximately(state['messages']) >= self.token_limit
        )

    @staticmethod
    def _replace_messages(state: ContextCompactionState, summary: str) -> dict[str, Any]:
        return {
            'messages': [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *state['messages'][: state.get('msg_to_keep', 1)],
                HumanMessage(content=f'{SUMMARY_PREFIX}\n{summary}'),
            ],
        }

    def before_model(self, state: ContextCompactionState, runtime: Runtime) -> dict[str, Any] | None:
        """Persist the summary before the ordinary model call."""
        if not self._should_compact(state):
            return None
        summary = self.summary_chain.invoke({'messages': state['messages']})
        return self._replace_messages(state, summary)

    async def abefore_model(self, state: ContextCompactionState, runtime: Runtime) -> dict[str, Any] | None:
        """Summarize and replace messages asynchronously."""
        if not self._should_compact(state):
            return None
        summary = await self.summary_chain.ainvoke({'messages': state['messages']})
        return self._replace_messages(state, summary)
