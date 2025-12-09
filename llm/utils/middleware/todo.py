from typing import Annotated, Awaitable, Callable, Literal, NotRequired, Optional

from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentMiddleware, ModelCallResult, OmitFromInput
from langchain_core.messages import RemoveMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """<To-DOList>
Use `write_todos` tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

- The `write_todos` tool should never be called multiple times in parallel.
- It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.
</To-DOList>"""


class Todo(BaseModel):
    """A single todo item with content and status."""

    index: Optional[int] = None
    content: str = Field(description='The content/description of the todo item, e.g. "Check samtools view command".')
    status: Literal['pending', 'in_progress', 'completed'] = Field(description='The current status of the todo item.')


class TodoState(AgentState):
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]


@tool(parse_docstring=True)
def add_todos(tasks: list[str], runtime: ToolRuntime):
    """Add new tasks to the todo list.

    Use this tool to initialize the plan or add discovered sub-tasks.

    Args:
        tasks: A list of task descriptions strings.
    """
    current_todos = runtime.state.get('todos', [])
    next_index = (current_todos[-1].index + 1) if current_todos and current_todos[-1].index is not None else 0

    new_todos = []
    for i, task in enumerate(tasks):
        new_todos.append(Todo(index=next_index + i, content=task, status='pending'))

    updated_todos = current_todos + new_todos
    return Command(
        update={
            'todos': updated_todos,
            'messages': [
                ToolMessage(f'Added {len(tasks)} tasks to todo list.', tool_call_id=runtime.tool_call_id),
            ],
        }
    )


@tool(parse_docstring=True)
def update_todo(index: int, status: Literal['pending', 'in_progress', 'completed'], runtime: ToolRuntime):
    """Update the status of a specific todo item.

    Use this to mark tasks as 'in_progress' or 'completed'.
    CRITICAL: Setting status to 'completed' will trigger a conversation history clear to free up context.

    Args:
        index: The index of the task to update.
        status: The new status.
    """
    current_todos = runtime.state.get('todos', [])

    # Find todo by index
    target_todo = next((t for t in current_todos if t.index == index), None)
    if not target_todo:
        return Command(
            update={
                'messages': [
                    ToolMessage(f'Error: Todo with index {index} not found.', tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    updated_todos = list(current_todos)
    for todo in updated_todos:
        if todo.index == index:
            todo.status = status
            break

    # If completing a task, clear history
    if status == 'completed':
        # 0: User Goal
        # ... logs ...
        # -1: AI calling this tool
        logger.debug(f'TODO complete: {target_todo.content}')
        remove_messages = [RemoveMessage(id=m.id) for m in runtime.state['messages'][1:-1]]

        pending_todos = [t for t in updated_todos if t.status == 'pending']
        in_progress_todos = [t for t in updated_todos if t.status == 'in_progress']
        remaining_count = len(pending_todos) + len(in_progress_todos)

        if remaining_count == 0:
            msg_content = f'Task {index} marked as {status}. Context cleared.\n\nAll tasks have been completed, and the mission can be ended.'
        else:
            completed_todos = [t for t in updated_todos if t.status == 'completed']
            display_todos = completed_todos + in_progress_todos
            if pending_todos:
                display_todos.append(pending_todos[0])

            display_todos.sort(key=lambda t: t.index if t.index is not None else 0)

            todo_list_str = '\n'.join(
                [f'  <todo index={t.index} content="{t.content}" status={t.status}>' for t in display_todos]
            )
            todo_msg = (
                f'<TODOs>\n{todo_list_str}\n  ...\n</TODOs>'
                if len(pending_todos) > 1
                else f'<TODOs>\n{todo_list_str}\n</TODOs>'
            )

            msg_content = f'{todo_msg}\n\nTask {index} marked as {status}. Context cleared.\n\n{remaining_count} todos remaining to be processed'

        update_messages = [
            *remove_messages,
            ToolMessage(
                msg_content,
                tool_call_id=runtime.tool_call_id,
            ),
        ]
    else:
        update_messages = [
            ToolMessage(
                f'Task {index}({target_todo.content}) status updated to {status}.',
                tool_call_id=runtime.tool_call_id,
            )
        ]

    return Command(update={'todos': updated_todos, 'messages': update_messages})


class TodoMiddleware(AgentMiddleware):
    state_schema = TodoState
    tools = [add_todos, update_todo]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        request.system_prompt = (
            request.system_prompt + '\n\n' + SYSTEM_PROMPT if request.system_prompt else SYSTEM_PROMPT
        )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        request.system_prompt = (
            request.system_prompt + '\n\n' + SYSTEM_PROMPT if request.system_prompt else SYSTEM_PROMPT
        )
        return await handler(request)
