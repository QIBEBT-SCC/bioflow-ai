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
Use the todo tools for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.

## Guidelines:
- Use `add_todos` for complex objectives to break down larger tasks into smaller, manageable steps
- Never call `add_todos` multiple times in parallel
- Always mark todos as 'completed' immediately after finishing a step - do not batch multiple completions
- Dynamically adjust the todo list as you go - use the `before` parameter to insert prerequisite tasks when you discover dependencies
- When you find that a task requires preparation, insert new tasks before it using `add_todos` with the `before` parameter
</To-DOList>"""


class Todo(BaseModel):
    """A single todo item with content and status."""

    index: Optional[int] = None
    content: str = Field(description='The content/description of the todo item, e.g. "Check samtools view command".')
    status: Literal['pending', 'in_progress', 'completed'] = Field(description='The current status of the todo item.')


class TodoState(AgentState):
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]


@tool(parse_docstring=True)
def add_todos(tasks: list[str], before: Optional[int], runtime: ToolRuntime):
    """Add new tasks to the todo list or insert tasks at a specific position

    - To append tasks to the end: call with `tasks` parameter only (before=None)
    - To insert tasks before an existing task: specify the `before` parameter with the target task index
    - Use this when discovering new sub-tasks or when you need to add prerequisite tasks before an existing task

    Args:
        tasks: A list of task descriptions strings.
        before: Optional index of the task before which to insert new tasks. If None, tasks are appended to the end.
    """
    writer = runtime.stream_writer
    writer('Update task list')

    current_todos = runtime.state.get('todos', [])

    # If no 'before' parameter, append to the end (original behavior)
    if before is None:
        next_index = (current_todos[-1].index + 1) if current_todos and current_todos[-1].index is not None else 0
        new_todos = []
        for i, task in enumerate(tasks):
            new_todos.append(Todo(index=next_index + i, content=task, status='pending'))

        updated_todos = current_todos + new_todos
        message = f'Added {len(tasks)} tasks to todo list.'
    else:
        # Find the position of the task with the specified index
        insert_position = None
        for i, todo in enumerate(current_todos):
            if todo.index == before:
                insert_position = i
                break

        if insert_position is None:
            return Command(
                update={
                    'messages': [
                        ToolMessage(f'Error: Todo with index {before} not found.', tool_call_id=runtime.tool_call_id)
                    ]
                }
            )

        # Create new todos, they will take indices starting from 'before'
        new_todos = []
        for i, task in enumerate(tasks):
            new_todos.append(Todo(index=before + i, content=task, status='pending'))

        # Insert new todos at the specified position
        updated_todos = current_todos[:insert_position] + new_todos + current_todos[insert_position:]

        # Reindex all todos from the insertion point onwards
        for i in range(insert_position + len(new_todos), len(updated_todos)):
            updated_todos[i].index = updated_todos[i - 1].index + 1 if updated_todos[i - 1].index is not None else i

        message = f'Inserted {len(tasks)} tasks before task {before}.'

    return Command(
        update={
            'todos': updated_todos,
            'messages': [
                ToolMessage(message, tool_call_id=runtime.tool_call_id),
            ],
        }
    )


@tool('update_todo', parse_docstring=True)
def update_todo_state(index: int, status: Literal['pending', 'in_progress', 'completed'], runtime: ToolRuntime):
    """Update the status of a specific todo item.

    - Use this to mark tasks as 'in_progress' when you start working on them
    - Mark tasks as 'completed' when finished
    CRITICAL: Setting status to 'completed' will trigger a conversation history clear to free up context.

    Args:
        index: The index of the task to update.
        status: The new status.
    """
    writer = runtime.stream_writer
    writer('Update task list')

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


@tool('update_todo', parse_docstring=True)
def update_todo_content(index: int, content: str, runtime: ToolRuntime):
    """Update the content of a specific todo item.

    Use this to modify the description or content of an existing task.

    Args:
        index: The index of the task to update.
        content: The new content/description for the todo item.
    """
    writer = runtime.stream_writer
    writer('Update task list')

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

    # Store old content for message
    old_content = target_todo.content

    updated_todos = list(current_todos)
    for todo in updated_todos:
        if todo.index == index:
            todo.content = content
            break

    return Command(
        update={
            'todos': updated_todos,
            'messages': [
                ToolMessage(
                    f'Task {index} content updated from "{old_content}" to "{content}".',
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(parse_docstring=True)
def delete_todo(indexs: list[int], runtime: ToolRuntime):
    """Delete specific todo items.

    Use this to remove existing tasks from the todo list.
    Deleting todos will cause all remaining task indices to be reassigned sequentially starting from 0.

    Args:
        indexs: A list of task indices to delete.
    """
    writer = runtime.stream_writer
    writer('Update task list')

    current_todos = runtime.state.get('todos', [])

    if not current_todos:
        return Command(
            update={'messages': [ToolMessage('Error: Todo list is empty.', tool_call_id=runtime.tool_call_id)]}
        )

    # Find todos to delete
    deleted_contents = []
    not_found_indices = []

    for idx in indexs:
        target_todo = next((t for t in current_todos if t.index == idx), None)
        if target_todo:
            deleted_contents.append(f'Task {idx}: "{target_todo.content}"')
        else:
            not_found_indices.append(idx)

    # If some indices were not found, return error
    if not_found_indices:
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        f'Error: Todo(s) with index {", ".join(map(str, not_found_indices))} not found.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    # Remove todos with specified indices
    updated_todos = [t for t in current_todos if t.index not in indexs]

    # Reassign indices sequentially starting from 0
    for i, todo in enumerate(updated_todos):
        todo.index = i

    # Create success message
    deleted_list = '\n'.join(deleted_contents)
    message = f'Deleted {len(indexs)} task(s):\n{deleted_list}\n\nAll remaining tasks have been reindexed.'

    return Command(
        update={
            'todos': updated_todos,
            'messages': [ToolMessage(message, tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def view_todos(runtime: ToolRuntime):
    """View the current todo list.

    Generally, you can see the current to-do list directly from the status bar; call this tool ONLY when you need to review it or require the full task list for other reasons.
    """
    current_todos = runtime.state.get('todos', [])
    if not current_todos:
        content = 'The todo list is currently empty.'
    else:
        todo_list_str = '\n'.join(
            [f'  <todo index={t.index} content="{t.content}" status={t.status}>' for t in current_todos]
        )
        content = f'<TODOs>\n{todo_list_str}\n</TODOs>'

    return content


class TodoMiddleware(AgentMiddleware):
    state_schema = TodoState
    tools = [add_todos, update_todo_state, view_todos]

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


class BuildTodoMiddleware(AgentMiddleware):
    state_schema = TodoState
    tools = [add_todos, update_todo_content, delete_todo, view_todos]


class InitedTodoMiddleware(AgentMiddleware):
    tools = [update_todo_state, view_todos]

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
