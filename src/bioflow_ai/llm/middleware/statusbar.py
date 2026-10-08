from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult

if TYPE_CHECKING:
    from bioflow_ai.llm.middleware.todo import Todo
    from bioflow_ai.schemas.tool_config import ImageConfig


class StatusBarMiddleware(AgentMiddleware):
    @staticmethod
    def _add_status_to_request(request: ModelRequest) -> ModelRequest:
        """添加状态信息到请求的最后一条消息中."""
        status = []
        state = cast('dict[str, Any]', request.state)

        if state.get('image'):
            image_info: ImageConfig = state['image']
            status.append(f'Current Image: {image_info.get_image()}')

        if state.get('todos'):
            for todo in state['todos']:
                todo: Todo
                if todo.status == 'in_progress':
                    status.append(f'Current To-DO: <todo index={todo.index}>{todo.content}</todo>')
                    break
                if todo.status == 'pending':
                    status.append(
                        f'No currently processed To-DO, the next To-DO in the queue is: <todo index={todo.index}>{todo.content}</todo>'
                    )
                    break

        if state.get('base_path'):
            base_path: str = state['base_path']
            status.append(f'Current base_path: "{base_path}"')

        if state.get('active_workflow_id'):
            active_workflow_id: str = state['active_workflow_id']
            workflow_drafts = state.get('workflow_drafts', {})
            active_draft = workflow_drafts.get(active_workflow_id)
            workflow_name = (
                active_draft.get('name', active_workflow_id)
                if isinstance(active_draft, dict)
                else getattr(active_draft, 'name', active_workflow_id)
            )
            workflow_scope = (
                active_draft.get('execution_scope', 'sample_level')
                if isinstance(active_draft, dict)
                else getattr(active_draft, 'execution_scope', 'sample_level')
            )
            status.append(
                f'Current Workflow Draft: '
                f'<workflow id="{active_workflow_id}" scope="{workflow_scope}">{workflow_name}</workflow>'
            )

        if status:
            last_message = request.messages[-1]
            suffix = f'\n[{"; ".join(status)}]'
            new_content = (
                last_message.content + suffix
                if isinstance(last_message.content, str)
                else [*last_message.content, {'type': 'text', 'text': suffix}]
            )
            new_last_message = last_message.model_copy(update={'content': new_content}, deep=True)

            new_messages = [*request.messages[:-1], new_last_message]
            request = request.override(messages=new_messages)

        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Add status-bar context before a sync model call."""
        request = self._add_status_to_request(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Add status-bar context before an async model call."""
        request = self._add_status_to_request(request)
        return await handler(request)
