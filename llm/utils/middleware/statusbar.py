from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult

if TYPE_CHECKING:
    from app.models.tools import ImageConfig
    from llm.utils.middleware.todo import Todo


class StatusBarMiddleware(AgentMiddleware):
    @staticmethod
    def _add_status_to_request(request: ModelRequest) -> ModelRequest:
        """添加状态信息到请求的最后一条消息中."""
        status = []

        if request.state.get('image'):
            image_info: ImageConfig = request.state['image']
            status.append(f'Current Image: {image_info.get_image()}')

        if request.state.get('todos'):
            for todo in request.state['todos']:
                todo: Todo
                if todo.status == 'in_progress':
                    status.append(f'Current To-DO: <todo index={todo.index}>{todo.content}</todo>')
                    break
                if todo.status == 'pending':
                    status.append(
                        f'No currently processed To-DO, the next To-DO in the queue is: <todo index={todo.index}>{todo.content}</todo>'
                    )
                    break

        if request.state.get('base_path'):
            base_path: str = request.state['base_path']
            status.append(f'Current base_path: "{base_path}"')

        if request.state.get('active_workflow_id'):
            active_workflow_id: str = request.state['active_workflow_id']
            workflow_drafts = request.state.get('workflow_drafts', {})
            active_draft = workflow_drafts.get(active_workflow_id)
            workflow_name = (
                active_draft.get('name', active_workflow_id)
                if isinstance(active_draft, dict)
                else getattr(active_draft, 'name', active_workflow_id)
            )
            status.append(f'Current Workflow Draft: <workflow id="{active_workflow_id}">{workflow_name}</workflow>')

        if status:
            last_message = request.messages[-1]
            # pyrefly: ignore [unsupported-operation]
            new_content = last_message.content + f'\n[{"; ".join(status)}]'
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
