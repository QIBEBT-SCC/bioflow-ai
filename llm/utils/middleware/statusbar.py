from typing import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult

from app.models.tools import ImageConfig
from llm.utils.middleware.todo import Todo


class StatusBarMiddleware(AgentMiddleware):
    @staticmethod
    def _add_status_to_request(request: ModelRequest) -> ModelRequest:
        """添加状态信息到请求的最后一条消息中."""
        status = []

        if request.state.get('image'):  # type: ignore
            image_info: ImageConfig = request.state['image']  # type: ignore
            status.append(f'Current Image: {image_info.get_image()}')

        if request.state.get('todos'):  # type: ignore
            for todo in request.state['todos']:  # type: ignore
                todo: Todo
                if todo.status == 'in_progress':
                    status.append(f'Current To-DO: <todo index={todo.index}>{todo.content}</todo>')
                    break
                if todo.status == 'pending':
                    status.append(
                        f'No currently processed To-DO, the next To-DO in the queue is: <todo index={todo.index}>{todo.content}</todo>'
                    )
                    break

        if request.state.get('base_path'):  # type: ignore
            base_path: str = request.state['base_path']  # type: ignore
            status.append(f'Current base_path: "{base_path}"')

        if status:
            last_message = request.messages[-1]
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
        request = self._add_status_to_request(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        request = self._add_status_to_request(request)
        return await handler(request)
