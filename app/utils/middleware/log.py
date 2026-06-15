import json

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class LogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # noqa: ANN001, ANN201, D102
        request_body = await request.body()
        if request_body:
            try:
                body = json.dumps(json.loads(request_body), ensure_ascii=False)
            except json.JSONDecodeError:
                body = request_body.decode('utf-8', errors='replace')
        else:
            body = ''

        logger.debug(f'Received request: {body}')
        return await call_next(request)
