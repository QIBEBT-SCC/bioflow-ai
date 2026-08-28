import json

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class LogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # noqa: ANN001, ANN201, D102
        path = request.url.path
        normalized_path = path.rstrip('/')
        sensitive = normalized_path in {'/api/v1/auth/login', '/api/v1/auth/password'} or (
            request.method == 'POST' and normalized_path == '/api/v1/user'
        ) or (normalized_path.startswith('/api/v1/user/') and normalized_path.endswith('/password'))
        request_body = b'' if sensitive else await request.body()
        if sensitive:
            body = '[REDACTED]'
        elif request_body:
            try:
                body = json.dumps(json.loads(request_body), ensure_ascii=False)
            except json.JSONDecodeError:
                body = request_body.decode('utf-8', errors='replace')
        else:
            body = ''

        logger.debug(f'Received request: {body}')
        return await call_next(request)
