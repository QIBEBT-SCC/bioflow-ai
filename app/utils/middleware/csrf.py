import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_SAFE_METHODS = {'GET', 'HEAD', 'OPTIONS', 'TRACE'}
_AUTH_PATHS = {'/api/v1/auth/token', '/api/v1/auth/refresh', '/api/v1/user'}


class CSRFMiddleware(BaseHTTPMiddleware):
    """Double-submit cookie CSRF protection.

    On login/refresh the backend sets a non-HttpOnly `csrf_token` cookie.
    Subsequent mutating requests must include an `X-CSRF-Token` header
    whose value matches the cookie value.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        if request.method in _SAFE_METHODS or request.url.path in _AUTH_PATHS:
            return await call_next(request)

        cookie_token = request.cookies.get('csrf_token')
        header_token = request.headers.get('X-CSRF-Token')

        if not cookie_token or not header_token or not secrets.compare_digest(cookie_token, header_token):
            return JSONResponse(status_code=403, content={'detail': 'CSRF token missing or invalid'})

        return await call_next(request)
