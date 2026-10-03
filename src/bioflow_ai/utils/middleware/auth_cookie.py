from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_401_UNAUTHORIZED

from bioflow_ai.core.config import get_settings
from bioflow_ai.services.auth import CSRF_COOKIE, LEGACY_AUTH_COOKIES, SESSION_COOKIE


class AuthCookieCleanupMiddleware(BaseHTTPMiddleware):
    """Remove unusable browser credentials from every 401 response."""

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        """Delete current and legacy auth cookies when authentication fails."""
        response = await call_next(request)
        if response.status_code != HTTP_401_UNAUTHORIZED:
            return response

        domain = get_settings().security.COOKIE_DOMAIN
        for name in (SESSION_COOKIE, CSRF_COOKIE, *LEGACY_AUTH_COOKIES):
            response.delete_cookie(name, path='/', domain=domain)
        return response
