import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.gateway.errors import ConfigurationError, ContentPolicyError, PromptInjectionError, UnknownModelError
from app.providers.base import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

_ERROR_STATUS_MAP: dict[type[Exception], int] = {
    AuthenticationError: 401,
    RateLimitError: 429,
    InvalidRequestError: 400,
    ProviderTimeoutError: 504,
    ProviderUnavailableError: 502,
    ProviderError: 502,
    UnknownModelError: 404,
    ConfigurationError: 500,
    PromptInjectionError: 400,
    ContentPolicyError: 400,
}


def register_exception_handlers(app: FastAPI) -> None:
    for exc_class, status_code in _ERROR_STATUS_MAP.items():
        app.add_exception_handler(exc_class, _make_handler(status_code))


def _make_handler(status_code: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception: %s", exc)
        body: dict = {"error": str(exc)}
        if provider := getattr(exc, "provider_name", None):
            body["provider"] = provider
        return JSONResponse(status_code=status_code, content=body)
    return handler
