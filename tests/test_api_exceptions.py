import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.exceptions import (
    register_exception_handlers,
    _make_handler,
    _ERROR_STATUS_MAP,
)
from app.gateway.errors import ConfigurationError, UnknownModelError
from app.providers.base import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)


class TestRegisterExceptionHandlers:
    def test_registers_all_handlers(self):
        app = FastAPI()
        register_exception_handlers(app)

        for exc_class in _ERROR_STATUS_MAP:
            handler = app.exception_handlers.get(exc_class)
            assert handler is not None


class TestMakeHandler:
    @pytest.mark.asyncio
    async def test_returns_json_response_with_status_code(self):
        handler = _make_handler(400)
        exc = InvalidRequestError("invalid request", provider_name="test")

        request = MagicMock(spec=Request)
        response = await handler(request, exc)

        assert response.status_code == 400
        assert b'"error":"invalid request"' in response.body

    @pytest.mark.asyncio
    async def test_includes_provider_in_response(self):
        handler = _make_handler(502)
        exc = ProviderError("provider failed", provider_name="anthropic")

        request = MagicMock(spec=Request)
        response = await handler(request, exc)

        assert b'"provider":"anthropic"' in response.body


class TestErrorStatusMap:
    def test_authentication_error_maps_to_401(self):
        assert _ERROR_STATUS_MAP[AuthenticationError] == 401

    def test_rate_limit_error_maps_to_429(self):
        assert _ERROR_STATUS_MAP[RateLimitError] == 429

    def test_invalid_request_error_maps_to_400(self):
        assert _ERROR_STATUS_MAP[InvalidRequestError] == 400

    def test_provider_timeout_error_maps_to_504(self):
        assert _ERROR_STATUS_MAP[ProviderTimeoutError] == 504

    def test_provider_unavailable_error_maps_to_502(self):
        assert _ERROR_STATUS_MAP[ProviderUnavailableError] == 502

    def test_unknown_model_error_maps_to_404(self):
        assert _ERROR_STATUS_MAP[UnknownModelError] == 404

    def test_configuration_error_maps_to_500(self):
        assert _ERROR_STATUS_MAP[ConfigurationError] == 500
