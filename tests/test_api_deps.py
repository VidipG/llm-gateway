import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from app.api.deps import verify_api_key, get_dispatcher


class TestVerifyApiKey:
    def test_rejects_invalid_api_key(self):
        with patch("app.api.deps.get_settings") as mock_settings:
            mock_settings.return_value.gateway_api_key = "valid-key"
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key("wrong-key")
            assert exc_info.value.status_code == 401
            assert "Invalid API key" in exc_info.value.detail

    def test_accepts_valid_api_key(self):
        with patch("app.api.deps.get_settings") as mock_settings:
            mock_settings.return_value.gateway_api_key = "valid-key"
            result = verify_api_key("valid-key")
            assert result is None


class TestGetDispatcher:
    def test_returns_dispatcher_from_request_state(self):
        mock_dispatcher = MagicMock()
        mock_request = MagicMock()
        mock_request.app.state.dispatcher = mock_dispatcher

        result = get_dispatcher(mock_request)
        assert result is mock_dispatcher
