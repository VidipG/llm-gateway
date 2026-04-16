import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

import app.main


class TestCompletionsRoute:
    def test_endpoint_is_registered(self):
        routes = [r.path for r in app.main.app.routes]
        assert any("/v1/chat/completions" in path for path in routes)


class TestHealthRoute:
    def test_health_endpoint_is_registered(self):
        routes = [r.path for r in app.main.app.routes]
        assert any("/health" in path for path in routes)
