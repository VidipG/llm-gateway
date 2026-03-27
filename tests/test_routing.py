import pytest
from unittest.mock import MagicMock

from app.config import Settings
from app.gateway.router import ConfigurationError, ModelRouter, UnknownModelError
from app.providers.base import Provider


def make_settings(**overrides) -> Settings:
    base = dict(
        gateway_api_key="test",
        anthropic_api_key="test",
        gemini_api_key="test",
    )
    return Settings(**{**base, **overrides})


def make_provider() -> Provider:
    return MagicMock(spec=Provider)


def test_resolves_known_model():
    provider = make_provider()
    router = ModelRouter(
        settings=make_settings(),
        providers={"gemini": provider},
    )
    resolved_provider, resolved_model = router.resolve("gemini-2.0-flash")
    assert resolved_provider is provider
    assert resolved_model == "gemini-2.0-flash"


def test_expands_alias():
    provider = make_provider()
    router = ModelRouter(
        settings=make_settings(),
        providers={"gemini": provider},
    )
    resolved_provider, resolved_model = router.resolve("fast")
    assert resolved_provider is provider
    assert resolved_model == "gemini-2.0-flash"


def test_raises_on_unknown_model():
    router = ModelRouter(
        settings=make_settings(),
        providers={"gemini": make_provider()},
    )
    with pytest.raises(UnknownModelError):
        router.resolve("gpt-4o")


def test_raises_on_missing_provider():
    router = ModelRouter(
        settings=make_settings(),
        providers={},
    )
    with pytest.raises(ConfigurationError):
        router.resolve("gemini-2.0-flash")
