import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import Settings
from app.gateway.errors import UnknownModelError, ConfigurationError
from app.gateway.router import Router
from app.providers.base import Provider
from app.schemas.request import Message


def make_settings(**overrides) -> Settings:
    return Settings.model_construct(
        gateway_api_key="test",
        anthropic_api_key="test",
        gemini_api_key="test",
        **overrides,
    )


def make_provider() -> Provider:
    return MagicMock(spec=Provider)


def make_router(providers: dict, **kwargs) -> Router:
    with patch("app.gateway.router.SemanticRouter"):
        return Router(settings=make_settings(), providers=providers, **kwargs)


@pytest.mark.asyncio
async def test_resolves_known_model():
    provider = make_provider()
    router = make_router({"gemini": provider})
    resolved_provider, resolved_model = await router.resolve("gemini-2.0-flash", [])
    assert resolved_provider is provider
    assert resolved_model == "gemini-2.0-flash"


@pytest.mark.asyncio
async def test_expands_alias():
    provider = make_provider()
    router = make_router({"gemini": provider})
    resolved_provider, resolved_model = await router.resolve("fast", [])
    assert resolved_provider is provider
    assert resolved_model == "gemini-2.0-flash"


@pytest.mark.asyncio
async def test_raises_on_unknown_model():
    router = make_router({"gemini": make_provider()})
    with pytest.raises(UnknownModelError):
        await router.resolve("gpt-4o", [])


@pytest.mark.asyncio
async def test_raises_on_missing_provider():
    router = make_router({})
    with pytest.raises(ConfigurationError):
        await router.resolve("gemini-2.0-flash", [])


@pytest.mark.asyncio
async def test_auto_delegates_to_semantic_router():
    provider = make_provider()
    semantic_router = MagicMock()
    semantic_router.resolve = AsyncMock(return_value=("gemini", "gemini-2.0-flash"))
    router = make_router({"gemini": provider}, semantic_router=semantic_router)

    resolved_provider, resolved_model = await router.resolve("auto", [Message(role="user", content="fast query")])

    semantic_router.resolve.assert_called_once()
    assert resolved_provider is provider
    assert resolved_model == "gemini-2.0-flash"


@pytest.mark.asyncio
async def test_auto_falls_back_to_static_when_semantic_router_unavailable():
    provider = make_provider()
    router = make_router({"gemini": provider}, semantic_router=None)
    resolved_provider, resolved_model = await router.resolve("auto", [Message(role="user", content="any query")])
    assert resolved_provider is provider
    assert resolved_model == "gemini-2.0-flash"
