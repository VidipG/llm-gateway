import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.gateway.errors import UnknownModelError
from app.gateway.semantic_router import SemanticRouter, _PROVIDER_ROUTES, _MODEL_ROUTES
from app.schemas.request import Message


def make_messages(*contents: str) -> list[Message]:
    return [Message(role="user", content=c) for c in contents]


def make_route_choice(name: str | None) -> MagicMock:
    choice = MagicMock()
    choice.name = name
    return choice


def make_router() -> SemanticRouter:
    """Return a SemanticRouter with all external dependencies mocked.

    _SemanticRouter is patched with side_effect so each call to
    _SemanticRouter(...) returns a distinct MagicMock instance.
    Without this, _provider_router and all _model_routers share the
    same object — assigning .acall on one overwrites the others.
    """
    with patch("app.gateway.semantic_router._AsyncFastEmbedEncoder"), \
         patch("app.gateway.semantic_router._SemanticRouter") as mock_sr_cls:
        mock_sr_cls.side_effect = lambda *a, **kw: MagicMock()
        return SemanticRouter()


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestSemanticRouterInit:

    def test_encoder_instantiated_inside_init(self):
        with patch("app.gateway.semantic_router._AsyncFastEmbedEncoder") as mock_encoder_cls, \
             patch("app.gateway.semantic_router._SemanticRouter") as mock_sr_cls:
            mock_sr_cls.side_effect = lambda *a, **kw: MagicMock()
            SemanticRouter()
            mock_encoder_cls.assert_called_once_with(name="BAAI/bge-small-en-v1.5")

    def test_model_routers_created_for_all_providers(self):
        router = make_router()
        assert set(router._model_routers.keys()) == set(_MODEL_ROUTES.keys())

    def test_encoder_not_instantiated_at_import_time(self):
        # Confirms no module-level _encoder variable exists.
        # If _AsyncFastEmbedEncoder were called at import time the attribute would
        # be present on the module regardless of patching.
        import app.gateway.semantic_router as mod
        assert not hasattr(mod, "_encoder")


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------

class TestSemanticRouterInitialize:

    @pytest.mark.asyncio
    async def test_aadd_called_on_provider_router(self):
        router = make_router()
        provider_aadd = AsyncMock()
        router._provider_router.aadd = provider_aadd
        for mr in router._model_routers.values():
            mr.aadd = AsyncMock()

        await router.initialize()

        provider_aadd.assert_called_once_with(_PROVIDER_ROUTES)

    @pytest.mark.asyncio
    async def test_aadd_called_on_every_model_router(self):
        router = make_router()
        router._provider_router.aadd = AsyncMock()
        model_aadd: dict[str, AsyncMock] = {}
        for p, mr in router._model_routers.items():
            mock = AsyncMock()
            mr.aadd = mock
            model_aadd[p] = mock

        await router.initialize()

        for provider, mock in model_aadd.items():
            mock.assert_called_once_with(_MODEL_ROUTES[provider])


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

class TestSemanticRouterResolve:

    def setup_router(
        self, provider_name: str, model_name: str, vector: list[float] | None = None
    ) -> tuple[SemanticRouter, list[float], AsyncMock, AsyncMock, dict[str, AsyncMock]]:
        """Wire a router to return the given provider and model.

        Returns the router, vector, and typed references to the mocks for
        each routing layer so tests can assert on them without going through
        the router attributes (which Pylance types as MethodType).
        """
        if vector is None:
            vector = [0.1, 0.2, 0.3]
        router = make_router()

        encoder_acall = AsyncMock(return_value=[vector])
        router._encoder.acall = encoder_acall

        provider_acall = AsyncMock(return_value=make_route_choice(provider_name))
        router._provider_router.acall = provider_acall

        model_acalls: dict[str, AsyncMock] = {}
        for p, mr in router._model_routers.items():
            mock = AsyncMock(
                return_value=make_route_choice(model_name if p == provider_name else None)
            )
            mr.acall = mock
            model_acalls[p] = mock

        return router, vector, encoder_acall, provider_acall, model_acalls

    @pytest.mark.asyncio
    async def test_returns_provider_and_model(self):
        router, *_ = self.setup_router("gemini", "gemini-2.0-flash")
        provider, model = await router.resolve(make_messages("fast response needed"))
        assert provider == "gemini"
        assert model == "gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_uses_last_message_as_query(self):
        router, _, encoder_acall, *_ = self.setup_router("anthropic", "claude-sonnet-4-6")
        await router.resolve(make_messages("first message", "last message"))
        encoder_acall.assert_called_once_with(["last message"])

    @pytest.mark.asyncio
    async def test_passes_same_vector_to_both_routing_layers(self):
        vector = [0.5, 0.6, 0.7]
        router, _, _, provider_acall, model_acalls = self.setup_router(
            "anthropic", "claude-sonnet-4-6", vector=vector
        )

        await router.resolve(make_messages("some query"))

        provider_acall.assert_called_once_with(vector=vector)
        model_acalls["anthropic"].assert_called_once_with(vector=vector)

    @pytest.mark.asyncio
    async def test_handles_provider_result_as_list(self):
        router = make_router()
        router._encoder.acall = AsyncMock(return_value=[[0.1]])
        router._provider_router.acall = AsyncMock(return_value=[make_route_choice("ollama")])
        router._model_routers["ollama"].acall = AsyncMock(return_value=make_route_choice("llama3.2"))

        provider, model = await router.resolve(make_messages("run this locally"))

        assert provider == "ollama"
        assert model == "llama3.2"

    @pytest.mark.asyncio
    async def test_handles_model_result_as_list(self):
        router = make_router()
        router._encoder.acall = AsyncMock(return_value=[[0.1]])
        router._provider_router.acall = AsyncMock(return_value=make_route_choice("anthropic"))
        router._model_routers["anthropic"].acall = AsyncMock(
            return_value=[make_route_choice("claude-opus-4-6")]
        )

        provider, model = await router.resolve(make_messages("deep multi-step reasoning"))

        assert provider == "anthropic"
        assert model == "claude-opus-4-6"

    @pytest.mark.asyncio
    async def test_raises_unknown_model_error_when_provider_unmatched(self):
        router = make_router()
        router._encoder.acall = AsyncMock(return_value=[[0.1]])
        router._provider_router.acall = AsyncMock(return_value=make_route_choice(None))

        with pytest.raises(UnknownModelError, match="provider"):
            await router.resolve(make_messages("some query"))

    @pytest.mark.asyncio
    async def test_raises_unknown_model_error_when_model_unmatched(self):
        router = make_router()
        router._encoder.acall = AsyncMock(return_value=[[0.1]])
        router._provider_router.acall = AsyncMock(return_value=make_route_choice("gemini"))
        router._model_routers["gemini"].acall = AsyncMock(return_value=make_route_choice(None))

        with pytest.raises(UnknownModelError, match="model"):
            await router.resolve(make_messages("some query"))

    @pytest.mark.asyncio
    async def test_model_error_is_distinct_from_provider_error(self):
        router = make_router()
        router._encoder.acall = AsyncMock(return_value=[[0.1]])
        router._provider_router.acall = AsyncMock(return_value=make_route_choice("anthropic"))
        router._model_routers["anthropic"].acall = AsyncMock(return_value=make_route_choice(None))

        with pytest.raises(UnknownModelError, match="model"):
            await router.resolve(make_messages("some query"))

    @pytest.mark.asyncio
    async def test_model_routers_not_called_when_provider_unmatched(self):
        router = make_router()
        router._encoder.acall = AsyncMock(return_value=[[0.1]])
        router._provider_router.acall = AsyncMock(return_value=make_route_choice(None))
        model_acalls: dict[str, AsyncMock] = {}
        for p, mr in router._model_routers.items():
            mock = AsyncMock()
            mr.acall = mock
            model_acalls[p] = mock

        with pytest.raises(UnknownModelError):
            await router.resolve(make_messages("some query"))

        for mock in model_acalls.values():
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_matched_provider_model_router_is_called(self):
        router, _, _, _, model_acalls = self.setup_router("gemini", "gemini-2.5-pro")

        await router.resolve(make_messages("complex reasoning task"))

        model_acalls["gemini"].assert_called_once()
        model_acalls["anthropic"].assert_not_called()
        model_acalls["ollama"].assert_not_called()
