import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from redisvl.query.filter import Tag
from app.gateway.dispatcher import Dispatcher, _format_event, _format_error
from app.gateway.errors import UnknownModelError, ConfigurationError
from app.providers.base import ProviderError
from app.schemas.request import CompletionRequest, Message
from app.schemas.response import StreamChunk, UsageEvent, ErrorEvent


def make_cache(hit: str | None = None) -> MagicMock:
    """Return a mock SemanticCache. hit=None simulates a miss; hit=<text> simulates a hit."""
    cache = MagicMock()
    cache.acheck = AsyncMock(return_value=[{"response": hit}] if hit else [])
    cache.astore = AsyncMock()
    return cache


class TestDispatcherStream:
    @pytest.mark.asyncio
    async def test_yields_chunks_from_provider(self):
        mock_router = MagicMock()
        mock_provider = MagicMock()

        async def gen():
            yield StreamChunk(id="req-1", model="test", delta="Hello")
            yield StreamChunk(id="req-1", model="test", delta=" world")

        mock_provider.stream = lambda *args, **kwargs: gen()
        mock_router.resolve = AsyncMock(return_value=(mock_provider, "test-model"))

        dispatcher = Dispatcher(mock_router, make_cache())

        request = CompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="hi")],
        )

        results = [r async for r in dispatcher.stream(request, "req-1")]

        assert len(results) == 3
        assert "Hello" in results[0]
        assert "world" in results[1]
        assert "[DONE]" in results[2]

    @pytest.mark.asyncio
    async def test_yields_error_on_unknown_model(self):
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(side_effect=UnknownModelError("unknown model"))
        mock_router.providers = {"test": MagicMock()}

        dispatcher = Dispatcher(mock_router, make_cache())

        request = CompletionRequest(
            model="unknown",
            messages=[Message(role="user", content="hi")],
        )

        results = [r async for r in dispatcher.stream(request, "req-1")]

        assert len(results) == 1
        assert "error" in results[0]
        assert "404" in results[0]

    @pytest.mark.asyncio
    async def test_yields_error_on_configuration_error(self):
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(
            side_effect=ConfigurationError("missing provider")
        )

        dispatcher = Dispatcher(mock_router, make_cache())

        request = CompletionRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
        )

        results = [r async for r in dispatcher.stream(request, "req-1")]

        assert len(results) == 1
        assert "error" in results[0]
        assert "500" in results[0]


    @pytest.mark.asyncio
    async def test_serves_cached_response_on_hit(self):
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(return_value=(MagicMock(), "gemini-2.0-flash"))
        cache = make_cache(hit="cached answer")

        dispatcher = Dispatcher(mock_router, cache)

        request = CompletionRequest(
            model="gemini-2.0-flash",
            messages=[Message(role="user", content="hi")],
        )
        results = [r async for r in dispatcher.stream(request, "req-1")]

        assert any("cached answer" in r for r in results)
        assert "[DONE]" in results[-1]

    @pytest.mark.asyncio
    async def test_stores_response_in_cache_on_miss(self):
        mock_router = MagicMock()
        mock_provider = MagicMock()

        async def gen():
            yield StreamChunk(id="req-1", model="test-model", delta="Hello")
            yield StreamChunk(id="req-1", model="test-model", delta=" world")

        mock_provider.stream = lambda *a, **kw: gen()
        mock_router.resolve = AsyncMock(return_value=(mock_provider, "test-model"))
        cache = make_cache()

        dispatcher = Dispatcher(mock_router, cache)

        request = CompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="hi")],
        )
        [r async for r in dispatcher.stream(request, "req-1")]

        cache.astore.assert_called_once()
        _, kwargs = cache.astore.call_args
        assert kwargs["response"] == "Hello world"
        assert kwargs["metadata"] == {"model": "test-model"}


class TestFormatEvent:
    def test_stream_chunk_returns_sse_string(self):
        chunk = StreamChunk(id="req-1", model="claude-sonnet-4-6", delta="Hello")
        result = _format_event(chunk, "req-1")
        assert result == f"data: {chunk.model_dump_json()}\n\n"

    def test_usage_event_returns_none(self):
        event = UsageEvent(
            input_tokens=10,
            output_tokens=25,
            model="claude-sonnet-4-6",
            provider="anthropic",
        )
        result = _format_event(event, "req-1")
        assert result is None

    def test_unregistered_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unhandled event type"):
            _format_event(object(), "req-1")


class TestFormatError:
    def test_returns_sse_error_string(self):
        result = _format_error("something went wrong", 500)
        assert "event: error" in result
        assert "500" in result
        assert "something went wrong" in result


def _make_request(*contents: str, model: str = "test-model") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content=c) for c in contents],
    )


def _make_streaming_provider(*deltas: str, model: str = "test-model") -> MagicMock:
    provider = MagicMock()

    async def gen():
        for delta in deltas:
            yield StreamChunk(id="req-1", model=model, delta=delta)

    provider.stream = lambda *a, **kw: gen()
    return provider


class TestSemanticCacheLayer:
    """Edge-case coverage for the semantic cache integration in Dispatcher."""

    # ------------------------------------------------------------------
    # Cache hit behaviour
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_hit_does_not_call_provider(self):
        """When the cache has a hit the provider stream must never be called."""
        mock_router = MagicMock()
        mock_provider = MagicMock()
        stream_mock = MagicMock(return_value=iter([]))
        mock_provider.stream = stream_mock
        mock_router.resolve = AsyncMock(return_value=(mock_provider, "test-model"))
        cache = make_cache(hit="cached")

        results = [r async for r in Dispatcher(mock_router, cache).stream(_make_request("hi"), "req-1")]

        stream_mock.assert_not_called()
        assert not any("should not appear" in r for r in results)

    @pytest.mark.asyncio
    async def test_hit_chunk_has_finish_reason_stop(self):
        """Cached chunks must be emitted with finish_reason='stop'."""
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(return_value=(MagicMock(), "test-model"))

        results = [r async for r in Dispatcher(mock_router, make_cache(hit="answer")).stream(_make_request("hi"), "req-1")]

        data = json.loads(results[0].removeprefix("data: "))
        assert data["finish_reason"] == "stop"
        assert data["delta"] == "answer"

    @pytest.mark.asyncio
    async def test_hit_done_is_last_event(self):
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(return_value=(MagicMock(), "test-model"))

        results = [r async for r in Dispatcher(mock_router, make_cache(hit="x")).stream(_make_request("hi"), "req-1")]

        assert len(results) == 2
        assert "[DONE]" in results[-1]

    # ------------------------------------------------------------------
    # Cache miss + store behaviour
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_miss_stores_joined_deltas(self):
        """All deltas from a stream must be concatenated before storing."""
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(return_value=(_make_streaming_provider("foo", " bar", " baz"), "m"))
        cache = make_cache()

        [r async for r in Dispatcher(mock_router, cache).stream(_make_request("q"), "req-1")]

        _, kwargs = cache.astore.call_args
        assert kwargs["response"] == "foo bar baz"

    @pytest.mark.asyncio
    async def test_miss_stores_prompt_from_last_message(self):
        """astore prompt must match the last message in the conversation."""
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(return_value=(_make_streaming_provider("ok"), "m"))
        cache = make_cache()

        request = _make_request("first msg", "second msg", "last msg")
        [r async for r in Dispatcher(mock_router, cache).stream(request, "req-1")]

        _, kwargs = cache.astore.call_args
        assert kwargs["prompt"] == "last msg"

    @pytest.mark.asyncio
    async def test_miss_stores_resolved_model_in_metadata(self):
        """Metadata stored must contain the resolved (post-alias) model name."""
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(return_value=(_make_streaming_provider("ok", model="gemini-2.0-flash"), "gemini-2.0-flash"))
        cache = make_cache()

        [r async for r in Dispatcher(mock_router, cache).stream(_make_request("hi", model="fast"), "req-1")]

        _, kwargs = cache.astore.call_args
        assert kwargs["metadata"] == {"model": "gemini-2.0-flash"}

    @pytest.mark.asyncio
    async def test_empty_stream_does_not_store(self):
        """No astore call when the provider yields no text deltas."""
        mock_router = MagicMock()
        provider = MagicMock()

        async def gen():
            yield UsageEvent(input_tokens=5, output_tokens=0, model="m", provider="p")

        provider.stream = lambda *a, **kw: gen()
        mock_router.resolve = AsyncMock(return_value=(provider, "m"))
        cache = make_cache()

        [r async for r in Dispatcher(mock_router, cache).stream(_make_request("hi"), "req-1")]

        cache.astore.assert_not_called()

    @pytest.mark.asyncio
    async def test_provider_error_does_not_store(self):
        """astore must not be called if the provider raises a ProviderError mid-stream."""
        mock_router = MagicMock()
        provider = MagicMock()

        async def gen():
            yield StreamChunk(id="req-1", model="m", delta="partial")
            raise ProviderError("upstream failed", provider_name="test", status_code=503)

        provider.stream = lambda *a, **kw: gen()
        mock_router.resolve = AsyncMock(return_value=(provider, "m"))
        cache = make_cache()

        [r async for r in Dispatcher(mock_router, cache).stream(_make_request("hi"), "req-1")]

        cache.astore.assert_not_called()

    # ------------------------------------------------------------------
    # acheck arguments
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_acheck_uses_last_message_as_prompt(self):
        """Cache lookup prompt is the content of the last message only."""
        mock_router = MagicMock()
        provider = MagicMock()

        async def gen():
            return
            yield  # noqa: unreachable — makes this an async generator

        provider.stream = lambda *a, **kw: gen()
        mock_router.resolve = AsyncMock(return_value=(provider, "m"))
        cache = make_cache()

        request = _make_request("ignored", "also ignored", "target prompt")
        [r async for r in Dispatcher(mock_router, cache).stream(request, "req-1")]

        _, kwargs = cache.acheck.call_args
        assert kwargs["prompt"] == "target prompt"

    @pytest.mark.asyncio
    async def test_acheck_uses_resolved_model_for_filter(self):
        """Cache filter must be built from the resolved model, not the alias."""
        mock_router = MagicMock()
        provider = MagicMock()

        async def gen():
            return
            yield

        provider.stream = lambda *a, **kw: gen()
        # Simulates alias "fast" → "gemini-2.0-flash"
        mock_router.resolve = AsyncMock(return_value=(provider, "gemini-2.0-flash"))
        cache = make_cache()

        [r async for r in Dispatcher(mock_router, cache).stream(_make_request("hi", model="fast"), "req-1")]

        _, kwargs = cache.acheck.call_args
        expected = Tag("model") == "gemini-2.0-flash"
        assert str(kwargs["filter_expression"]) == str(expected)

    @pytest.mark.asyncio
    async def test_same_prompt_different_models_get_different_filters(self):
        """Two models checking the cache with the same prompt use distinct filters."""
        async def gen():
            return
            yield

        cache_a = make_cache()
        cache_b = make_cache()

        for cache, resolved_model in [(cache_a, "model-alpha"), (cache_b, "model-beta")]:
            mock_router = MagicMock()
            provider = MagicMock()
            provider.stream = lambda *a, **kw: gen()
            mock_router.resolve = AsyncMock(return_value=(provider, resolved_model))
            [r async for r in Dispatcher(mock_router, cache).stream(_make_request("same prompt"), "req-1")]

        _, kwargs_a = cache_a.acheck.call_args
        _, kwargs_b = cache_b.acheck.call_args
        assert str(kwargs_a["filter_expression"]) != str(kwargs_b["filter_expression"])

    # ------------------------------------------------------------------
    # Miss → store → hit (sequential flow)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_miss_then_store_then_hit(self):
        """First request is a miss (stores); second request returns the stored value."""
        stored: dict[str, str] = {}

        async def fake_acheck(prompt, filter_expression):
            return [{"response": stored["r"]}] if stored else []

        async def fake_astore(prompt, response, metadata):
            stored["r"] = response

        cache = MagicMock()
        cache.acheck = fake_acheck
        cache.astore = AsyncMock(side_effect=fake_astore)

        mock_router = MagicMock()
        call_count = 0

        async def gen():
            yield StreamChunk(id="r", model="m", delta="live answer")

        def make_stream(*a, **kw):
            nonlocal call_count
            call_count += 1
            return gen()

        provider = MagicMock()
        provider.stream = make_stream
        mock_router.resolve = AsyncMock(return_value=(provider, "m"))

        dispatcher = Dispatcher(mock_router, cache)
        request = _make_request("what is ai")

        # First call — miss, provider invoked, response stored
        results1 = [r async for r in dispatcher.stream(request, "req-1")]
        assert any("live answer" in r for r in results1)
        assert call_count == 1

        # Second call — hit, provider must NOT be invoked again
        results2 = [r async for r in dispatcher.stream(request, "req-2")]
        assert any("live answer" in r for r in results2)
        assert call_count == 1  # provider was not called a second time

    # ------------------------------------------------------------------
    # Same model, same prompt — two different stored responses
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_same_model_different_cached_responses(self):
        """Two separate cache entries for different prompts with same model stay independent."""
        cache_hello = make_cache(hit="Hello response")
        cache_bye = make_cache(hit="Goodbye response")

        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(return_value=(MagicMock(), "test-model"))

        results_hello = [r async for r in Dispatcher(mock_router, cache_hello).stream(_make_request("hello"), "req-1")]
        results_bye = [r async for r in Dispatcher(mock_router, cache_bye).stream(_make_request("bye"), "req-2")]

        assert any("Hello response" in r for r in results_hello)
        assert not any("Goodbye response" in r for r in results_hello)
        assert any("Goodbye response" in r for r in results_bye)
        assert not any("Hello response" in r for r in results_bye)
